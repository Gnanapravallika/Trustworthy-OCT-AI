"""Unified TrustOCT architecture model."""

import os
import sys
import torch
import torch.nn as nn

try:
    from src.models.backbone import build_backbone
    from src.models.multiscale import MultiScaleFusion
    from src.models.cbam import CBAM
    from src.domain_generalization.identity import IdentityDG
    from src.domain_generalization.mixstyle import MixStyle
    from src.domain_generalization.coral import CoralFeatureAlignment
    from src.models.heads.softmax import SoftmaxHead
    from src.models.heads.edl import EvidentialHead
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.models.backbone import build_backbone
    from src.models.multiscale import MultiScaleFusion
    from src.models.cbam import CBAM
    from src.domain_generalization.identity import IdentityDG
    from src.domain_generalization.mixstyle import MixStyle
    from src.domain_generalization.coral import CoralFeatureAlignment
    from src.models.heads.softmax import SoftmaxHead
    from src.models.heads.edl import EvidentialHead


class TrustOCT(nn.Module):
    """The unified TrustOCT framework architecture (configurable like LEGO blocks)."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        use_multiscale: bool = True,
        use_cbam: bool = True,
        dg_type: str = "mixstyle",
        head_type: str = "edl",
        num_classes: int = 4,
        dropout_prob: float = 0.5,
        dg_p: float = 0.5,
        dg_alpha: float = 0.1
    ):
        """Initialize TrustOCT model.

        Args:
            backbone_name: CNN/Transformer backbone model type.
            pretrained: Load ImageNet weights.
            use_multiscale: Enable Multi-Scale Layer 3 + Layer 4 Feature Fusion.
            use_cbam: Enable CBAM attention module.
            dg_type: Generalization layer type ('mixstyle', 'coral', 'identity').
            head_type: Prediction head output type ('edl', 'softmax').
            num_classes: Number of disease targets.
            dropout_prob: Dropout probability.
            dg_p: MixStyle activation probability.
            dg_alpha: MixStyle beta distribution parameter.
        """
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_cbam = use_cbam
        self.dg_type = dg_type.lower()
        self.head_type = head_type.lower()

        # 1. Backbone selection
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)

        # Determine feature channels based on backbone layers
        # For ResNet50: Layer 3 has 1024 channels, Layer 4 has 2048 channels
        self.layer3_channels = 1024
        self.layer4_channels = 2048

        # 2. Domain Generalization module inside backbone blocks
        if self.dg_type == "mixstyle":
            self.dg = MixStyle(p=dg_p, alpha=dg_alpha)
        elif self.dg_type == "coral":
            self.dg = CoralFeatureAlignment()
        else:
            self.dg = IdentityDG()

        # 3. Feature modules: MultiScale Feature Fusion
        if self.use_multiscale:
            self.fusion = MultiScaleFusion(
                layer3_channels=self.layer3_channels,
                layer4_channels=self.layer4_channels,
                out_channels=512
            )
            self.feature_channels = 512
        else:
            self.feature_channels = self.layer4_channels

        # 4. Attention blocks: CBAM
        if self.use_cbam:
            self.attention = CBAM(gate_channels=self.feature_channels)

        # 5. Pooling to get 1D vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 6. Classifier Heads
        if self.head_type == "softmax":
            self.head = SoftmaxHead(self.feature_channels, num_classes, dropout_prob)
        elif self.head_type == "edl":
            self.head = EvidentialHead(self.feature_channels, num_classes, dropout_prob)
        else:
            raise ValueError(f"Unknown head type '{head_type}'")

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input image tensor of shape [B, 3, H, W].

        Returns:
            Varies depending on head:
                - If head_type == 'softmax': Logits tensor of shape [B, num_classes]
                - If head_type == 'edl': Tuple of (evidence, alpha) tensors
        """
        # Extract features
        layer3_out, layer4_out = self.backbone(x)

        # Apply Domain Generalization style mixing if selected
        if self.dg_type == "mixstyle":
            # Apply statistics mixing on intermediate Layer 3 representation
            layer3_out = self.dg(layer3_out)

        # Apply Multi-scale feature fusion
        if self.use_multiscale:
            features = self.fusion(layer3_out, layer4_out)
        else:
            features = layer4_out

        # Apply attention gate
        if self.use_cbam:
            features = self.attention(features)

        # Store features if using CORAL covariance alignment
        if self.dg_type == "coral":
            features = self.dg(features)

        # Perform global average pooling
        pooled = self.pool(features)
        pooled = torch.flatten(pooled, start_dim=1)

        # Pass through classification/evidential head
        return self.head(pooled)
