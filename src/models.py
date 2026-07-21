"""Unified model components, domain generalization modules, and TrustOCT assembler."""

import os
import random
from typing import Tuple, Dict, Union
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# Relative import of heads from same src directory
from src.heads import SoftmaxHead, EvidentialHead


# =====================================================================
# 1. Backbones
# =====================================================================

class ResNet50Backbone(nn.Module):
    """ResNet50 backbone extracting intermediate layer 3 and layer 4 features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)

        return layer3_out, layer4_out


# =====================================================================
# 2. Attention and Fusion Layers
# =====================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(concat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, gate_channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(gate_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class MultiScaleFusion(nn.Module):
    """Concatenation and channel projection of multi-scale CNN layer outputs."""

    def __init__(self, layer3_channels: int = 1024, layer4_channels: int = 2048, out_channels: int = 512):
        super().__init__()
        in_channels = layer3_channels + layer4_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, layer3_out: torch.Tensor, layer4_out: torch.Tensor) -> torch.Tensor:
        h4, w4 = layer4_out.shape[2], layer4_out.shape[3]
        layer3_resized = F.interpolate(layer3_out, size=(h4, w4), mode="bilinear", align_corners=False)
        fused = torch.cat([layer3_resized, layer4_out], dim=1)
        out = self.proj(fused)
        out = self.bn(out)
        return self.relu(out)


# =====================================================================
# 3. Domain Generalization Layers
# =====================================================================

class MixStyle(nn.Module):
    """MixStyle module (Zhou et al., ICLR 2021) for style statistics blending."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x

        batch_size = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        x_norm = (x - mu) / sig

        # Shuffle styles
        perm = torch.randperm(batch_size).to(x.device)
        mu_shuffled = mu[perm]
        sig_shuffled = sig[perm]

        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        lmda = beta_dist.sample((batch_size, 1, 1, 1)).to(x.device)

        mu_mixed = lmda * mu + (1.0 - lmda) * mu_shuffled
        sig_mixed = lmda * sig + (1.0 - lmda) * sig_shuffled

        return x_norm * sig_mixed + mu_mixed


def compute_covariance(x: torch.Tensor) -> torch.Tensor:
    """Compute the covariance matrix of input features."""
    n = x.size(0)
    if n <= 1:
        return torch.zeros((x.size(1), x.size(1)), device=x.device)
    mean = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - mean
    covariance = torch.matmul(x_centered.t(), x_centered) / (n - 1)
    return covariance


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Deep CORAL loss."""
    d = source.size(1)
    source_cov = compute_covariance(source)
    target_cov = compute_covariance(target)
    loss = torch.mean((source_cov - target_cov) ** 2) / (4 * d * d)
    return loss


class CoralFeatureAlignment(nn.Module):
    """Wrapper storing features for CORAL covariance alignment."""

    def __init__(self):
        super().__init__()
        self.last_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            self.last_features = torch.mean(x, dim=[2, 3])
        else:
            self.last_features = x
        return x


# =====================================================================
# 4. TrustOCT Unified Class
# =====================================================================

class TrustOCT(nn.Module):
    """Unified framework architecture combining swappable component layers."""

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
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_cbam = use_cbam
        self.dg_type = dg_type.lower()
        self.head_type = head_type.lower()

        # 1. Backbone wrapper
        if backbone_name.lower() == "resnet50":
            self.backbone = ResNet50Backbone(pretrained=pretrained)
            self.layer3_channels = 1024
            self.layer4_channels = 2048
        else:
            raise NotImplementedError(f"Backbone '{backbone_name}' not implemented.")

        # 2. Domain Generalization
        if self.dg_type == "mixstyle":
            self.dg = MixStyle(p=dg_p, alpha=dg_alpha)
        elif self.dg_type == "coral":
            self.dg = CoralFeatureAlignment()
        else:
            self.dg = nn.Identity()

        # 3. Multi-scale fusion
        if self.use_multiscale:
            self.fusion = MultiScaleFusion(
                layer3_channels=self.layer3_channels,
                layer4_channels=self.layer4_channels,
                out_channels=512
            )
            self.feature_channels = 512
        else:
            self.feature_channels = self.layer4_channels

        # 4. Attention
        if self.use_cbam:
            self.attention = CBAM(gate_channels=self.feature_channels)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 5. Classifier Heads
        if self.head_type == "softmax":
            self.head = SoftmaxHead(self.feature_channels, num_classes, dropout_prob)
        elif self.head_type == "edl":
            self.head = EvidentialHead(self.feature_channels, num_classes, dropout_prob)
        else:
            raise ValueError(f"Unknown head type '{head_type}'")

    def forward(self, x: torch.Tensor):
        layer3_out, layer4_out = self.backbone(x)

        if self.dg_type == "mixstyle":
            layer3_out = self.dg(layer3_out)

        if self.use_multiscale:
            features = self.fusion(layer3_out, layer4_out)
        else:
            features = layer4_out

        if self.use_cbam:
            features = self.attention(features)

        if self.dg_type == "coral":
            features = self.dg(features)

        pooled = self.pool(features)
        pooled = torch.flatten(pooled, start_dim=1)
        return self.head(pooled)


# =====================================================================
# 5. Registry Builder
# =====================================================================

def build_model(model_config_path: str) -> nn.Module:
    """Build the TrustOCT model dynamically from config file."""
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config not found at {model_config_path}")
        
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    backbone_name = config.get("backbone", "resnet50")
    pretrained = config.get("pretrained", True)
    use_multiscale = (config.get("feature_module", "multiscale") == "multiscale")
    use_cbam = (config.get("attention", "cbam") == "cbam")
    
    dg_type = config.get("domain_generalization", "mixstyle")
    mixstyle_cfg = config.get("mixstyle", {})
    dg_p = mixstyle_cfg.get("mix_prob", 0.5)
    dg_alpha = mixstyle_cfg.get("alpha", 0.1)

    head_type = config.get("head", "edl")
    num_classes = config.get("num_classes", 4)
    dropout_prob = config.get("dropout", 0.5)

    model = TrustOCT(
        backbone_name=backbone_name,
        pretrained=pretrained,
        use_multiscale=use_multiscale,
        use_cbam=use_cbam,
        dg_type=dg_type,
        head_type=head_type,
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        dg_p=dg_p,
        dg_alpha=dg_alpha
    )

    print(f"Successfully compiled TrustOCT Model V1.0:")
    print(f"  • Backbone:     {backbone_name} (Pretrained: {pretrained})")
    print(f"  • Feature Fusion: {config.get('feature_module', 'multiscale')}")
    print(f"  • Attention:     {config.get('attention', 'cbam')}")
    print(f"  • Generalization: {dg_type}")
    print(f"  • Decision Head:  {head_type}")
    print(f"  • Classes:       {num_classes}")

    return model
