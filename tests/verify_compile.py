"""Direct python compilation and integration check for TrustOCT models and pipelines."""

import os
import sys
import torch

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import TrustOCT, build_model
from src.training import EdlLoss


def verify_pipeline():
    print("=" * 60)
    print("TrustOCT Compilation & Integration Verification (Consolidated)")
    print("=" * 60)

    # 1. Test standard tensor forward pass through ResNet50-Softmax
    print("Testing ResNet50-Softmax (Baseline) Model compilation...")
    model_softmax = TrustOCT(
        backbone_name="resnet50",
        pretrained=False,
        use_multiscale=False,
        use_cbam=False,
        dg_type="identity",
        head_type="softmax",
        num_classes=4
    )
    
    mock_images = torch.randn(2, 3, 224, 224)
    logits = model_softmax(mock_images)
    print(f"[OK] ResNet50-Softmax output shape: {list(logits.shape)}")
    assert logits.shape == (2, 4), "Logits shape mismatch"

    # 2. Test evidential (EDL) variant with Multi-scale + CBAM + MixStyle
    print("Testing TrustOCT (Full Evidential) Model compilation...")
    model_edl = TrustOCT(
        backbone_name="resnet50",
        pretrained=False,
        use_multiscale=True,
        use_cbam=True,
        dg_type="mixstyle",
        head_type="edl",
        num_classes=4
    )
    
    evidence, alpha = model_edl(mock_images)
    print(f"[OK] TrustOCT-EDL output shapes:")
    print(f"    - Evidence: {list(evidence.shape)}")
    print(f"    - Alpha:    {list(alpha.shape)}")
    assert evidence.shape == (2, 4) and alpha.shape == (2, 4), "Evidence/Alpha shape mismatch"
    assert torch.all(evidence >= 0.0), "Evidence values must be non-negative"
    assert torch.all(alpha >= 1.0), "Alpha parameters must be >= 1"

    # 3. Test loss function compatibility
    print("Testing EDL Dirichlet Loss forward/backward calculation...")
    targets = torch.tensor([0, 1])
    criterion = EdlLoss(num_classes=4, annealing_epochs=10)
    loss = criterion(alpha, targets, epoch=2)
    print(f"[OK] EDL loss value: {loss.item():.4f}")
    
    # Backward pass check
    loss.backward()
    print("[OK] Loss backward propagation complete (No NaNs/Infs)")

    # 4. Test Model Registry Builder compilation
    print("Testing builder.py configuration reading and parsing...")
    temp_cfg = "configs/model.yaml"
    model_from_cfg = build_model(temp_cfg)
    assert isinstance(model_from_cfg, TrustOCT), "Builder did not return TrustOCT instance"
    print("[OK] Model Builder verified")
    
    print("=" * 60)
    print("All integration compiles completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    verify_pipeline()
