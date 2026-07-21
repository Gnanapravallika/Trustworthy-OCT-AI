"""Model and component builder registry for TrustOCT framework."""

import os
import sys
from typing import Dict, Union
import torch.nn as nn
import yaml

try:
    from src.models.trustoct import TrustOCT
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.models.trustoct import TrustOCT


def load_yaml(path: str) -> dict:
    """Load configuration from a YAML path.

    Args:
        path: Path to the YAML file.

    Returns:
        Dict representing YAML content.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_config_path: str) -> nn.Module:
    """Build the TrustOCT model dynamically from config file.

    Args:
        model_config_path: Path to the model configuration YAML file.

    Returns:
        Instantiated and compiled TrustOCT model (nn.Module).
    """
    config = load_yaml(model_config_path)

    backbone_name = config.get("backbone", "resnet50")
    pretrained = config.get("pretrained", True)
    
    # Configure modular components
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
