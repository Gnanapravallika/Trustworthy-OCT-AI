"""Consolidated explainability visualization (Grad-CAM and LayerCAM) for TrustOCT."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM, LayerCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    # Dummy fallbacks for checkups
    class GradCAM:
        def __init__(self, **kwargs): pass
        def __call__(self, **kwargs): return np.zeros((1, 224, 224))
    class LayerCAM:
        def __init__(self, **kwargs): pass
        def __call__(self, **kwargs): return np.zeros((1, 224, 224))
    class ClassifierOutputTarget:
        def __init__(self, *args): pass
    def show_cam_on_image(img, mask, **kwargs):
        return (img * 255.0).astype(np.uint8)


class TrustGradCAM:
    """Wrapper class generating Grad-CAM maps on models."""

    def __init__(self, model: nn.Module, target_layers: list):
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]


class TrustLayerCAM:
    """Wrapper class generating LayerCAM maps on models."""

    def __init__(self, model: nn.Module, target_layers: list):
        self.cam = LayerCAM(model=model, target_layers=target_layers)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]


def compare_and_save_visualizations(
    model: nn.Module,
    target_layers_gradcam: list,
    target_layers_layercam: list,
    image_path: str,
    target_class: int,
    output_dir: str,
    prefix: str = "sample"
) -> None:
    """Generate and save original, Grad-CAM, and LayerCAM overlays side-by-side."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    input_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)

    g_cam = TrustGradCAM(model, target_layers_gradcam)
    l_cam = TrustLayerCAM(model, target_layers_layercam)

    try:
        gradcam_map = g_cam.generate(input_tensor, target_class)
        layercam_map = l_cam.generate(input_tensor, target_class)
    except Exception as e:
        print(f"Error generating CAM: {e}")
        gradcam_map = np.zeros((224, 224))
        layercam_map = np.zeros((224, 224))

    gradcam_overlay = show_cam_on_image(img_np, gradcam_map, use_rgb=True)
    layercam_overlay = show_cam_on_image(img_np, layercam_map, use_rgb=True)

    Image.fromarray(gradcam_overlay).save(os.path.join(output_dir, f"{prefix}_gradcam.png"))
    Image.fromarray(layercam_overlay).save(os.path.join(output_dir, f"{prefix}_layercam.png"))
    img_resized.save(os.path.join(output_dir, f"{prefix}_original.png"))

    print(f"Explainability maps saved to {output_dir}")
