"""Bilateral filter, CLAHE and normalization transformations for TrustOCT."""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


class BilateralFilter(ImageOnlyTransform):
    """Custom Albumentations wrapper for OpenCV Bilateral Filter."""

    def __init__(
        self,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
        always_apply: bool = False,
        p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.dtype != np.uint8:
            img_uint8 = (img * 255.0).astype(np.uint8)
            denoised = cv2.bilateralFilter(img_uint8, self.d, self.sigma_color, self.sigma_space)
            return (denoised / 255.0).astype(np.float32)
        return cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)

    def get_transform_init_args_names(self):
        return ("d", "sigma_color", "sigma_space")


def get_train_transforms(config: dict) -> A.Compose:
    """Build train transformation pipeline based on YAML configuration."""
    preprocessing_cfg = config.get("preprocessing", {})
    augmentations_cfg = config.get("augmentations", {})
    resize_h, resize_w = preprocessing_cfg.get("resize", [224, 224])
    
    transform_list = []

    # 1. Bilateral Denoising
    bf_cfg = preprocessing_cfg.get("bilateral_filter", {})
    if bf_cfg.get("enabled", True):
        transform_list.append(
            BilateralFilter(
                d=bf_cfg.get("d", 9),
                sigma_color=bf_cfg.get("sigma_color", 75.0),
                sigma_space=bf_cfg.get("sigma_space", 75.0),
                p=1.0
            )
        )

    # 2. Contrast Enhancement CLAHE
    clahe_cfg = preprocessing_cfg.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        transform_list.append(
            A.CLAHE(
                clip_limit=clahe_cfg.get("clip_limit", 2.0),
                tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
                p=1.0
            )
        )

    # 3. Resize
    transform_list.append(A.Resize(height=resize_h, width=resize_w))

    # 4. Augmentations
    rot_cfg = augmentations_cfg.get("random_rotate", {})
    transform_list.append(
        A.Rotate(limit=rot_cfg.get("limit", 15), p=rot_cfg.get("p", 0.5))
    )

    hf_cfg = augmentations_cfg.get("horizontal_flip", {})
    transform_list.append(A.HorizontalFlip(p=hf_cfg.get("p", 0.5)))

    bc_cfg = augmentations_cfg.get("random_brightness_contrast", {})
    transform_list.append(
        A.RandomBrightnessContrast(
            brightness_limit=bc_cfg.get("brightness_limit", 0.1),
            contrast_limit=bc_cfg.get("contrast_limit", 0.1),
            p=bc_cfg.get("p", 0.5)
        )
    )

    # 5. Normalization
    norm_cfg = augmentations_cfg.get("normalize", {})
    transform_list.append(
        A.Normalize(
            mean=norm_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=norm_cfg.get("std", [0.229, 0.224, 0.225]),
            p=1.0
        )
    )

    # 6. Tensor conversion
    transform_list.append(ToTensorV2())

    return A.Compose(transform_list)


def get_val_transforms(config: dict) -> A.Compose:
    """Build validation/testing transformation pipeline (no augmentations)."""
    preprocessing_cfg = config.get("preprocessing", {})
    augmentations_cfg = config.get("augmentations", {})
    resize_h, resize_w = preprocessing_cfg.get("resize", [224, 224])
    
    transform_list = []

    # 1. Bilateral Denoising
    bf_cfg = preprocessing_cfg.get("bilateral_filter", {})
    if bf_cfg.get("enabled", True):
        transform_list.append(
            BilateralFilter(
                d=bf_cfg.get("d", 9),
                sigma_color=bf_cfg.get("sigma_color", 75.0),
                sigma_space=bf_cfg.get("sigma_space", 75.0),
                p=1.0
            )
        )

    # 2. Contrast Enhancement CLAHE
    clahe_cfg = preprocessing_cfg.get("clahe", {})
    if clahe_cfg.get("enabled", True):
        transform_list.append(
            A.CLAHE(
                clip_limit=clahe_cfg.get("clip_limit", 2.0),
                tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
                p=1.0
            )
        )

    # 3. Resize
    transform_list.append(A.Resize(height=resize_h, width=resize_w))

    # 4. Normalization
    norm_cfg = augmentations_cfg.get("normalize", {})
    transform_list.append(
        A.Normalize(
            mean=norm_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=norm_cfg.get("std", [0.229, 0.224, 0.225]),
            p=1.0
        )
    )

    # 5. Tensor conversion
    transform_list.append(ToTensorV2())

    return A.Compose(transform_list)
