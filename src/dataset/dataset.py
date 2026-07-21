"""Consolidated dataset loaders, verification utilities, and statistics calculations for TrustOCT."""

import json
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union
import yaml
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. Constants
# =====================================================================

CLASS_NAMES: List[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpeg", ".jpg", ".png")
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)


# =====================================================================
# 2. Dataset Classes
# =====================================================================

class BaseOCTDataset(Dataset, ABC):
    """Abstract Base Class for all OCT datasets."""

    def __init__(
        self,
        base_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.base_path = base_path
        self.transform = transform
        self.target_transform = target_transform
        
        self.filepaths: List[str] = []
        self.labels: List[int] = []
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(filepath).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {filepath}: {e}")

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0

        if self.target_transform:
            label = self.target_transform(label)

        return image_tensor, label


class KermanyDataset(BaseOCTDataset):
    """Kermany OCT2017 Dataset loader implementation."""

    def _load_metadata(self) -> None:
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Kermany path does not exist at {self.base_path}")

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.base_path, class_name)
            if not os.path.exists(class_dir):
                continue

            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if os.path.isdir(file_path):
                    continue

                _, ext = os.path.splitext(filename)
                if ext.lower() in SUPPORTED_EXTENSIONS:
                    self.filepaths.append(file_path)
                    self.labels.append(CLASS_TO_INDEX[class_name])

        print(f"Loaded Kermany split at {self.base_path} with {len(self.filepaths)} images.")


# =====================================================================
# 3. Loader & Factory
# =====================================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """Create a standard PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last
    )


def get_dataset_and_loader(
    split: str,
    dataset_config_path: str,
    augmentation_config_path: str
) -> Tuple[KermanyDataset, DataLoader]:
    """Factory loading transforms and datasets from YAML configs."""
    # Lazy imports to avoid circular dependencies
    from src.preprocessing import get_train_transforms, get_val_transforms

    with open(dataset_config_path, "r") as f:
        dataset_cfg = yaml.safe_load(f)
    with open(augmentation_config_path, "r") as f:
        aug_cfg = yaml.safe_load(f)

    combined_cfg = {**dataset_cfg, **aug_cfg}

    dataset_name = dataset_cfg.get("dataset", {}).get("name", "Kermany OCT2017")
    paths = dataset_cfg.get("paths", {})
    loader_cfg = dataset_cfg.get("loader", {})

    split_path = paths.get(split)
    if not split_path:
        raise ValueError(f"Split '{split}' path is not defined in dataset config.")

    if split == "train":
        transforms = get_train_transforms(combined_cfg)
        shuffle = loader_cfg.get("shuffle", True)
    else:
        transforms = get_val_transforms(combined_cfg)
        shuffle = False

    if "kermany" in dataset_name.lower():
        dataset = KermanyDataset(base_path=split_path, transform=transforms)
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' not yet supported.")

    loader = create_dataloader(
        dataset=dataset,
        batch_size=loader_cfg.get("batch_size", 32),
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=loader_cfg.get("pin_memory", True),
        shuffle=shuffle,
        drop_last=(split == "train")
    )

    return dataset, loader


# =====================================================================
# 4. Dataset Verification
# =====================================================================

def verify_folder_structure(train_path: str, val_path: str, test_path: str) -> List[str]:
    missing_dirs = []
    for name, path in [("Train", train_path), ("Validation", val_path), ("Test", test_path)]:
        if not os.path.exists(path):
            missing_dirs.append(path)
    return missing_dirs


def verify_class_folders(base_path: str, classes: List[str]) -> Dict[str, bool]:
    class_status = {}
    for c in classes:
        class_path = os.path.join(base_path, c)
        class_status[c] = os.path.exists(class_path)
    return class_status


def verify_images(
    base_path: str, classes: List[str], extensions: Tuple[str, ...]
) -> Tuple[int, Dict[str, int], List[str], List[str]]:
    total_images = 0
    class_counts = {c: 0 for c in classes}
    corrupt_images = []
    unsupported_files = []

    for c in classes:
        class_path = os.path.join(base_path, c)
        if not os.path.exists(class_path):
            continue

        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            if os.path.isdir(file_path):
                continue

            _, ext = os.path.splitext(filename)
            if ext.lower() not in extensions:
                unsupported_files.append(file_path)
                continue

            total_images += 1
            class_counts[c] += 1

            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                corrupt_images.append(file_path)

    return total_images, class_counts, corrupt_images, unsupported_files


def generate_dataset_report(config: dict) -> dict:
    paths = config.get("paths", {})
    train_path = paths.get("train", "datasets/raw/Kermany/train")
    val_path = paths.get("val", "datasets/raw/Kermany/val")
    test_path = paths.get("test", "datasets/raw/Kermany/test")

    classes = config.get("classes", CLASS_NAMES)
    extensions = tuple(config.get("image", {}).get("extensions", SUPPORTED_EXTENSIONS))

    missing_dirs = verify_folder_structure(train_path, val_path, test_path)
    
    report = {
        "dataset_name": config.get("dataset", {}).get("name", "Kermany OCT2017"),
        "verification_status": "SUCCESS" if not missing_dirs else "FAILED",
        "missing_directories": missing_dirs,
        "splits": {}
    }

    for split_name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if path in missing_dirs:
            report["splits"][split_name] = {
                "exists": False, "path": path, "total_images": 0, "class_counts": {},
                "corrupt_images": [], "unsupported_files": []
            }
            continue

        class_status = verify_class_folders(path, classes)
        total_images, class_counts, corrupt, unsupported = verify_images(path, classes, extensions)

        report["splits"][split_name] = {
            "exists": True, "path": path, "class_folders_status": class_status,
            "total_images": total_images, "class_counts": class_counts,
            "corrupt_images": corrupt, "unsupported_files": unsupported
        }

    return report


def verify_dataset(config_path: str, report_output_path: str) -> bool:
    print("=" * 40)
    print("TrustOCT Dataset Verification")
    print("=" * 40)

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    report = generate_dataset_report(config)
    
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=4)

    if report["verification_status"] == "FAILED":
        print(f"[ERROR] Verification FAILED. Missing directories: {report['missing_directories']}")
        return False

    for split_name, split_info in report["splits"].items():
        if split_info.get("corrupt_images"):
            print(f"[ERROR] Found corrupt images in {split_name} split!")
            return False

    print("[OK] Folder structure verified")
    print("[OK] Classes verified")
    print("[OK] Images verified")
    print("[OK] No corrupted images")
    print(f"Report saved to: {report_output_path}")
    print("=" * 40)
    return True


# =====================================================================
# 5. Dataset Statistics
# =====================================================================

def calculate_statistics(
    base_path: str, classes: List[str], extensions: Tuple[str, ...], max_samples: int = 1000
) -> Tuple[List[float], List[float], Tuple[float, float], Dict[str, int]]:
    class_distribution = {}
    all_filepaths = []

    for c in classes:
        class_path = os.path.join(base_path, c)
        if not os.path.exists(class_path):
            class_distribution[c] = 0
            continue

        files = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        class_distribution[c] = len(files)
        all_filepaths.extend(files)

    if not all_filepaths:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], (0.0, 0.0), class_distribution

    np.random.seed(42)
    sample_filepaths = all_filepaths
    if len(all_filepaths) > max_samples:
        indices = np.random.choice(len(all_filepaths), max_samples, replace=False)
        sample_filepaths = [all_filepaths[i] for i in indices]

    print(f"Calculating pixel statistics over {len(sample_filepaths)} images...")

    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    total_pixels = 0
    heights = []
    widths = []

    for filepath in sample_filepaths:
        try:
            with Image.open(filepath) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                widths.append(w)
                heights.append(h)

                arr = np.array(img_rgb) / 255.0
                pixels = arr.reshape(-1, 3)
                channel_sum += pixels.sum(axis=0)
                channel_sum_sq += (pixels ** 2).sum(axis=0)
                total_pixels += pixels.shape[0]
        except Exception:
            pass

    mean = (channel_sum / total_pixels).tolist()
    std = np.sqrt(np.maximum(0, (channel_sum_sq / total_pixels) - (channel_sum / total_pixels) ** 2)).tolist()
    avg_resolution = (float(np.mean(heights)), float(np.mean(widths)))

    return mean, std, avg_resolution, class_distribution


def generate_statistics_report(config_path: str, output_path: str, max_samples: int = 1000) -> bool:
    print("=" * 40)
    print("TrustOCT Dataset Statistics Calculator")
    print("=" * 40)

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    train_path = paths.get("train", "datasets/raw/Kermany/train")
    classes = config.get("classes", CLASS_NAMES)
    extensions = tuple(config.get("image", {}).get("extensions", SUPPORTED_EXTENSIONS))

    if not os.path.exists(train_path):
        print(f"Error: Train directory not found at {train_path}")
        return False

    mean, std, avg_res, distribution = calculate_statistics(
        train_path, classes, extensions, max_samples=max_samples
    )

    stats_report = {
        "dataset_name": config.get("dataset", {}).get("name", "Kermany OCT2017"),
        "subsampled_images": min(max_samples, sum(distribution.values())),
        "pixel_mean_rgb": mean,
        "pixel_std_rgb": std,
        "average_height": avg_res[0],
        "average_width": avg_res[1],
        "class_distribution": distribution
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats_report, f, indent=4)

    print(f"Statistics saved to: {output_path}")
    print(f"[OK] RGB Mean: {[round(m, 4) for m in mean]}")
    print(f"[OK] RGB Std:  {[round(s, 4) for s in std]}")
    print("=" * 40)
    return True


if __name__ == "__main__":
    # Direct verify triggers
    cfg_p = "configs/dataset.yaml"
    verify_dataset(cfg_p, "outputs/reports/dataset_report.json")
    generate_statistics_report(cfg_p, "outputs/reports/dataset_statistics.json")
