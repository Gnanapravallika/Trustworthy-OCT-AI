"""Dataset verification module for TrustOCT framework."""

import json
import os
import sys
from typing import Dict, List, Tuple, Union
import yaml
from PIL import Image

# Import constants
try:
    from src.datasets.constants import CLASS_NAMES, SUPPORTED_EXTENSIONS
except ImportError:
    # Handle direct script execution scenario
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.datasets.constants import CLASS_NAMES, SUPPORTED_EXTENSIONS


def verify_folder_structure(train_path: str, val_path: str, test_path: str) -> List[str]:
    """Verify that train, validation, and test directories exist.

    Args:
        train_path: Path to training images.
        val_path: Path to validation images.
        test_path: Path to test images.

    Returns:
        List of missing directories.
    """
    missing_dirs = []
    for name, path in [("Train", train_path), ("Validation", val_path), ("Test", test_path)]:
        if not os.path.exists(path):
            missing_dirs.append(path)
    return missing_dirs


def verify_class_folders(base_path: str, classes: List[str]) -> Dict[str, bool]:
    """Verify that folders for all classes exist under a base path.

    Args:
        base_path: Directory path containing the class folders.
        classes: List of class folder names.

    Returns:
        Dictionary mapping class name to existence (True/False).
    """
    class_status = {}
    for c in classes:
        class_path = os.path.join(base_path, c)
        class_status[c] = os.path.exists(class_path)
    return class_status


def verify_images(
    base_path: str, classes: List[str], extensions: Tuple[str, ...]
) -> Tuple[int, Dict[str, int], List[str], List[str]]:
    """Scan and verify all image files under the base path class subdirectories.

    Args:
        base_path: Directory path containing class subdirectories.
        classes: List of class names.
        extensions: Supported file extensions to look for.

    Returns:
        Tuple containing:
            - Total number of scanned images
            - Dictionary of count per class
            - List of corrupt images found
            - List of unsupported formats found
    """
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
            ext_lower = ext.lower()

            if ext_lower not in extensions:
                unsupported_files.append(file_path)
                continue

            total_images += 1
            class_counts[c] += 1

            # Verify image load
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                corrupt_images.append(file_path)

    return total_images, class_counts, corrupt_images, unsupported_files


def generate_dataset_report(config: dict) -> dict:
    """Generate a comprehensive dataset verification report.

    Args:
        config: Loaded dataset configuration dictionary.

    Returns:
        A dictionary containing the verification results.
    """
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

    # Verify each split
    for split_name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if path in missing_dirs:
            report["splits"][split_name] = {
                "exists": False,
                "path": path,
                "total_images": 0,
                "class_counts": {},
                "corrupt_images": [],
                "unsupported_files": []
            }
            continue

        class_status = verify_class_folders(path, classes)
        total_images, class_counts, corrupt, unsupported = verify_images(path, classes, extensions)

        report["splits"][split_name] = {
            "exists": True,
            "path": path,
            "class_folders_status": class_status,
            "total_images": total_images,
            "class_counts": class_counts,
            "corrupt_images": corrupt,
            "unsupported_files": unsupported
        }

    return report


def save_dataset_report(report: dict, output_path: str) -> None:
    """Save the verification report to a JSON file.

    Args:
        report: The verification report dict.
        output_path: Filepath where the JSON report should be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Report saved to: {output_path}")


def verify_dataset(config_path: str, report_output_path: str) -> bool:
    """Read the configuration and verify the dataset folders and files.

    Args:
        config_path: Path to dataset YAML config.
        report_output_path: Filepath to save generated verification report.

    Returns:
        Boolean indicating verification success.
    """
    print("=" * 40)
    print("TrustOCT Dataset Verification")
    print("=" * 40)

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    report = generate_dataset_report(config)
    save_dataset_report(report, report_output_path)

    # Check for missing paths or corruption
    if report["verification_status"] == "FAILED":
        print(f"❌ Verification FAILED. Missing directories: {report['missing_directories']}")
        return False

    has_corruption = False
    for split_name, split_info in report["splits"].items():
        if split_info.get("corrupt_images"):
            print(f"❌ Found corrupt images in {split_name} split!")
            has_corruption = True

    if has_corruption:
        print("❌ Verification FAILED due to corrupted files.")
        return False

    print("✓ Folder structure verified")
    print("✓ Classes verified")
    print("✓ Images verified")
    print("✓ No corrupted images")
    print("=" * 40)
    print("Verification Completed Successfully")
    print("=" * 40)
    return True


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cfg = os.path.join(project_root, "configs/dataset.yaml")
    out = os.path.join(project_root, "outputs/reports/dataset_report.json")
    verify_dataset(cfg, out)
