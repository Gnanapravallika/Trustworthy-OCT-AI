"""Dataset statistics calculator for TrustOCT framework."""

import json
import os
import sys
from typing import Dict, List, Tuple
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from src.datasets.constants import CLASS_NAMES, SUPPORTED_EXTENSIONS
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.datasets.constants import CLASS_NAMES, SUPPORTED_EXTENSIONS


def calculate_statistics(
    base_path: str,
    classes: List[str],
    extensions: Tuple[str, ...],
    max_samples: int = 1000
) -> Tuple[List[float], List[float], Tuple[float, float], Dict[str, int]]:
    """Calculate mean, standard deviation, average resolution, and class distribution.

    Args:
        base_path: Root folder of the dataset split.
        classes: List of folder names per class.
        extensions: Tuple of supported file extensions.
        max_samples: Maximum number of images to subsample for faster mean/std estimation.

    Returns:
        Tuple containing:
            - Mean per RGB channel [R, G, B]
            - Std per RGB channel [R, G, B]
            - Average height and width (H, W)
            - Class distribution dictionary
    """
    class_distribution = {}
    all_filepaths = []

    # Gather file paths and compute distribution
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

    # Subsample if necessary for fast execution
    np.random.seed(42)
    sample_filepaths = all_filepaths
    if len(all_filepaths) > max_samples:
        indices = np.random.choice(len(all_filepaths), max_samples, replace=False)
        sample_filepaths = [all_filepaths[i] for i in indices]

    print(f"Calculating pixel statistics over a subset of {len(sample_filepaths)} images...")

    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    total_pixels = 0
    heights = []
    widths = []

    for filepath in tqdm(sample_filepaths, desc="Scanning Images"):
        try:
            with Image.open(filepath) as img:
                # Ensure RGB mode
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                widths.append(w)
                heights.append(h)

                # Convert to numpy array in [0, 1]
                arr = np.array(img_rgb) / 255.0

                # Reshape to [Pixels, Channels]
                pixels = arr.reshape(-1, 3)
                channel_sum += pixels.sum(axis=0)
                channel_sum_sq += (pixels ** 2).sum(axis=0)
                total_pixels += pixels.shape[0]
        except Exception as e:
            print(f"Skipping problematic image {filepath}: {e}")

    # Compute mean and standard deviation
    mean = (channel_sum / total_pixels).tolist()
    std = np.sqrt((channel_sum_sq / total_pixels) - (channel_sum / total_pixels) ** 2).tolist()
    avg_resolution = (float(np.mean(heights)), float(np.mean(widths)))

    return mean, std, avg_resolution, class_distribution


def generate_statistics_report(config_path: str, output_path: str, max_samples: int = 1000) -> bool:
    """Load configuration, compute dataset statistics, and output a JSON report.

    Args:
        config_path: Path to dataset YAML configuration.
        output_path: Path to save statistics JSON report.
        max_samples: Limit for image subsampling.

    Returns:
        Boolean indicating calculation success.
    """
    print("=" * 40)
    print("TrustOCT Dataset Statistics Calculator")
    print("=" * 40)

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    train_path = paths.get("train", "datasets/raw/Kermany/train")
    classes = config.get("classes", CLASS_NAMES)
    extensions = tuple(config.get("image", {}).get("extensions", SUPPORTED_EXTENSIONS))

    if not os.path.exists(train_path):
        print(f"Error: Train directory not found at {train_path}. Cannot calculate stats.")
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
    print(f"✓ RGB Mean: {[round(m, 4) for m in mean]}")
    print(f"✓ RGB Std:  {[round(s, 4) for s in std]}")
    print(f"✓ Average Resolution: {round(avg_res[0], 1)}x{round(avg_res[1], 1)}")
    print("=" * 40)
    return True


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cfg = os.path.join(project_root, "configs/dataset.yaml")
    out = os.path.join(project_root, "outputs/reports/dataset_statistics.json")
    generate_statistics_report(cfg, out)
