"""Master training execution script for TrustOCT framework."""

import argparse
import os
import sys

# Add project root to python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.models import build_model
from src.datasets import get_dataset_and_loader
from src.trainer import Trainer


def parse_args():
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description="TrustOCT Framework Training Script")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model.yaml",
        help="Path to model architecture configuration YAML."
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train.yaml",
        help="Path to training loop configuration YAML."
    )
    parser.add_argument(
        "--augmentation_config",
        type=str,
        default="configs/augmentation.yaml",
        help="Path to data preprocessing/augmentation YAML."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="EXP001_Baseline",
        help="Name of the experiment run folder."
    )
    return parser.parse_args()


def main():
    """Execute model training pipeline."""
    args = parse_args()
    
    # Resolve config paths relative to project root
    dataset_cfg_path = os.path.join(project_root, args.dataset_config)
    model_cfg_path = os.path.join(project_root, args.model_config)
    train_cfg_path = os.path.join(project_root, args.train_config)
    aug_cfg_path = os.path.join(project_root, args.augmentation_config)

    # Set up experiment directory
    experiment_dir = os.path.join(project_root, "experiments", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Save copy of configurations in the experiment folder for exact reproducibility
    for path, name in [
        (dataset_cfg_path, "dataset.yaml"),
        (model_cfg_path, "model.yaml"),
        (train_cfg_path, "train.yaml"),
        (aug_cfg_path, "augmentation.yaml")
    ]:
        with open(path, "r") as src, open(os.path.join(experiment_dir, name), "w") as dst:
            dst.write(src.read())

    print("=" * 60)
    print(f"TrustOCT Training Session: {args.experiment_name}")
    print("=" * 60)

    # 1. Build Data Loaders
    print("Initializing Data Loaders...")
    train_dataset, train_loader = get_dataset_and_loader("train", dataset_cfg_path, aug_cfg_path)
    val_dataset, val_loader = get_dataset_and_loader("val", dataset_cfg_path, aug_cfg_path)

    # 2. Build Model
    print("Assembling TrustOCT model from configurations...")
    model = build_model(model_cfg_path)

    # 3. Instantiate Trainer
    print("Compiling training engine...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config_path=train_cfg_path,
        model_config_path=model_cfg_path,
        experiment_dir=experiment_dir
    )

    # 4. Fit Model
    print("Commencing model fit...")
    trainer.fit()
    print("=" * 60)
    print(f"Training completed successfully. Weights saved to {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
