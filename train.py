"""Master training execution script for TrustOCT framework."""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
import numpy as np
import torch
import yaml

# Add project root to python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.models import build_model
from src.dataset import get_dataset_and_loader
from src.training import Trainer


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


def set_seed(seed: int = 42):
    """Enforce exact random seeds for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enforce deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed controlled: {seed}")


def get_git_commit() -> str:
    """Retrieve current Git commit hash for lineage verification."""
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash
    except Exception:
        return "unknown"


def main():
    """Execute model training pipeline."""
    args = parse_args()
    
    # Resolve config paths relative to project root
    dataset_cfg_path = os.path.join(project_root, args.dataset_config)
    model_cfg_path = os.path.join(project_root, args.model_config)
    train_cfg_path = os.path.join(project_root, args.train_config)
    aug_cfg_path = os.path.join(project_root, args.augmentation_config)

    # Load training configuration to extract seed
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(dataset_cfg_path, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    # 0. Apply random seed control
    seed = train_cfg.get("seed", 42) if train_cfg else 42
    set_seed(seed)

    # Set up experiment directory
    experiment_dir = os.path.join(project_root, "outputs", args.experiment_name)
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

    # 1. Gather and save experiment metadata
    metadata = {
        "experiment": args.experiment_name,
        "dataset": dataset_cfg.get("dataset", {}).get("name", "Kermany OCT2017"),
        "epochs": train_cfg.get("epochs", 30) if train_cfg else 30,
        "optimizer": train_cfg.get("optimizer", "adamw") if train_cfg else "adamw",
        "scheduler": train_cfg.get("scheduler", "cosine") if train_cfg else "cosine",
        "learning_rate": train_cfg.get("learning_rate", 1e-4) if train_cfg else 1e-4,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit()
    }
    
    metadata_path = os.path.join(experiment_dir, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Experiment metadata cataloged: {metadata_path}")

    print("=" * 60)
    print(f"TrustOCT Training Session: {args.experiment_name}")
    print("=" * 60)

    # 2. Build Data Loaders
    print("Initializing Data Loaders...")
    train_dataset, train_loader = get_dataset_and_loader("train", dataset_cfg_path, aug_cfg_path)
    val_dataset, val_loader = get_dataset_and_loader("val", dataset_cfg_path, aug_cfg_path)

    # 3. Build Model
    print("Assembling TrustOCT model from configurations...")
    model = build_model(model_cfg_path)

    # 4. Instantiate Trainer
    print("Compiling training engine...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config_path=train_cfg_path,
        model_config_path=model_cfg_path,
        experiment_dir=experiment_dir
    )

    # 5. Fit Model
    print("Commencing model fit...")
    trainer.fit()
    print("=" * 60)
    print(f"Training completed successfully. Weights saved to {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
