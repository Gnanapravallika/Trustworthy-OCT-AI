"""End-to-end integration smoke test for TrustOCT pipeline."""

import os
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch
import yaml

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(project_root)

from src.dataset import CLASS_NAMES, verify_dataset, generate_statistics_report
from src.models import build_model
from src.training import Trainer
from src.dataset import get_dataset_and_loader
from src.xai import calculate_classification_metrics
from src.xai import plot_confusion_matrix, plot_reliability_diagram


def run_smoke_test():
    print("=" * 60)
    print("TrustOCT End-to-End Integration Smoke Test")
    print("=" * 60)

    # 1. Create a temporary directory for datasets and outputs
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary testing sandbox: {temp_dir}")
    
    try:
        # Create directories for mock dataset splits
        dataset_dir = os.path.join(temp_dir, "datasets")
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(dataset_dir, split)
            for class_name in CLASS_NAMES:
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Write 4 mock images per class (64x64 small images for speed)
                for i in range(4):
                    img_path = os.path.join(class_dir, f"mock_{i}.jpeg")
                    img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    Image.fromarray(img_data).save(img_path)

        # 2. Write temporary config YAMLs
        configs_dir = os.path.join(temp_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        
        dataset_yaml_path = os.path.join(configs_dir, "dataset.yaml")
        dataset_cfg = {
            "dataset": {"name": "Mock Kermany"},
            "paths": {
                "train": os.path.join(dataset_dir, "train"),
                "val": os.path.join(dataset_dir, "val"),
                "test": os.path.join(dataset_dir, "test")
            },
            "classes": CLASS_NAMES,
            "image": {"extensions": [".jpeg", ".jpg", ".png"]},
            "loader": {
                "batch_size": 2,
                "num_workers": 0,  # Zero workers for single-process speed
                "pin_memory": False,
                "shuffle": True
            }
        }
        with open(dataset_yaml_path, "w") as f:
            yaml.dump(dataset_cfg, f)

        aug_yaml_path = os.path.join(configs_dir, "augmentation.yaml")
        aug_cfg = {
            "preprocessing": {
                "resize": [64, 64],
                "clahe": {"enabled": False},
                "bilateral_filter": {"enabled": False}
            },
            "augmentations": {
                "random_rotate": {"limit": 10, "p": 0.5},
                "horizontal_flip": {"p": 0.5},
                "random_brightness_contrast": {"p": 0.5},
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
            }
        }
        with open(aug_yaml_path, "w") as f:
            yaml.dump(aug_cfg, f)

        train_yaml_path = os.path.join(configs_dir, "train.yaml")
        train_cfg = {
            "device": "cpu",  # Force CPU for smoke testing
            "epochs": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "seed": 42,
            "logging": {"tensorboard": False},
            "checkpoint": {"save_freq": 1, "patience": 3}
        }
        with open(train_yaml_path, "w") as f:
            yaml.dump(train_cfg, f)

        model_yaml_path = os.path.join(configs_dir, "model.yaml")
        model_cfg = {
            "backbone": "resnet50",
            "pretrained": False,  # No pretraining downloads for test speed
            "feature_module": "multiscale",
            "attention": "cbam",
            "domain_generalization": "mixstyle",
            "mixstyle": {"mix_prob": 1.0, "alpha": 0.1},
            "head": "edl",
            "num_classes": 4,
            "dropout": 0.1
        }
        with open(model_yaml_path, "w") as f:
            yaml.dump(model_cfg, f)

        # 3. Trigger verify and statistics reports
        print("Running verify_dataset check on mock layout...")
        verify_report_path = os.path.join(temp_dir, "dataset_report.json")
        verify_success = verify_dataset(dataset_yaml_path, verify_report_path)
        assert verify_success, "Dataset verification failed!"
        
        print("Running generate_statistics_report...")
        stats_report_path = os.path.join(temp_dir, "dataset_stats.json")
        stats_success = generate_statistics_report(dataset_yaml_path, stats_report_path, max_samples=10)
        assert stats_success, "Stats calculation failed!"

        # 4. Load datasets, build model and Trainer
        print("Initializing dataset loaders...")
        _, train_loader = get_dataset_and_loader("train", dataset_yaml_path, aug_yaml_path)
        _, val_loader = get_dataset_and_loader("val", dataset_yaml_path, aug_yaml_path)

        print("Building model...")
        model = build_model(model_yaml_path)

        print("Compiling Trainer engine...")
        experiment_dir = os.path.join(temp_dir, "outputs", "EXP_SmokeTest")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_config_path=train_yaml_path,
            model_config_path=model_yaml_path,
            experiment_dir=experiment_dir
        )

        print("Running one epoch fit cycle...")
        trainer.fit()

        # 5. Check if checkpoints are saved correctly
        checkpoint_path = os.path.join(experiment_dir, "weights_best.pth")
        assert os.path.exists(checkpoint_path), f"Checkpoint not saved at {checkpoint_path}"
        print("[OK] Model checkpoint saved successfully")

        # 6. Run evaluation and plotting tests
        print("Generating mock predictions for evaluation & plots...")
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        # Random mock probabilities
        y_prob = np.random.dirichlet(np.ones(4), size=8)
        y_pred = np.argmax(y_prob, axis=1)

        metrics = calculate_classification_metrics(y_true, y_pred, y_prob, num_classes=4)
        print(f"[OK] Classification metrics calculated (Accuracy: {metrics['accuracy']:.4f})")

        cm_path = os.path.join(experiment_dir, "confusion_matrix.png")
        plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_path)
        assert os.path.exists(cm_path), "Confusion matrix figure not saved!"
        print("[OK] Confusion matrix figure generated")

        rel_path = os.path.join(experiment_dir, "reliability_diagram.png")
        plot_reliability_diagram(y_true, y_prob, num_bins=5, save_path=rel_path)
        assert os.path.exists(rel_path), "Reliability diagram figure not saved!"
        print("[OK] Reliability diagram figure generated")

        print("=" * 60)
        print("SMOKE TEST SUCCESS: The TrustOCT framework is 100% operational!")
        print("=" * 60)

    finally:
        # Clean up temp sandbox
        print(f"Cleaning up sandbox: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_smoke_test()
