"""Unit and pipeline integration tests for TrustOCT dataset loader."""

import os
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.datasets.constants import CLASS_NAMES, CLASS_TO_INDEX
from src.datasets.kermany import KermanyDataset
from src.datasets.loader import create_dataloader
from src.datasets.verify import generate_dataset_report, verify_dataset
from src.preprocessing.transforms import get_train_transforms, get_val_transforms


@pytest.fixture
def mock_dataset_dir():
    """Create a temporary directory simulating Kermany OCT2017 folder layout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create train, val, test splits
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(tmpdir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            # Create subfolders per class and write mock images
            for class_name in CLASS_NAMES:
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Write mock image
                img_path = os.path.join(class_dir, f"mock_img_0.jpeg")
                img_data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                img = Image.fromarray(img_data)
                img.save(img_path)
                
        yield tmpdir


def test_constants():
    """Verify constants are correctly set."""
    assert len(CLASS_NAMES) == 4
    assert CLASS_TO_INDEX["NORMAL"] == 3
    assert CLASS_TO_INDEX["CNV"] == 0


def test_dataset_verification(mock_dataset_dir):
    """Test verify_dataset using mock layout configuration."""
    config = {
        "dataset": {"name": "Mock Kermany"},
        "paths": {
            "train": os.path.join(mock_dataset_dir, "train"),
            "val": os.path.join(mock_dataset_dir, "val"),
            "test": os.path.join(mock_dataset_dir, "test")
        },
        "classes": CLASS_NAMES,
        "image": {"extensions": [".jpeg", ".jpg", ".png"]},
        "loader": {"batch_size": 2}
    }
    
    config_path = os.path.join(mock_dataset_dir, "dataset.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    report = generate_dataset_report(config)
    assert report["verification_status"] == "SUCCESS"
    assert report["splits"]["train"]["total_images"] == 4
    
    # Run the integration call
    report_output_path = os.path.join(mock_dataset_dir, "report.json")
    success = verify_dataset(config_path, report_output_path)
    assert success is True
    assert os.path.exists(report_output_path)


def test_kermany_dataset_transforms(mock_dataset_dir):
    """Test dataset class loading and augmentation pipeline shapes."""
    config = {
        "preprocessing": {
            "resize": [224, 224],
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "bilateral_filter": {"enabled": True, "d": 5, "sigma_color": 50, "sigma_space": 50}
        },
        "augmentations": {
            "random_rotate": {"limit": 10, "p": 0.5},
            "horizontal_flip": {"p": 0.5},
            "random_brightness_contrast": {"brightness_limit": 0.05, "contrast_limit": 0.05, "p": 0.5},
            "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        }
    }
    
    # Train transforms
    train_transforms = get_train_transforms(config)
    dataset = KermanyDataset(
        base_path=os.path.join(mock_dataset_dir, "train"),
        transform=train_transforms
    )
    
    assert len(dataset) == 4
    image, label = dataset[0]
    
    # Image tensor checks
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert label in [0, 1, 2, 3]


def test_dataloader_creation(mock_dataset_dir):
    """Verify dataloader fetches batches correctly."""
    dataset = KermanyDataset(
        base_path=os.path.join(mock_dataset_dir, "train"),
        transform=None
    )
    loader = create_dataloader(dataset, batch_size=2, shuffle=False)
    
    for images, labels in loader:
        assert images.shape[0] == 2
        assert images.shape[1] == 3
        assert labels.shape[0] == 2
        break
