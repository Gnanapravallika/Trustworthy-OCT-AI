"""Base abstract dataset class for TrustOCT framework."""

import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from src.datasets.constants import CLASS_TO_INDEX
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.datasets.constants import CLASS_TO_INDEX


class BaseOCTDataset(Dataset, ABC):
    """Abstract Base Class for OCT datasets."""

    def __init__(
        self,
        base_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """Initialize base dataset.

        Args:
            base_path: Root folder of the dataset split.
            transform: Optional image transform pipeline (e.g. Albumentations).
            target_transform: Optional label transform pipeline.
        """
        self.base_path = base_path
        self.transform = transform
        self.target_transform = target_transform
        
        self.filepaths: List[str] = []
        self.labels: List[int] = []
        
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> None:
        """Scan folder structure or read metadata file to populate filepaths and labels."""
        pass

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Fetch the image and label at the specified index.

        Args:
            idx: Index integer.

        Returns:
            Tuple of (image_tensor, label_int).
        """
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        # Load image in RGB
        try:
            image = Image.open(filepath).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {filepath}: {e}")

        # Apply image transformations (e.g. Albumentations)
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            # Fallback to standard tensor conversion if no transforms are provided
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0

        # Apply label transformations
        if self.target_transform:
            label = self.target_transform(label)

        return image_tensor, label
