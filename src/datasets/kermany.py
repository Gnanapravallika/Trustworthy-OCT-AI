"""Kermany OCT2017 dataset class for TrustOCT framework."""

import os
import sys
from typing import Callable, Optional

try:
    from src.datasets.base_dataset import BaseOCTDataset
    from src.datasets.constants import CLASS_NAMES, CLASS_TO_INDEX, SUPPORTED_EXTENSIONS
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.datasets.base_dataset import BaseOCTDataset
    from src.datasets.constants import CLASS_NAMES, CLASS_TO_INDEX, SUPPORTED_EXTENSIONS


class KermanyDataset(BaseOCTDataset):
    """Dataset class for Kermany OCT2017 dataset."""

    def _load_metadata(self) -> None:
        """Scan folders to populate filepaths and labels."""
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Kermany dataset path does not exist at {self.base_path}")

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
