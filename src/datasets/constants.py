"""Dataset constants for TrustOCT framework."""

from typing import Dict, List, Tuple

# Class names mapping
CLASS_NAMES: List[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Class to index mapping
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Supported image file extensions
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpeg", ".jpg", ".png")

# Default image size for modeling
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
