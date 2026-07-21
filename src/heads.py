"""Prediction heads (Softmax and Evidential) for TrustOCT framework."""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseHead(nn.Module, ABC):
    """Abstract base class for all prediction heads."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class SoftmaxHead(BaseHead):
    """Standard softmax classifier head producing class logits."""

    def __init__(self, in_features: int, num_classes: int, dropout_prob: float = 0.5):
        super().__init__(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


class EvidentialHead(BaseHead):
    """Evidential classification head outputting positive Dirichlet evidence parameters."""

    def __init__(self, in_features: int, num_classes: int, dropout_prob: float = 0.5):
        super().__init__(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(x)
        logits = self.fc(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        return evidence, alpha
