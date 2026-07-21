"""Custom loss functions for evidential learning in TrustOCT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute KL Divergence between Dirichlet distribution alpha and uniform Dirichlet(1)."""
    device = alpha.device
    beta = torch.ones((1, num_classes), device=device)
    
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)
    
    ln_gamma_alpha = torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_gamma_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
    
    kl = (
        torch.lgamma(sum_alpha) - torch.lgamma(sum_beta) -
        ln_gamma_alpha + ln_gamma_beta +
        torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=1, keepdim=True)
    )
    return kl


class EdlLoss(nn.Module):
    """Evidential Deep Learning (EDL) Loss function using Dirichlet parameters."""

    def __init__(self, num_classes: int = 4, annealing_epochs: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
        # Convert target to one-hot encoding
        y_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        
        # Calculate sum of alpha parameters (Dirichlet strength S)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        
        # 1. Expected classification loss (Sum of Squares)
        prob = alpha / sum_alpha
        cls_loss = torch.sum((y_one_hot - prob) ** 2, dim=1, keepdim=True) + \
                   torch.sum(prob * (1.0 - prob) / (sum_alpha + 1.0), dim=1, keepdim=True)
        cls_loss = torch.mean(cls_loss)

        # 2. KL Divergence Regularization (penalizes high evidence on incorrect classes)
        alpha_hat = y_one_hot + (1.0 - y_one_hot) * alpha
        kl_val = kl_divergence(alpha_hat, self.num_classes)
        
        # Linear annealing weight lambda
        annealing_coef = min(1.0, float(epoch) / float(self.annealing_epochs))
        kl_loss = annealing_coef * torch.mean(kl_val)
        
        return cls_loss + kl_loss
