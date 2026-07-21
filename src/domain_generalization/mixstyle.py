"""MixStyle domain statistic mixing module for TrustOCT framework."""

import random
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle module (Zhou et al., ICLR 2021) for feature-level statistic mixing."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        """Initialize MixStyle module.

        Args:
            p: Activation probability of style mixing.
            alpha: Parameter of Beta distribution.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature maps of shape [B, C, H, W].

        Returns:
            Style-mixed feature maps of same shape.
        """
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        batch_size = x.size(0)

        # 1. Compute instance mean and standard deviation
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        # 2. Normalize features
        x_norm = (x - mu) / sig

        # 3. Shuffle statistics
        perm = torch.randperm(batch_size).to(x.device)
        mu_shuffled = mu[perm]
        sig_shuffled = sig[perm]

        # 4. Sample mixing factor from Beta distribution
        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        # Sample lambda shape [B, 1, 1, 1]
        lmda = beta_dist.sample((batch_size, 1, 1, 1)).to(x.device)

        # 5. Mix statistics
        mu_mixed = lmda * mu + (1.0 - lmda) * mu_shuffled
        sig_mixed = lmda * sig + (1.0 - lmda) * sig_shuffled

        # 6. Apply mixed statistics to normalized features
        return x_norm * sig_mixed + mu_mixed
