from typing import Literal

import torch
import torch.nn as nn


Reduction = Literal["none", "mean", "sum"]


def kl_standard_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) for a diagonal Gaussian posterior."""
    if mu.shape != logvar.shape:
        raise ValueError(f"mu and logvar must have the same shape, got {tuple(mu.shape)} and {tuple(logvar.shape)}")
    if mu.dim() != 2:
        raise ValueError(f"mu and logvar must be 2D tensors of shape (batch, latent_dim), got {tuple(mu.shape)}")

    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    if reduction == "none":
        return kl_per_sample
    if reduction == "mean":
        return kl_per_sample.mean()
    if reduction == "sum":
        return kl_per_sample.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return kl_standard_normal(mu, logvar, reduction=self.reduction)


__all__ = ["kl_standard_normal", "KLDivergenceLoss"]
