from typing import Dict

import torch


def multimodal_vae_total_loss(
    reconstruction_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    balance_loss: torch.Tensor,
    beta: float = 1.0,
    lambda_balance: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Combine reconstruction, KL, and MoE balance terms."""
    total = reconstruction_loss + beta * kl_loss + lambda_balance * balance_loss
    return {
        "reconstruction": reconstruction_loss,
        "kl": kl_loss,
        "balance": balance_loss,
        "total": total,
    }


__all__ = ["multimodal_vae_total_loss"]
