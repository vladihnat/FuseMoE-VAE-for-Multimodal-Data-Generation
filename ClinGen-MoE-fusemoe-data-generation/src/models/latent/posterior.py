from typing import Dict

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class PosteriorHead(nn.Module):
    """Minimal VAE posterior head for the FuseMoE-based MVP.

    Input:
        fused pooled embedding of shape (batch, input_dim)

    Output:
        {
            "mu": (batch, latent_dim),
            "logvar": (batch, latent_dim),
            "std": (batch, latent_dim),
            "z": (batch, latent_dim),
        }
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        min_logvar: float = -10.0,
        max_logvar: float = 10.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"x must have shape (batch, {self.input_dim}); got {tuple(x.shape)}"
            )

        h = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=self.min_logvar, max=self.max_logvar)
        std = torch.exp(0.5 * logvar)
        z = self.reparameterize(mu, logvar, deterministic=deterministic)

        return {
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "z": z,
        }


__all__ = ["PosteriorHead"]


if __name__ == "__main__":
    torch.manual_seed(0)
    head = PosteriorHead(input_dim=32, latent_dim=8, hidden_dim=64, num_layers=2, dropout=0.1)
    x = torch.randn(4, 32)
    out = head(x)
    print("mu:", tuple(out["mu"].shape))
    print("logvar:", tuple(out["logvar"].shape))
    print("std:", tuple(out["std"].shape))
    print("z:", tuple(out["z"].shape))
