from typing import Dict, Optional, Sequence

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


class TabularDecoder(nn.Module):
    """Minimal decoder for tabular data in the MVP multimodal VAE."""

    def __init__(
        self,
        latent_dim: int,
        num_numeric_features: int = 0,
        categorical_cardinalities: Optional[Sequence[int]] = None,
        hidden_dims: Sequence[int] = (64, 64),
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.num_numeric_features = int(num_numeric_features)
        self.categorical_cardinalities = list(categorical_cardinalities or [])
        self.num_categorical_features = len(self.categorical_cardinalities)

        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if self.num_numeric_features < 0:
            raise ValueError("num_numeric_features must be >= 0")
        if any(card <= 1 for card in self.categorical_cardinalities):
            raise ValueError("Each categorical cardinality must be > 1")
        if self.num_numeric_features == 0 and self.num_categorical_features == 0:
            raise ValueError("At least one output type must be configured")

        layers = []
        in_dim = self.latent_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers")
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.backbone_output_dim = in_dim

        self.numeric_head = (
            nn.Linear(self.backbone_output_dim, self.num_numeric_features)
            if self.num_numeric_features > 0
            else None
        )

        self.categorical_heads = nn.ModuleList(
            [nn.Linear(self.backbone_output_dim, card) for card in self.categorical_cardinalities]
        )

    def forward(self, z: torch.Tensor) -> Dict[str, object]:
        if z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(
                f"z must have shape (batch, {self.latent_dim}); got {tuple(z.shape)}"
            )

        h = self.backbone(z)

        tab_num_recon = self.numeric_head(h) if self.numeric_head is not None else None
        tab_cat_logits = [head(h) for head in self.categorical_heads]
        tab_cat_probs = [torch.softmax(logits, dim=-1) for logits in tab_cat_logits]
        tab_cat_pred = (
            torch.stack([logits.argmax(dim=-1) for logits in tab_cat_logits], dim=-1)
            if tab_cat_logits
            else None
        )

        return {
            "tab_num_recon": tab_num_recon,
            "tab_cat_logits": tab_cat_logits,
            "tab_cat_probs": tab_cat_probs,
            "tab_cat_pred": tab_cat_pred,
        }


__all__ = ["TabularDecoder"]
