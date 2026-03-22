
from typing import Dict, Optional, Sequence, Union

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


class TabularEncoder(nn.Module):
    """Minimal tabular encoder for the FuseMoE-based MVP.

    Supports numerical, categorical, or mixed tabular inputs.

    Inputs:
        x_num: (batch, num_numeric_features) float tensor, optional
        x_cat: (batch, num_categorical_features) long tensor, optional

    Returns:
        {
            "pooled":   (batch, output_dim),
            "sequence": (batch, 1, output_dim),
        }
    """

    def __init__(
        self,
        num_numeric_features: int = 0,
        categorical_cardinalities: Optional[Sequence[int]] = None,
        categorical_embedding_dim: Optional[Union[int, Sequence[int]]] = None,
        hidden_dims: Sequence[int] = (128,),
        output_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()

        self.num_numeric_features = int(num_numeric_features)
        self.categorical_cardinalities = list(categorical_cardinalities or [])
        self.num_categorical_features = len(self.categorical_cardinalities)
        self.output_dim_ = int(output_dim)

        if self.num_numeric_features < 0:
            raise ValueError("num_numeric_features must be >= 0")
        if any(card <= 0 for card in self.categorical_cardinalities):
            raise ValueError("All categorical cardinalities must be positive integers")
        if self.num_numeric_features == 0 and self.num_categorical_features == 0:
            raise ValueError("At least one input type must be configured")

        self.embeddings = nn.ModuleList()
        embedding_dims = []

        if self.num_categorical_features > 0:
            if categorical_embedding_dim is None:
                embedding_dims = [min(32, max(4, (card + 1) // 2)) for card in self.categorical_cardinalities]
            elif isinstance(categorical_embedding_dim, int):
                embedding_dims = [categorical_embedding_dim] * self.num_categorical_features
            else:
                embedding_dims = list(categorical_embedding_dim)
                if len(embedding_dims) != self.num_categorical_features:
                    raise ValueError(
                        "categorical_embedding_dim must be an int or a sequence with the same "
                        "length as categorical_cardinalities"
                    )

            for card, emb_dim in zip(self.categorical_cardinalities, embedding_dims):
                self.embeddings.append(nn.Embedding(card, emb_dim))

        input_dim = self.num_numeric_features + sum(embedding_dims)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim_))
        self.encoder = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self.output_dim_

    def _encode_numeric(self, x_num: Optional[torch.Tensor], batch_size: Optional[int]) -> Optional[torch.Tensor]:
        if self.num_numeric_features == 0:
            return None
        if x_num is None:
            raise ValueError("x_num must be provided because num_numeric_features > 0")
        if x_num.dim() != 2 or x_num.size(1) != self.num_numeric_features:
            raise ValueError(
                f"x_num must have shape (batch, {self.num_numeric_features}); got {tuple(x_num.shape)}"
            )
        if batch_size is not None and x_num.size(0) != batch_size:
            raise ValueError("x_num and x_cat must have the same batch size")
        return x_num.float()

    def _encode_categorical(
        self,
        x_cat: Optional[torch.Tensor],
        batch_size: Optional[int],
    ) -> Optional[torch.Tensor]:
        if self.num_categorical_features == 0:
            return None
        if x_cat is None:
            raise ValueError("x_cat must be provided because categorical_cardinalities is not empty")
        if x_cat.dim() != 2 or x_cat.size(1) != self.num_categorical_features:
            raise ValueError(
                f"x_cat must have shape (batch, {self.num_categorical_features}); got {tuple(x_cat.shape)}"
            )
        if batch_size is not None and x_cat.size(0) != batch_size:
            raise ValueError("x_num and x_cat must have the same batch size")

        x_cat = x_cat.long()
        embedded = []
        for col_idx, embedding in enumerate(self.embeddings):
            col = x_cat[:, col_idx]
            if torch.any(col < 0) or torch.any(col >= embedding.num_embeddings):
                raise ValueError(
                    f"x_cat column {col_idx} contains indices outside [0, {embedding.num_embeddings - 1}]"
                )
            embedded.append(embedding(col))
        return torch.cat(embedded, dim=-1)

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = None
        if x_num is not None:
            batch_size = x_num.size(0)
        if x_cat is not None and batch_size is None:
            batch_size = x_cat.size(0)

        num_features = self._encode_numeric(x_num, batch_size)
        cat_features = self._encode_categorical(x_cat, batch_size)

        features = []
        if num_features is not None:
            features.append(num_features)
        if cat_features is not None:
            features.append(cat_features)

        if not features:
            raise ValueError("At least one of x_num or x_cat must be provided")

        x = torch.cat(features, dim=-1)
        pooled = self.encoder(x)

        return {
            "pooled": pooled,
            "sequence": pooled.unsqueeze(1),
        }


__all__ = ["TabularEncoder"]


if __name__ == "__main__":
    torch.manual_seed(0)

    encoder_num = TabularEncoder(num_numeric_features=6, hidden_dims=(32,), output_dim=16)
    x_num = torch.randn(4, 6)
    out_num = encoder_num(x_num=x_num)
    print("numeric only:", tuple(out_num["pooled"].shape), tuple(out_num["sequence"].shape))

    encoder_mixed = TabularEncoder(
        num_numeric_features=3,
        categorical_cardinalities=[5, 10, 4],
        hidden_dims=(32, 16),
        output_dim=12,
    )
    x_num = torch.randn(4, 3)
    x_cat = torch.tensor([[0, 1, 2], [1, 3, 0], [4, 5, 1], [2, 9, 3]])
    out_mixed = encoder_mixed(x_num=x_num, x_cat=x_cat)
    print("mixed:", tuple(out_mixed["pooled"].shape), tuple(out_mixed["sequence"].shape))
