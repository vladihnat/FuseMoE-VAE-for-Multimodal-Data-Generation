
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MultiTimeAttention(nn.Module):
    """Minimal mTAND-style attention block adapted from FuseMoE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_time: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if embed_time % num_heads != 0:
            raise ValueError("embed_time must be divisible by num_heads")

        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, hidden_dim),
            ]
        )

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        value_dim = value.size(-1)
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(value_dim, dim=-1)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e4)

        attn = F.softmax(scores, dim=-2)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=self.training)

        return torch.sum(attn * value.unsqueeze(-3), dim=-2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        batch_size, _, value_dim = value.size()

        if mask is not None:
            mask = mask.unsqueeze(1)

        value = value.unsqueeze(1)

        query, key = [
            linear(x).view(x.size(0), -1, self.num_heads, self.embed_time_k).transpose(1, 2)
            for linear, x in zip(self.linears[:2], (query, key))
        ]

        context = self.attention(query, key, value, mask=mask, dropout_p=dropout_p)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * value_dim)
        return self.linears[-1](context)


class TSIrregularEncoder(nn.Module):
    """Irregular time-series encoder inspired by FuseMoE's multiTimeAttention.

    Inputs:
        values: (batch, seq_len, input_dim)
        mask:   (batch, seq_len, input_dim) or (batch, seq_len)
        times:  (batch, seq_len)

    Returns a dictionary with:
        sequence:   (batch, query_len, hidden_dim)
        pooled:     (batch, hidden_dim)
        query_times:(batch, query_len)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_time: int = 64,
        num_heads: int = 4,
        num_query_steps: int = 48,
        dropout: float = 0.1,
        pooling: str = "mean",
        normalize_times: bool = True,
        use_mask_channel: bool = True,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "max"}:
            raise ValueError("pooling must be 'mean' or 'max'")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.num_query_steps = num_query_steps
        self.dropout_p = dropout
        self.pooling = pooling
        self.normalize_times = normalize_times
        self.use_mask_channel = use_mask_channel
        self.eps = 1e-6

        attn_input_dim = input_dim * 2 if use_mask_channel else input_dim
        self.time_attn = _MultiTimeAttention(
            input_dim=attn_input_dim,
            hidden_dim=hidden_dim,
            embed_time=embed_time,
            num_heads=num_heads,
        )

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def learn_time_embedding(self, tt: torch.Tensor) -> torch.Tensor:
        if tt.dim() == 1:
            tt = tt.unsqueeze(0)
        tt = tt.float().unsqueeze(-1)
        out_periodic = torch.sin(self.periodic(tt))
        out_linear = self.linear(tt)
        return torch.cat([out_linear, out_periodic], dim=-1)

    def _expand_mask(self, mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, values.size(-1))
        if mask.shape != values.shape:
            raise ValueError(
                f"mask must have shape {tuple(values.shape)} or (batch, seq_len); got {tuple(mask.shape)}"
            )
        return mask.float()

    def _normalize_times(self, times: torch.Tensor, observed_steps: torch.Tensor) -> torch.Tensor:
        large = torch.full_like(times, float("inf"))
        small = torch.full_like(times, float("-inf"))

        t_min = torch.where(observed_steps, times, large).amin(dim=1, keepdim=True)
        t_max = torch.where(observed_steps, times, small).amax(dim=1, keepdim=True)

        has_obs = observed_steps.any(dim=1, keepdim=True)
        t_min = torch.where(has_obs, t_min, torch.zeros_like(t_min))
        t_max = torch.where(has_obs, t_max, torch.ones_like(t_max))

        denom = (t_max - t_min).clamp_min(self.eps)
        times = (times - t_min) / denom
        return torch.where(observed_steps, times, torch.zeros_like(times))

    def _build_query_times(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        query_times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_times is None:
            base = torch.linspace(0.0, 1.0, self.num_query_steps, device=device, dtype=dtype)
            return base.unsqueeze(0).expand(batch_size, -1)

        if query_times.dim() == 1:
            query_times = query_times.unsqueeze(0).expand(batch_size, -1)
        if query_times.dim() != 2 or query_times.size(0) != batch_size:
            raise ValueError("query_times must have shape (query_len,) or (batch, query_len)")
        return query_times.to(device=device, dtype=dtype)

    def _pool(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return sequence.mean(dim=1)
        return sequence.max(dim=1).values

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        query_times: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if values.dim() != 3:
            raise ValueError("values must have shape (batch, seq_len, input_dim)")
        if times.dim() != 2:
            raise ValueError("times must have shape (batch, seq_len)")
        if values.shape[:2] != times.shape:
            raise ValueError(
                f"values and times must agree on (batch, seq_len); got {tuple(values.shape)} and {tuple(times.shape)}"
            )

        mask = self._expand_mask(mask, values)
        observed_steps = mask.any(dim=-1)

        if self.normalize_times:
            times = self._normalize_times(times.float(), observed_steps)
        else:
            times = times.float()

        query_times = self._build_query_times(
            batch_size=values.size(0),
            device=values.device,
            dtype=times.dtype,
            query_times=query_times,
        )

        time_key = self.learn_time_embedding(times)
        time_query = self.learn_time_embedding(query_times)

        if self.use_mask_channel:
            value_input = torch.cat([values, mask], dim=-1)
            attn_mask = torch.cat([mask, mask], dim=-1)
        else:
            value_input = values
            attn_mask = mask

        sequence = self.time_attn(
            query=time_query,
            key=time_key,
            value=value_input,
            mask=attn_mask,
            dropout_p=self.dropout_p,
        )

        sequence = self.out_dropout(self.out_norm(sequence))

        has_obs = observed_steps.any(dim=1).float().view(-1, 1, 1)
        sequence = sequence * has_obs

        pooled = self._pool(sequence)

        return {
            "sequence": sequence,
            "pooled": pooled,
            "query_times": query_times,
        }


__all__ = ["TSIrregularEncoder"]


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, input_dim = 4, 17, 6
    encoder = TSIrregularEncoder(
        input_dim=input_dim,
        hidden_dim=32,
        embed_time=16,
        num_heads=4,
        num_query_steps=12,
        dropout=0.1,
    )

    values = torch.randn(batch_size, seq_len, input_dim)
    mask = (torch.rand(batch_size, seq_len, input_dim) > 0.2).float()
    times = torch.sort(torch.rand(batch_size, seq_len), dim=1).values

    out = encoder(values=values, mask=mask, times=times)
    print("sequence:", tuple(out["sequence"].shape))
    print("pooled:", tuple(out["pooled"].shape))
    print("query_times:", tuple(out["query_times"].shape))
