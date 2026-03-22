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


class IrregularTSDecoder(nn.Module):
    """Decoder for irregular time-series from a latent code.

    Mirrors the TSIrregularEncoder design: uses the same learnable time embedding
    (periodic + linear) to condition each output timestep on its position, then
    fuses the latent representation with the per-step time embedding to produce
    reconstructed values and optionally observation-mask logits.

    Inputs:
        z:           (batch, latent_dim)
        query_times: (batch, num_steps) or (num_steps,)  [optional, defaults to uniform grid]

    Returns a dictionary with:
        ts_recon:       (batch, num_steps, output_dim)   reconstructed values
        ts_mask_logits: (batch, num_steps, output_dim)   raw logits for observation mask
                        (None if decode_mask is False)
        query_times:    (batch, num_steps)               the time grid used
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        embed_time: int = 64,
        hidden_layers: Sequence[int] = (128,),
        num_query_steps: int = 48,
        dropout: float = 0.1,
        activation: str = "gelu",
        decode_mask: bool = True,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if embed_time < 2:
            raise ValueError("embed_time must be >= 2")
        if num_query_steps <= 0:
            raise ValueError("num_query_steps must be > 0")

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_time = embed_time
        self.num_query_steps = num_query_steps
        self.decode_mask = decode_mask

        # --- latent backbone: z -> (batch, hidden_dim) ---
        layers = []
        in_dim = latent_dim
        for h in hidden_layers:
            if h <= 0:
                raise ValueError("hidden_layers must contain positive integers")
            layers.append(nn.Linear(in_dim, h))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.backbone = nn.Sequential(*layers)

        # --- time embedding (same design as TSIrregularEncoder) ---
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

        # --- per-step fusion: [latent_repr | time_emb] -> hidden_dim ---
        self.fuse = nn.Linear(hidden_dim + embed_time, hidden_dim)
        self.fuse_act = _make_activation(activation)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_dropout = nn.Dropout(dropout)

        # --- output heads ---
        self.value_head = nn.Linear(hidden_dim, output_dim)
        self.mask_head = nn.Linear(hidden_dim, output_dim) if decode_mask else None

    def learn_time_embedding(self, tt: torch.Tensor) -> torch.Tensor:
        """Map time tensor to learned embedding, matching encoder convention.

        Args:
            tt: (batch, num_steps)

        Returns:
            (batch, num_steps, embed_time)
        """
        if tt.dim() == 1:
            tt = tt.unsqueeze(0)
        tt = tt.float().unsqueeze(-1)          # (batch, num_steps, 1)
        out_periodic = torch.sin(self.periodic(tt))   # (batch, num_steps, embed_time-1)
        out_linear = self.linear(tt)                  # (batch, num_steps, 1)
        return torch.cat([out_linear, out_periodic], dim=-1)  # (batch, num_steps, embed_time)

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
            raise ValueError("query_times must have shape (num_steps,) or (batch, num_steps)")
        return query_times.to(device=device, dtype=dtype)

    def forward(
        self,
        z: torch.Tensor,
        query_times: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(
                f"z must have shape (batch, {self.latent_dim}); got {tuple(z.shape)}"
            )

        batch_size = z.size(0)

        query_times = self._build_query_times(
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
            query_times=query_times,
        )
        num_steps = query_times.size(1)

        # latent -> backbone representation: (batch, hidden_dim)
        h = self.backbone(z)

        # time embedding at query positions: (batch, num_steps, embed_time)
        time_emb = self.learn_time_embedding(query_times)

        # expand latent across time steps and fuse with time embedding
        h_expanded = h.unsqueeze(1).expand(-1, num_steps, -1)    # (batch, num_steps, hidden_dim)
        fused = torch.cat([h_expanded, time_emb], dim=-1)        # (batch, num_steps, hidden_dim + embed_time)
        fused = self.fuse_act(self.fuse(fused))                   # (batch, num_steps, hidden_dim)
        fused = self.out_dropout(self.out_norm(fused))

        ts_recon = self.value_head(fused)                         # (batch, num_steps, output_dim)
        ts_mask_logits = self.mask_head(fused) if self.mask_head is not None else None

        return {
            "ts_recon": ts_recon,
            "ts_mask_logits": ts_mask_logits,
            "query_times": query_times,
        }


__all__ = ["IrregularTSDecoder"]


if __name__ == "__main__":
    torch.manual_seed(0)
    latent_dim, output_dim = 16, 6
    decoder = IrregularTSDecoder(
        latent_dim=latent_dim,
        output_dim=output_dim,
        hidden_dim=32,
        embed_time=16,
        hidden_layers=(64,),
        num_query_steps=48,
        dropout=0.1,
        decode_mask=True,
    )

    z = torch.randn(4, latent_dim)
    out = decoder(z)
    print("ts_recon:      ", tuple(out["ts_recon"].shape))
    print("ts_mask_logits:", tuple(out["ts_mask_logits"].shape))
    print("query_times:   ", tuple(out["query_times"].shape))

    # with explicit query times
    query_times = torch.sort(torch.rand(4, 20), dim=1).values
    out2 = decoder(z, query_times=query_times)
    print("ts_recon (custom times):", tuple(out2["ts_recon"].shape))
