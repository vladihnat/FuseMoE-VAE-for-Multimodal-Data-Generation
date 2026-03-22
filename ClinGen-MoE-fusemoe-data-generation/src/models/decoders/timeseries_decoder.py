import math
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


class TSIrregularDecoder(nn.Module):
    """Minimal decoder for irregular time-series in the MVP multimodal VAE.
    
    This decoder reconstructs timestep-wise features by conditioning the global 
    latent representation `z` on specific target timesteps.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        embed_time: int = 64,
        hidden_dims: Sequence[int] = (128, 128),
        default_seq_len: int = 48,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.embed_time = int(embed_time)
        self.default_seq_len = int(default_seq_len)

        if self.latent_dim <= 0 or self.output_dim <= 0:
            raise ValueError("latent_dim and output_dim must be > 0")
        if self.embed_time <= 1:
            raise ValueError("embed_time must be > 1 to support periodic/linear splits")

        # Time embedding layers (mirroring the encoder's approach)
        self.periodic = nn.Linear(1, self.embed_time - 1)
        self.linear = nn.Linear(1, 1)

        # Build the backbone
        layers = []
        in_dim = self.latent_dim + self.embed_time
        
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers")
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Output head for the reconstructed time-series values
        self.output_head = nn.Linear(in_dim, self.output_dim)

    def learn_time_embedding(self, tt: torch.Tensor) -> torch.Tensor:
        """Embeds continuous time into a fixed-dimensional vector."""
        if tt.dim() == 1:
            tt = tt.unsqueeze(0)
        tt = tt.float().unsqueeze(-1)
        out_periodic = torch.sin(self.periodic(tt))
        out_linear = self.linear(tt)
        return torch.cat([out_linear, out_periodic], dim=-1)

    def forward(
        self, 
        z: torch.Tensor, 
        target_times: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: Latent representation of shape (batch, latent_dim).
            target_times: Optional tensor of shape (batch, seq_len) dictating 
                          the timesteps to reconstruct. If None, defaults to 
                          an evenly spaced grid.
                          
        Returns:
            Dictionary containing the reconstructed time-series values and 
            the times they were evaluated at.
        """
        if z.dim() != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(f"z must have shape (batch, {self.latent_dim}); got {tuple(z.shape)}")

        batch_size = z.size(0)

        # Handle target times
        if target_times is None:
            # Fallback to a default sequence of times between 0 and 1
            base = torch.linspace(0.0, 1.0, self.default_seq_len, device=z.device, dtype=z.dtype)
            target_times = base.unsqueeze(0).expand(batch_size, -1)
        elif target_times.dim() != 2 or target_times.size(0) != batch_size:
            raise ValueError("target_times must have shape (batch, seq_len)")

        seq_len = target_times.size(1)

        # 1. Embed the target times -> (batch, seq_len, embed_time)
        time_emb = self.learn_time_embedding(target_times)

        # 2. Expand latent z to match sequence length -> (batch, seq_len, latent_dim)
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)

        # 3. Combine latent representation with time embeddings -> (batch, seq_len, latent_dim + embed_time)
        h = torch.cat([z_expanded, time_emb], dim=-1)

        # 4. Decode point-wise through the backbone
        h = self.backbone(h)
        ts_recon = self.output_head(h)

        return {
            "ts_recon": ts_recon,
            "target_times": target_times,
        }


__all__ = ["TSIrregularDecoder"]

