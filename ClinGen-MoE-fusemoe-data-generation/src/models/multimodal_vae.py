from typing import Dict, Optional

import torch
import torch.nn as nn

from losses.kl import kl_standard_normal


class MultimodalVAE(nn.Module):
    """Minimal multimodal VAE wrapper for the MVP pipeline."""

    def __init__(
        self,
        ts_encoder: nn.Module,
        tabular_encoder: nn.Module,
        fusion: nn.Module,
        posterior: nn.Module,
        tabular_decoder: nn.Module,
        ts_decoder: Optional[nn.Module] = None,
        kl_reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.ts_encoder = ts_encoder
        self.tabular_encoder = tabular_encoder
        self.fusion = fusion
        self.posterior = posterior
        self.tabular_decoder = tabular_decoder
        self.ts_decoder = ts_decoder
        self.kl_reduction = kl_reduction

    def forward(self, batch: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, object]:
        # 1. Time-Series Encoding
        ts_out = self.ts_encoder(
            values=batch["ts_values"],
            mask=batch["ts_mask"],
            times=batch["ts_times"],
        )
        
        # 2. Tabular Encoding (Using positional arguments based on our smoke tests)
        tab_out = self.tabular_encoder(
            batch.get("tab_num"),
            batch.get("tab_cat"),
        )
        
        # Safely extract tabular embedding (handles "embedding" or "pooled" keys)
        tab_emb = tab_out.get("embedding", tab_out.get("pooled"))

        # 3. Fusion
        fused = self.fusion({
            "ts": ts_out["pooled"],
            "tab": tab_emb,
        })
        
        # 4. Posterior Head
        post = self.posterior(fused["pooled"], deterministic=deterministic)

        # 5. Tabular Decoding
        tab_decoded = self.tabular_decoder(post["z"])
        
        # 6. Time-Series Decoding (CRITICAL FIX: Pass the target_times)
        if self.ts_decoder is not None:
            ts_decoded = self.ts_decoder(
                z=post["z"], 
                target_times=batch.get("ts_times")
            )
        else:
            ts_decoded = None

        # 7. KL Loss
        kl = kl_standard_normal(post["mu"], post["logvar"], reduction=self.kl_reduction)

        return {
            "ts_encoder": ts_out,
            "tabular_encoder": tab_out,
            "fusion": fused,
            "posterior": post,
            "tabular_decoder": tab_decoded,
            "ts_decoder": ts_decoded,
            "losses": {
                "kl": kl,
                "balance_loss": fused.get("balance_loss", torch.tensor(0.0, device=post["z"].device)),
            },
        }


__all__ = ["MultimodalVAE"]