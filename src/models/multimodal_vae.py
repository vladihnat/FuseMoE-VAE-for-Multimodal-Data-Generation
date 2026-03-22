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
        # 6. Time-Series Decoding
        ts_decoded = (
            self.ts_decoder(post["z"], query_times=batch.get("ts_times"))
            if self.ts_decoder is not None
            else None
        )

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

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        query_times: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, object]:
        """Sample z ~ N(0, I) and decode into synthetic tabular and TS data.

        Args:
            n_samples:    number of synthetic samples to generate
            query_times:  (n_samples, num_steps) or (num_steps,) time grid for TS decoding;
                          if None the decoder uses its default uniform grid
            device:       target device; defaults to the model's current device

        Returns:
            {
                "z":           (n_samples, latent_dim)
                "tab_num":     (n_samples, num_numeric)   reconstructed numeric features
                "tab_cat":     (n_samples, num_categorical) predicted category indices
                "ts_recon":    (n_samples, num_steps, ts_dim) or None
                "ts_mask":     (n_samples, num_steps, ts_dim) binarized mask or None
                "query_times": (n_samples, num_steps) time grid used or None
            }
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(n_samples, self.posterior.latent_dim, device=device)

        tab_decoded = self.tabular_decoder(z)

        ts_recon      = None
        ts_mask       = None
        out_query_times = None
        if self.ts_decoder is not None:
            ts_decoded = self.ts_decoder(z, query_times=query_times)
            ts_recon   = ts_decoded["ts_recon"]
            out_query_times = ts_decoded.get("query_times", ts_decoded.get("target_times"))
            mask_logits = ts_decoded.get("ts_mask_logits")
            if mask_logits is not None:
                ts_mask = (mask_logits.sigmoid() > 0.5).float()

        return {
            "z":           z,
            "tab_num":     tab_decoded["tab_num_recon"],
            "tab_cat":     tab_decoded["tab_cat_pred"],
            "ts_recon":    ts_recon,
            "ts_mask":     ts_mask,
            "query_times": out_query_times,
        }


__all__ = ["MultimodalVAE"]