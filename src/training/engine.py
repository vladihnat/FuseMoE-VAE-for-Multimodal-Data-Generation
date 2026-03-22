from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from losses.reconstruction import tabular_reconstruction_loss, ts_reconstruction_loss
from losses.total import multimodal_vae_total_loss


def move_batch_to_device(batch: Dict[str, object], device: Optional[torch.device]) -> Dict[str, object]:
    if device is None:
        return batch
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def compute_mvp_losses(
    batch: Dict[str, torch.Tensor],
    model_out: Dict[str, object],
    beta: float = 1.0,
    lambda_balance: float = 1.0,
    numeric_loss: str = "mse",
) -> Dict[str, torch.Tensor]:
    tab_recon = tabular_reconstruction_loss(
        target_num=batch.get("tab_num"),
        pred_num=model_out["tabular_decoder"]["tab_num_recon"],
        target_cat=batch.get("tab_cat"),
        pred_cat_logits=model_out["tabular_decoder"]["tab_cat_logits"],
        numeric_loss=numeric_loss,
    )

    ts_recon_loss = torch.zeros((), device=batch["ts_values"].device)
    ts_mask_loss = torch.zeros((), device=batch["ts_values"].device)
    if model_out["ts_decoder"] is not None:
        ts_dec = model_out["ts_decoder"]
        ts_recon = ts_reconstruction_loss(
            target_values=batch["ts_values"],
            pred_values=ts_dec["ts_recon"],
            mask=batch.get("ts_mask"),
            target_mask=batch.get("ts_mask") if ts_dec.get("ts_mask_logits") is not None else None,
            pred_mask_logits=ts_dec.get("ts_mask_logits"),
            value_loss=numeric_loss,
        )
        ts_recon_loss = ts_recon["val_loss"]
        ts_mask_loss = ts_recon["mask_loss"]

    reconstruction_loss = tab_recon["total"] + ts_recon_loss + ts_mask_loss

    total = multimodal_vae_total_loss(
        reconstruction_loss=reconstruction_loss,
        kl_loss=model_out["losses"]["kl"],
        balance_loss=model_out["losses"]["balance_loss"],
        beta=beta,
        lambda_balance=lambda_balance,
    )

    return {
        "num_reconstruction": tab_recon["num_loss"],
        "cat_reconstruction": tab_recon["cat_loss"],
        "ts_reconstruction": ts_recon_loss,
        "ts_mask_reconstruction": ts_mask_loss,
        "reconstruction": total["reconstruction"],
        "kl": total["kl"],
        "balance": total["balance"],
        "total": total["total"],
    }


def _detach_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in metrics.items()}


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    beta: float = 1.0,
    lambda_balance: float = 1.0,
    numeric_loss: str = "mse",
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    batch = move_batch_to_device(batch, device)

    optimizer.zero_grad(set_to_none=True)
    model_out = model(batch)
    losses = compute_mvp_losses(
        batch=batch,
        model_out=model_out,
        beta=beta,
        lambda_balance=lambda_balance,
        numeric_loss=numeric_loss,
    )
    losses["total"].backward()

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

    optimizer.step()
    return _detach_metrics(losses)


@torch.no_grad()
def evaluate_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
    beta: float = 1.0,
    lambda_balance: float = 1.0,
    numeric_loss: str = "mse",
) -> Dict[str, float]:
    model.eval()
    batch = move_batch_to_device(batch, device)
    model_out = model(batch, deterministic=True)
    losses = compute_mvp_losses(
        batch=batch,
        model_out=model_out,
        beta=beta,
        lambda_balance=lambda_balance,
        numeric_loss=numeric_loss,
    )
    return _detach_metrics(losses)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    beta: float = 1.0,
    lambda_balance: float = 1.0,
    numeric_loss: str = "mse",
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    running = None
    num_steps = 0

    for batch in dataloader:
        metrics = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            beta=beta,
            lambda_balance=lambda_balance,
            numeric_loss=numeric_loss,
            grad_clip_norm=grad_clip_norm,
        )
        if running is None:
            running = {k: 0.0 for k in metrics}
        for k, v in metrics.items():
            running[k] += v
        num_steps += 1

    if num_steps == 0:
        raise ValueError("Dataloader is empty")

    return {k: v / num_steps for k, v in running.items()}


__all__ = [
    "move_batch_to_device",
    "compute_mvp_losses",
    "train_step",
    "evaluate_step",
    "train_one_epoch",
]