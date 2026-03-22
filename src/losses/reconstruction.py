from typing import Dict, Literal, Optional, Sequence

import torch
import torch.nn.functional as F


Reduction = Literal["mean", "sum"]


def tabular_reconstruction_loss(
    target_num: Optional[torch.Tensor],
    pred_num: Optional[torch.Tensor],
    target_cat: Optional[torch.Tensor],
    pred_cat_logits: Optional[Sequence[torch.Tensor]],
    reduction: Reduction = "mean",
    numeric_loss: Literal["mse", "l1"] = "mse",
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction losses for mixed tabular data."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    device = None
    for tensor in (target_num, pred_num, target_cat):
        if tensor is not None:
            device = tensor.device
            break
    if device is None and pred_cat_logits:
        device = pred_cat_logits[0].device
    if device is None:
        raise ValueError("At least one target/prediction tensor must be provided")

    num_loss = torch.zeros((), device=device)
    cat_loss = torch.zeros((), device=device)

    if target_num is not None or pred_num is not None:
        if target_num is None or pred_num is None:
            raise ValueError("target_num and pred_num must either both be provided or both be None")
        if target_num.shape != pred_num.shape:
            raise ValueError(
                f"target_num and pred_num must have the same shape, got {tuple(target_num.shape)} and {tuple(pred_num.shape)}"
            )
        if target_num.dim() != 2:
            raise ValueError(f"target_num and pred_num must be 2D, got {tuple(target_num.shape)}")

        if numeric_loss == "mse":
            num_loss = F.mse_loss(pred_num, target_num, reduction=reduction)
        elif numeric_loss == "l1":
            num_loss = F.l1_loss(pred_num, target_num, reduction=reduction)
        else:
            raise ValueError(f"Unsupported numeric_loss: {numeric_loss}")

    if target_cat is not None or pred_cat_logits is not None:
        if target_cat is None or pred_cat_logits is None:
            raise ValueError("target_cat and pred_cat_logits must either both be provided or both be None")
        if target_cat.dim() != 2:
            raise ValueError(f"target_cat must be 2D, got {tuple(target_cat.shape)}")
        if len(pred_cat_logits) != target_cat.size(1):
            raise ValueError(
                f"Expected {target_cat.size(1)} categorical heads, got {len(pred_cat_logits)}"
            )

        losses = []
        for j, logits in enumerate(pred_cat_logits):
            if logits.dim() != 2:
                raise ValueError(f"Categorical logits at index {j} must be 2D, got {tuple(logits.shape)}")
            if logits.size(0) != target_cat.size(0):
                raise ValueError(
                    f"Batch size mismatch for categorical head {j}: got {logits.size(0)} and {target_cat.size(0)}"
                )
            losses.append(F.cross_entropy(logits, target_cat[:, j], reduction=reduction))

        if losses:
            cat_loss = sum(losses)

    total = num_loss + cat_loss
    return {
        "num_loss": num_loss,
        "cat_loss": cat_loss,
        "total": total,
    }


def timeseries_reconstruction_loss(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: Reduction = "mean",
    loss_type: Literal["mse", "l1"] = "mse",
) -> torch.Tensor:
    """Compute reconstruction loss for irregular time-series data.

    Args:
        pred_values:   (batch, seq_len, dim)  predicted values
        target_values: (batch, seq_len, dim)  ground-truth values
        mask:          (batch, seq_len) or (batch, seq_len, dim)  1 = observed, 0 = missing
        reduction:     "mean" or "sum"
        loss_type:     "mse" or "l1"
    """
    if pred_values.shape != target_values.shape:
        raise ValueError(
            f"pred_values and target_values must have the same shape, "
            f"got {tuple(pred_values.shape)} and {tuple(target_values.shape)}"
        )

    if loss_type == "mse":
        loss = F.mse_loss(pred_values, target_values, reduction="none")
    elif loss_type == "l1":
        loss = F.l1_loss(pred_values, target_values, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand_as(loss)
        elif mask.shape != loss.shape:
            raise ValueError(
                f"mask must have shape {tuple(loss.shape)} or (batch, seq_len); "
                f"got {tuple(mask.shape)}"
            )
        loss = loss * mask.float()
        if reduction == "mean":
            num_observed = mask.sum().clamp_min(1.0)
            return loss.sum() / num_observed
        return loss.sum()

    if reduction == "mean":
        return loss.mean()
    return loss.sum()


def ts_reconstruction_loss(
    target_values: torch.Tensor,
    pred_values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    pred_mask_logits: Optional[torch.Tensor] = None,
    reduction: Reduction = "mean",
    value_loss: Literal["mse", "l1"] = "mse",
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction losses for irregular time-series data.

    Assumes the decoder was called with query_times matching the target time steps,
    so pred_values and target_values align on the time axis.

    The value loss is computed only at observed positions (mask == 1) when a mask
    is provided. The mask loss (optional) is binary cross-entropy between
    pred_mask_logits and target_mask.

    Args:
        target_values:    (batch, num_steps, output_dim)  ground-truth values
        pred_values:      (batch, num_steps, output_dim)  decoder output
        mask:             (batch, num_steps, output_dim) or (batch, num_steps)
                          binary mask of observed positions; if None all positions count
        target_mask:      (batch, num_steps, output_dim)  ground-truth binary mask
                          required when pred_mask_logits is provided
        pred_mask_logits: (batch, num_steps, output_dim)  raw logits from mask head
        reduction:        "mean" or "sum" over observed elements
        value_loss:       "mse" or "l1" element-wise loss for values
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    if value_loss not in {"mse", "l1"}:
        raise ValueError(f"Unsupported value_loss: {value_loss}")

    if target_values.shape != pred_values.shape:
        raise ValueError(
            f"target_values and pred_values must have the same shape, "
            f"got {tuple(target_values.shape)} and {tuple(pred_values.shape)}"
        )
    if target_values.dim() != 3:
        raise ValueError(f"target_values and pred_values must be 3D, got {tuple(target_values.shape)}")

    device = target_values.device

    # --- value loss (masked) ---
    if value_loss == "mse":
        elementwise = F.mse_loss(pred_values, target_values, reduction="none")
    else:
        elementwise = F.l1_loss(pred_values, target_values, reduction="none")

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand_as(target_values)
        if mask.shape != target_values.shape:
            raise ValueError(
                f"mask must have shape {tuple(target_values.shape)} or "
                f"(batch, num_steps); got {tuple(mask.shape)}"
            )
        mask = mask.float()
        num_observed = mask.sum().clamp_min(1.0)
        masked = elementwise * mask
        val_loss = masked.sum() / num_observed if reduction == "mean" else masked.sum()
    else:
        val_loss = elementwise.mean() if reduction == "mean" else elementwise.sum()

    # --- mask loss (optional) ---
    mask_loss = torch.zeros((), device=device)

    if target_mask is not None or pred_mask_logits is not None:
        if target_mask is None or pred_mask_logits is None:
            raise ValueError(
                "target_mask and pred_mask_logits must either both be provided or both be None"
            )
        if target_mask.shape != pred_mask_logits.shape:
            raise ValueError(
                f"target_mask and pred_mask_logits must have the same shape, "
                f"got {tuple(target_mask.shape)} and {tuple(pred_mask_logits.shape)}"
            )
        if target_mask.dim() != 3:
            raise ValueError(f"target_mask must be 3D, got {tuple(target_mask.shape)}")

        mask_loss = F.binary_cross_entropy_with_logits(
            pred_mask_logits, target_mask.float(), reduction=reduction
        )

    total = val_loss + mask_loss
    return {
        "val_loss": val_loss,
        "mask_loss": mask_loss,
        "total": total,
    }


__all__ = ["tabular_reconstruction_loss", "timeseries_reconstruction_loss", "ts_reconstruction_loss"]
