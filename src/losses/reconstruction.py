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
        pred_values: Predicted values of shape (batch, seq_len, dim).
        target_values: Ground truth values of shape (batch, seq_len, dim).
        mask: Optional boolean or float mask of shape (batch, seq_len) or (batch, seq_len, dim).
              1.0 indicates an observed value, 0.0 indicates missing/padded.
        reduction: "mean" or "sum".
        loss_type: "mse" or "l1".
    """
    if pred_values.shape != target_values.shape:
        raise ValueError(
            f"pred_values and target_values must have the same shape, "
            f"got {tuple(pred_values.shape)} and {tuple(target_values.shape)}"
        )

    # Compute unreduced loss
    if loss_type == "mse":
        loss = F.mse_loss(pred_values, target_values, reduction="none")
    elif loss_type == "l1":
        loss = F.l1_loss(pred_values, target_values, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 2:
            # Expand (batch, seq_len) to (batch, seq_len, dim)
            mask = mask.unsqueeze(-1).expand_as(loss)
        elif mask.shape != loss.shape:
            raise ValueError(
                f"mask must have shape {tuple(loss.shape)} or (batch, seq_len); "
                f"got {tuple(mask.shape)}"
            )
        
        # Zero out loss where mask is 0
        loss = loss * mask.float()

        if reduction == "mean":
            # Compute mean only over the observed elements
            num_observed = mask.sum().clamp_min(1.0)
            return loss.sum() / num_observed
        elif reduction == "sum":
            return loss.sum()
    
    # If no mask, standard reduction applies
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


__all__ = ["tabular_reconstruction_loss", "timeseries_reconstruction_loss"]
