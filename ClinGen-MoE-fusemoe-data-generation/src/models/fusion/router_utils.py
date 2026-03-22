from typing import Tuple

import torch


def cv_squared(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Squared coefficient of variation used as a load-balancing penalty."""
    if x.numel() <= 1:
        return x.new_tensor(0.0)
    x = x.float()
    return x.var(unbiased=False) / (x.mean().pow(2) + eps)


def gates_to_load(gates: torch.Tensor) -> torch.Tensor:
    """Count how many items route to each expert.

    Supports gates of shape:
        - (batch, experts)
        - (batch, modalities, experts)
    """
    dims_to_sum = tuple(range(gates.dim() - 1))
    return (gates > 0).sum(dim=dims_to_sum).to(gates.dtype)


def topk_gates(logits: torch.Tensor, k: int, mode: str = "softmax") -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn logits into sparse top-k gates and return selected expert indices."""
    if logits.dim() not in (2, 3):
        raise ValueError(f"logits must be 2D or 3D, got shape {tuple(logits.shape)}")

    num_experts = logits.size(-1)
    if not 1 <= k <= num_experts:
        raise ValueError(f"k must be in [1, {num_experts}], got {k}")

    top_vals, top_idx = torch.topk(logits, k=k, dim=-1)

    if mode == "softmax":
        top_gates = torch.softmax(top_vals, dim=-1)
    elif mode in {"laplace", "gaussian"}:
        top_gates = torch.exp(top_vals - torch.logsumexp(top_vals, dim=-1, keepdim=True))
    else:
        raise ValueError(f"Unsupported gating mode: {mode}")

    gates = torch.zeros_like(logits)
    gates.scatter_(-1, top_idx, top_gates)
    return gates, top_idx
