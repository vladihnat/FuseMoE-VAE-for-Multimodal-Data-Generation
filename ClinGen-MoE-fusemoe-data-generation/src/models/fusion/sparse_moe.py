from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .router_utils import cv_squared, gates_to_load, topk_gates


TensorLikeInputs = Union[Mapping[str, torch.Tensor], Sequence[torch.Tensor]]


class ExpertMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FuseMoEFusion(nn.Module):
    """Minimal sparse MoE fusion block for multimodal pooled embeddings.

    Inspired by FuseMoE's sparse MoE: noisy top-k gating, router variants
    (`joint`, `permod`, `disjoint`), and a load-balancing auxiliary loss.
    This version is simplified for the MVP and fuses modality embeddings
    into a single pooled representation suitable for a latent variable model.

    Expected inputs:
        - dict[str, Tensor] or list[Tensor]
        - each tensor may have shape (batch, dim) or (batch, seq, dim)

    Returns:
        {
            "pooled": fused pooled representation, shape (batch, model_dim)
            "modality_tokens": fused modality tokens, shape (batch, num_modalities, model_dim)
            "projected_modalities": projected input tokens before MoE
            "gates": sparse routing weights
            "balance_loss": scalar auxiliary routing loss
        }
    """

    def __init__(
        self,
        modality_dims: Union[Mapping[str, int], Sequence[int]],
        model_dim: int = 32,
        num_experts: int = 4,
        expert_hidden_dim: int = 64,
        top_k: int = 2,
        router_type: str = "joint",
        gating: str = "softmax",
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2,
        dropout: float = 0.1,
        balance_loss_coef: float = 1e-2,
        activation: str = "gelu",
        pool: str = "mean",
    ) -> None:
        super().__init__()

        if isinstance(modality_dims, Mapping):
            self.modality_names = list(modality_dims.keys())
            dims = list(modality_dims.values())
        else:
            dims = list(modality_dims)
            self.modality_names = [f"modality_{i}" for i in range(len(dims))]

        if len(dims) == 0:
            raise ValueError("At least one modality must be provided")
        if router_type not in {"joint", "permod", "disjoint"}:
            raise ValueError("router_type must be one of: joint, permod, disjoint")
        if pool not in {"mean", "sum"}:
            raise ValueError("pool must be 'mean' or 'sum'")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if router_type == "disjoint" and num_experts % len(dims) != 0:
            raise ValueError("For disjoint routing, num_experts must be divisible by num_modalities")

        self.num_modalities = len(dims)
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.top_k = top_k
        self.router_type = router_type
        self.gating = gating
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon
        self.balance_loss_coef = balance_loss_coef
        self.pool = pool

        self.projectors = nn.ModuleDict({
            name: nn.Linear(dim, model_dim) if dim != model_dim else nn.Identity()
            for name, dim in zip(self.modality_names, dims)
        })

        self.experts = nn.ModuleList([
            ExpertMLP(
                input_dim=model_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=model_dim,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_experts)
        ])

        self.token_norm = nn.LayerNorm(model_dim)
        self.token_dropout = nn.Dropout(dropout)

        if router_type == "joint":
            gate_in_dim = model_dim * self.num_modalities
            if gating == "softmax":
                self.gate_proj = nn.Linear(gate_in_dim, num_experts)
                self.noise_proj = nn.Linear(gate_in_dim, num_experts)
            else:
                self.router_prototypes = nn.Parameter(torch.randn(num_experts, gate_in_dim))
                self.noise_proj = nn.Linear(gate_in_dim, num_experts)
        elif router_type == "permod":
            if gating == "softmax":
                self.gate_proj = nn.ModuleDict({
                    name: nn.Linear(model_dim, num_experts) for name in self.modality_names
                })
                self.noise_proj = nn.ModuleDict({
                    name: nn.Linear(model_dim, num_experts) for name in self.modality_names
                })
            else:
                self.router_prototypes = nn.ParameterDict({
                    name: nn.Parameter(torch.randn(num_experts, model_dim)) for name in self.modality_names
                })
                self.noise_proj = nn.ModuleDict({
                    name: nn.Linear(model_dim, num_experts) for name in self.modality_names
                })
        else:
            self.local_num_experts = num_experts // self.num_modalities
            if top_k > self.local_num_experts:
                raise ValueError(
                    f"For disjoint routing, top_k={top_k} cannot exceed local_num_experts={self.local_num_experts}"
                )
            if gating == "softmax":
                self.gate_proj = nn.ModuleDict({
                    name: nn.Linear(model_dim, self.local_num_experts) for name in self.modality_names
                })
                self.noise_proj = nn.ModuleDict({
                    name: nn.Linear(model_dim, self.local_num_experts) for name in self.modality_names
                })
            else:
                self.router_prototypes = nn.ParameterDict({
                    name: nn.Parameter(torch.randn(self.local_num_experts, model_dim))
                    for name in self.modality_names
                })
                self.noise_proj = nn.ModuleDict({
                    name: nn.Linear(model_dim, self.local_num_experts) for name in self.modality_names
                })

    def _canonicalize_inputs(
        self,
        inputs: TensorLikeInputs,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        if isinstance(inputs, Mapping):
            names = list(inputs.keys())
            tensors = list(inputs.values())
        else:
            names = self.modality_names
            tensors = list(inputs)

        if len(names) != self.num_modalities or len(tensors) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(tensors)}"
            )

        batch_size = None
        pooled = []
        for name, tensor in zip(names, tensors):
            if tensor.dim() == 3:
                tensor = tensor.mean(dim=1)
            elif tensor.dim() != 2:
                raise ValueError(
                    f"Each modality tensor must be 2D or 3D, got {tuple(tensor.shape)} for {name}"
                )

            if batch_size is None:
                batch_size = tensor.size(0)
            elif tensor.size(0) != batch_size:
                raise ValueError("All modalities must share the same batch size")

            pooled.append(self.projectors[name](tensor))
        return names, pooled

    def _distance_logits(self, x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        if self.gating == "laplace":
            return -torch.cdist(x, prototypes)
        if self.gating == "gaussian":
            return -torch.cdist(x, prototypes).pow(2)
        raise ValueError(f"Unsupported distance-based gating: {self.gating}")

    def _add_noise(self, logits: torch.Tensor, raw_noise: torch.Tensor) -> torch.Tensor:
        if not self.noisy_gating or not self.training:
            return logits
        noise_std = torch.nn.functional.softplus(raw_noise) + self.noise_epsilon
        return logits + torch.randn_like(logits) * noise_std

    def _joint_gates(self, tokens: torch.Tensor) -> torch.Tensor:
        context = tokens.reshape(tokens.size(0), -1)
        if self.gating == "softmax":
            logits = self.gate_proj(context)
        else:
            logits = self._distance_logits(context, self.router_prototypes)
        raw_noise = self.noise_proj(context)
        logits = self._add_noise(logits, raw_noise)
        gates, _ = topk_gates(logits, k=self.top_k, mode=self.gating)
        return gates

    def _permod_gates(self, names: List[str], tokens: torch.Tensor) -> torch.Tensor:
        gates = []
        for idx, name in enumerate(names):
            x = tokens[:, idx, :]
            if self.gating == "softmax":
                logits = self.gate_proj[name](x)
            else:
                logits = self._distance_logits(x, self.router_prototypes[name])
            raw_noise = self.noise_proj[name](x)
            logits = self._add_noise(logits, raw_noise)
            g, _ = topk_gates(logits, k=self.top_k, mode=self.gating)
            gates.append(g)
        return torch.stack(gates, dim=1)

    def _disjoint_gates(self, names: List[str], tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.size(0)
        gates = tokens.new_zeros(batch_size, self.num_modalities, self.num_experts)

        for idx, name in enumerate(names):
            x = tokens[:, idx, :]
            if self.gating == "softmax":
                logits = self.gate_proj[name](x)
            else:
                logits = self._distance_logits(x, self.router_prototypes[name])
            raw_noise = self.noise_proj[name](x)
            logits = self._add_noise(logits, raw_noise)
            local_gates, _ = topk_gates(logits, k=self.top_k, mode=self.gating)

            start = idx * self.local_num_experts
            end = start + self.local_num_experts
            gates[:, idx, start:end] = local_gates
        return gates

    def _balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        if gates.dim() == 2:
            importance = gates.sum(dim=0)
        else:
            importance = gates.sum(dim=(0, 1))
        load = gates_to_load(gates)
        return self.balance_loss_coef * (cv_squared(importance) + cv_squared(load))

    def forward(self, inputs: TensorLikeInputs) -> Dict[str, torch.Tensor]:
        names, pooled_inputs = self._canonicalize_inputs(inputs)
        tokens = torch.stack(pooled_inputs, dim=1)  # (batch, modalities, model_dim)

        if self.router_type == "joint":
            gates = self._joint_gates(tokens)  # (batch, experts)
            expert_outputs = torch.stack([expert(tokens) for expert in self.experts], dim=2)
            mixed = torch.sum(expert_outputs * gates[:, None, :, None], dim=2)
        elif self.router_type == "permod":
            gates = self._permod_gates(names, tokens)  # (batch, modalities, experts)
            expert_outputs = torch.stack([expert(tokens) for expert in self.experts], dim=2)
            mixed = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=2)
        else:
            gates = self._disjoint_gates(names, tokens)  # (batch, modalities, experts)
            expert_outputs = torch.stack([expert(tokens) for expert in self.experts], dim=2)
            mixed = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=2)

        fused_tokens = self.token_norm(tokens + self.token_dropout(mixed))
        if self.pool == "mean":
            pooled = fused_tokens.mean(dim=1)
        else:
            pooled = fused_tokens.sum(dim=1)

        return {
            "pooled": pooled,
            "modality_tokens": fused_tokens,
            "projected_modalities": tokens,
            "gates": gates,
            "balance_loss": self._balance_loss(gates),
        }


__all__ = ["ExpertMLP", "FuseMoEFusion"]
