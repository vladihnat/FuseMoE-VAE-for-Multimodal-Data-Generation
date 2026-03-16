from pathlib import Path
import sys

import torch


def _load_fusion_class():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    if not src_path.exists():
        raise FileNotFoundError(f"Could not find src directory at: {src_path}")

    sys.path.insert(0, str(src_path))

    # from src.models.fusion.sparse_moe import FuseMoEFusion
    from models.fusion.sparse_moe import FuseMoEFusion
    return FuseMoEFusion


FuseMoEFusion = _load_fusion_class()


def _make_inputs(batch_size=4, model_dim=32):
    torch.manual_seed(0)
    return {
        "ts": torch.randn(batch_size, model_dim),
        "tab": torch.randn(batch_size, model_dim),
    }


def test_joint_router_forward():
    fusion = FuseMoEFusion(
        modality_dims={"ts": 32, "tab": 32},
        model_dim=32,
        num_experts=4,
        top_k=2,
        router_type="joint",
    )
    out = fusion(_make_inputs())

    assert out["pooled"].shape == (4, 32)
    assert out["modality_tokens"].shape == (4, 2, 32)
    assert out["gates"].shape == (4, 4)
    assert out["balance_loss"].ndim == 0
    assert torch.isfinite(out["pooled"]).all()


def test_permod_router_forward():
    fusion = FuseMoEFusion(
        modality_dims={"ts": 32, "tab": 32},
        model_dim=32,
        num_experts=6,
        top_k=2,
        router_type="permod",
    )
    out = fusion(_make_inputs())

    assert out["pooled"].shape == (4, 32)
    assert out["modality_tokens"].shape == (4, 2, 32)
    assert out["gates"].shape == (4, 2, 6)
    assert out["balance_loss"].ndim == 0


def test_disjoint_router_forward():
    fusion = FuseMoEFusion(
        modality_dims={"ts": 32, "tab": 32},
        model_dim=32,
        num_experts=6,
        top_k=2,
        router_type="disjoint",
    )
    out = fusion(_make_inputs())

    assert out["pooled"].shape == (4, 32)
    assert out["modality_tokens"].shape == (4, 2, 32)
    assert out["gates"].shape == (4, 2, 6)
    assert out["balance_loss"].ndim == 0


def test_accepts_sequence_inputs():
    fusion = FuseMoEFusion(
        modality_dims={"ts": 32, "tab": 32},
        model_dim=32,
        num_experts=4,
        top_k=2,
        router_type="joint",
    )

    inputs = {
        "ts": torch.randn(3, 12, 32),
        "tab": torch.randn(3, 1, 32),
    }
    out = fusion(inputs)

    assert out["pooled"].shape == (3, 32)
    assert out["modality_tokens"].shape == (3, 2, 32)


def test_backward_runs():
    fusion = FuseMoEFusion(
        modality_dims={"ts": 32, "tab": 32},
        model_dim=32,
        num_experts=4,
        top_k=2,
        router_type="joint",
    )

    inputs = {
        "ts": torch.randn(4, 32, requires_grad=True),
        "tab": torch.randn(4, 32, requires_grad=True),
    }
    out = fusion(inputs)
    loss = out["pooled"].mean() + out["balance_loss"]
    loss.backward()

    assert inputs["ts"].grad is not None
    assert inputs["tab"].grad is not None
    assert torch.isfinite(inputs["ts"].grad).all()
    assert torch.isfinite(inputs["tab"].grad).all()
