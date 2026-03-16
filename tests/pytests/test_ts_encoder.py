
import importlib.util
from pathlib import Path

import pytest
import torch


def _load_encoder_class():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "encoders" / "ts_irregular.py"
    
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find encoder file at: {module_path}")

    spec = importlib.util.spec_from_file_location("ts_irregular", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TSIrregularEncoder


TSIrregularEncoder = _load_encoder_class()


def _make_batch(batch_size=4, seq_len=12, input_dim=5):
    torch.manual_seed(0)
    values = torch.randn(batch_size, seq_len, input_dim)
    mask = (torch.rand(batch_size, seq_len, input_dim) > 0.2).float()
    times = torch.sort(torch.rand(batch_size, seq_len), dim=1).values
    return values, mask, times


def test_forward_shapes_with_feature_mask():
    encoder = TSIrregularEncoder(
        input_dim=5,
        hidden_dim=16,
        embed_time=8,
        num_heads=2,
        num_query_steps=10,
    )
    values, mask, times = _make_batch(batch_size=3, seq_len=11, input_dim=5)

    out = encoder(values=values, mask=mask, times=times)

    assert set(out.keys()) == {"sequence", "pooled", "query_times"}
    assert out["sequence"].shape == (3, 10, 16)
    assert out["pooled"].shape == (3, 16)
    assert out["query_times"].shape == (3, 10)
    assert torch.isfinite(out["sequence"]).all()
    assert torch.isfinite(out["pooled"]).all()


def test_forward_shapes_with_step_mask():
    encoder = TSIrregularEncoder(
        input_dim=5,
        hidden_dim=12,
        embed_time=8,
        num_heads=2,
        num_query_steps=7,
    )
    values, mask, times = _make_batch(batch_size=2, seq_len=9, input_dim=5)
    step_mask = mask.any(dim=-1).float()

    out = encoder(values=values, mask=step_mask, times=times)

    assert out["sequence"].shape == (2, 7, 12)
    assert out["pooled"].shape == (2, 12)


def test_custom_query_times_shape():
    encoder = TSIrregularEncoder(
        input_dim=4,
        hidden_dim=10,
        embed_time=8,
        num_heads=2,
        num_query_steps=6,
    )
    values, mask, times = _make_batch(batch_size=2, seq_len=8, input_dim=4)
    custom_query_times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    out = encoder(values=values, mask=mask, times=times, query_times=custom_query_times)

    assert out["sequence"].shape == (2, 5, 10)
    assert out["query_times"].shape == (2, 5)
    assert torch.allclose(out["query_times"][0], custom_query_times, atol=1e-6)


def test_all_missing_sample_returns_zero_representation():
    encoder = TSIrregularEncoder(
        input_dim=3,
        hidden_dim=8,
        embed_time=8,
        num_heads=2,
        num_query_steps=6,
    )
    values, mask, times = _make_batch(batch_size=2, seq_len=7, input_dim=3)
    mask[0] = 0.0

    out = encoder(values=values, mask=mask, times=times)

    assert torch.allclose(out["sequence"][0], torch.zeros_like(out["sequence"][0]), atol=1e-6)
    assert torch.allclose(out["pooled"][0], torch.zeros_like(out["pooled"][0]), atol=1e-6)


def test_invalid_mask_shape_raises_value_error():
    encoder = TSIrregularEncoder(
        input_dim=5,
        hidden_dim=16,
        embed_time=8,
        num_heads=2,
        num_query_steps=10,
    )
    values, _, times = _make_batch(batch_size=3, seq_len=11, input_dim=5)
    bad_mask = torch.ones(3, 11, 4)

    with pytest.raises(ValueError, match="mask must have shape"):
        encoder(values=values, mask=bad_mask, times=times)


def test_invalid_times_shape_raises_value_error():
    encoder = TSIrregularEncoder(
        input_dim=5,
        hidden_dim=16,
        embed_time=8,
        num_heads=2,
        num_query_steps=10,
    )
    values, mask, _ = _make_batch(batch_size=3, seq_len=11, input_dim=5)
    bad_times = torch.randn(3, 10)

    with pytest.raises(ValueError, match="values and times must agree"):
        encoder(values=values, mask=mask, times=bad_times)


def test_backward_pass_runs():
    encoder = TSIrregularEncoder(
        input_dim=5,
        hidden_dim=16,
        embed_time=8,
        num_heads=2,
        num_query_steps=10,
    )
    values, mask, times = _make_batch(batch_size=3, seq_len=11, input_dim=5)
    values.requires_grad_(True)

    out = encoder(values=values, mask=mask, times=times)
    loss = out["sequence"].mean() + out["pooled"].mean()
    loss.backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
