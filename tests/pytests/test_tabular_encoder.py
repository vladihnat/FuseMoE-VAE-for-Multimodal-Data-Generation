
import importlib.util
from pathlib import Path

import pytest
import torch


def _load_encoder_class():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "models" / "encoders" / "tabular.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find encoder file at: {module_path}")

    spec = importlib.util.spec_from_file_location("tabular", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.TabularEncoder


TabularEncoder = _load_encoder_class()


def test_numeric_only_forward_shapes():
    encoder = TabularEncoder(
        num_numeric_features=6,
        hidden_dims=(32,),
        output_dim=16,
    )
    x_num = torch.randn(4, 6)

    out = encoder(x_num=x_num)

    assert set(out.keys()) == {"pooled", "sequence"}
    assert out["pooled"].shape == (4, 16)
    assert out["sequence"].shape == (4, 1, 16)
    assert torch.isfinite(out["pooled"]).all()
    assert torch.isfinite(out["sequence"]).all()


def test_categorical_only_forward_shapes():
    encoder = TabularEncoder(
        num_numeric_features=0,
        categorical_cardinalities=[5, 10, 4],
        hidden_dims=(32,),
        output_dim=12,
    )
    x_cat = torch.tensor([[0, 1, 2], [1, 3, 0], [4, 5, 1], [2, 9, 3]])

    out = encoder(x_cat=x_cat)

    assert out["pooled"].shape == (4, 12)
    assert out["sequence"].shape == (4, 1, 12)


def test_mixed_forward_shapes():
    encoder = TabularEncoder(
        num_numeric_features=3,
        categorical_cardinalities=[5, 10],
        hidden_dims=(32, 16),
        output_dim=14,
    )
    x_num = torch.randn(4, 3)
    x_cat = torch.tensor([[0, 1], [1, 3], [4, 5], [2, 9]])

    out = encoder(x_num=x_num, x_cat=x_cat)

    assert out["pooled"].shape == (4, 14)
    assert out["sequence"].shape == (4, 1, 14)


def test_numeric_shape_mismatch_raises():
    encoder = TabularEncoder(
        num_numeric_features=3,
        hidden_dims=(16,),
        output_dim=8,
    )
    bad_x_num = torch.randn(4, 4)

    with pytest.raises(ValueError, match="x_num must have shape"):
        encoder(x_num=bad_x_num)


def test_categorical_shape_mismatch_raises():
    encoder = TabularEncoder(
        num_numeric_features=0,
        categorical_cardinalities=[5, 10],
        hidden_dims=(16,),
        output_dim=8,
    )
    bad_x_cat = torch.tensor([[0, 1, 2], [1, 3, 0]])

    with pytest.raises(ValueError, match="x_cat must have shape"):
        encoder(x_cat=bad_x_cat)


def test_out_of_range_category_raises():
    encoder = TabularEncoder(
        num_numeric_features=0,
        categorical_cardinalities=[5, 10],
        hidden_dims=(16,),
        output_dim=8,
    )
    x_cat = torch.tensor([[0, 1], [5, 3]])

    with pytest.raises(ValueError, match="contains indices outside"):
        encoder(x_cat=x_cat)


def test_missing_required_numeric_input_raises():
    encoder = TabularEncoder(
        num_numeric_features=2,
        hidden_dims=(16,),
        output_dim=8,
    )

    with pytest.raises(ValueError, match="x_num must be provided"):
        encoder()


def test_missing_required_categorical_input_raises():
    encoder = TabularEncoder(
        num_numeric_features=0,
        categorical_cardinalities=[4, 6],
        hidden_dims=(16,),
        output_dim=8,
    )

    with pytest.raises(ValueError, match="x_cat must be provided"):
        encoder()


def test_batch_size_mismatch_raises():
    encoder = TabularEncoder(
        num_numeric_features=2,
        categorical_cardinalities=[4, 6],
        hidden_dims=(16,),
        output_dim=8,
    )
    x_num = torch.randn(4, 2)
    x_cat = torch.tensor([[0, 1], [1, 2], [2, 3]])

    with pytest.raises(ValueError, match="same batch size"):
        encoder(x_num=x_num, x_cat=x_cat)


def test_backward_pass_runs():
    encoder = TabularEncoder(
        num_numeric_features=3,
        categorical_cardinalities=[5, 10],
        hidden_dims=(32, 16),
        output_dim=14,
    )
    x_num = torch.randn(4, 3, requires_grad=True)
    x_cat = torch.tensor([[0, 1], [1, 3], [4, 5], [2, 9]])

    out = encoder(x_num=x_num, x_cat=x_cat)
    loss = out["pooled"].mean() + out["sequence"].mean()
    loss.backward()

    assert x_num.grad is not None
    assert torch.isfinite(x_num.grad).all()
