import sys
from pathlib import Path

import pytest
import torch

from models.decoders.tabular_decoder import TabularDecoder


def test_numeric_only_decoder_forward():
    decoder = TabularDecoder(latent_dim=8, num_numeric_features=6, hidden_dims=(32,))
    z = torch.randn(4, 8)
    out = decoder(z)

    assert out["tab_num_recon"].shape == (4, 6)
    assert out["tab_cat_logits"] == []
    assert out["tab_cat_pred"] is None


def test_categorical_only_decoder_forward():
    decoder = TabularDecoder(latent_dim=8, categorical_cardinalities=[3, 4], hidden_dims=(32,))
    z = torch.randn(5, 8)
    out = decoder(z)

    assert out["tab_num_recon"] is None
    assert len(out["tab_cat_logits"]) == 2
    assert out["tab_cat_logits"][0].shape == (5, 3)
    assert out["tab_cat_logits"][1].shape == (5, 4)
    assert out["tab_cat_pred"].shape == (5, 2)


def test_mixed_decoder_forward():
    decoder = TabularDecoder(
        latent_dim=8,
        num_numeric_features=6,
        categorical_cardinalities=[3, 4],
        hidden_dims=(32, 16),
    )
    z = torch.randn(7, 8)
    out = decoder(z)

    assert out["tab_num_recon"].shape == (7, 6)
    assert len(out["tab_cat_logits"]) == 2
    assert out["tab_cat_pred"].shape == (7, 2)


def test_invalid_z_shape_raises():
    decoder = TabularDecoder(latent_dim=8, num_numeric_features=6, hidden_dims=(32,))
    bad_z = torch.randn(4, 7)

    with pytest.raises(ValueError, match="z must have shape"):
        decoder(bad_z)


def test_backward_runs():
    decoder = TabularDecoder(
        latent_dim=8,
        num_numeric_features=6,
        categorical_cardinalities=[3, 4],
        hidden_dims=(32,),
    )
    z = torch.randn(4, 8, requires_grad=True)
    out = decoder(z)

    loss = out["tab_num_recon"].mean()
    for logits in out["tab_cat_logits"]:
        loss = loss + logits.mean()
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
