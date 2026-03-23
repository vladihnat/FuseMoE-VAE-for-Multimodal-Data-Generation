import sys
from pathlib import Path

import pytest
import torch


def _load_decoder_class():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"Could not find src directory at: {src_path}")

    sys.path.insert(0, str(src_path))
    from models.decoders.TS_decoder import IrregularTSDecoder
    return IrregularTSDecoder


IrregularTSDecoder = _load_decoder_class()


def test_decoder_default_query_times():
    decoder = IrregularTSDecoder(latent_dim=8, output_dim=6, num_query_steps=24, hidden_dim=32, hidden_layers=())
    z = torch.randn(4, 8)
    out = decoder(z)

    assert out["ts_recon"].shape == (4, 24, 6)
    assert out["query_times"].shape == (4, 24)


def test_decoder_custom_query_times():
    decoder = IrregularTSDecoder(latent_dim=8, output_dim=6, hidden_dim=32, hidden_layers=())
    z = torch.randn(5, 8)
    query_times = torch.rand(5, 15)
    out = decoder(z, query_times=query_times)

    assert out["ts_recon"].shape == (5, 15, 6)
    assert out["query_times"].shape == (5, 15)


def test_invalid_z_shape_raises():
    decoder = IrregularTSDecoder(latent_dim=8, output_dim=6, hidden_dim=32, hidden_layers=())
    bad_z = torch.randn(4, 7)

    with pytest.raises(ValueError, match="z must have shape"):
        decoder(bad_z)


def test_invalid_target_times_shape_raises():
    decoder = IrregularTSDecoder(latent_dim=8, output_dim=6, hidden_dim=32, hidden_layers=())
    z = torch.randn(4, 8)
    bad_times = torch.rand(3, 15)

    with pytest.raises(ValueError, match="query_times must have shape"):
        decoder(z, query_times=bad_times)


def test_backward_runs():
    decoder = IrregularTSDecoder(latent_dim=8, output_dim=6, hidden_dim=32, hidden_layers=())
    z = torch.randn(4, 8, requires_grad=True)
    out = decoder(z)

    loss = out["ts_recon"].mean()
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
