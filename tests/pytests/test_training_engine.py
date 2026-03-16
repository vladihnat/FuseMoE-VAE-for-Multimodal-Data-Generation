import sys
from pathlib import Path

import torch


def _add_src_to_path():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"Could not find src directory at: {src_path}")
    sys.path.insert(0, str(src_path))


_add_src_to_path()

from models.decoders.tabular_decoder import TabularDecoder
from models.encoders.tabular import TabularEncoder
from models.encoders.ts_irregular import TSIrregularEncoder
from models.fusion.sparse_moe import FuseMoEFusion
from models.latent.posterior import PosteriorHead
from models.multimodal_vae import MultimodalVAE
from training.engine import evaluate_step, train_step


def _make_model():
    ts_encoder = TSIrregularEncoder(input_dim=5, hidden_dim=32, embed_time=16, num_heads=4, num_query_steps=12)
    tab_encoder = TabularEncoder(
        num_numeric_features=6,
        categorical_cardinalities=[3, 4],
        hidden_dims=(32,),
        output_dim=32,
    )
    fusion = FuseMoEFusion(
        modality_dims={"ts": 32, "tab": 32},
        model_dim=32,
        num_experts=4,
        top_k=2,
        router_type="joint",
    )
    posterior = PosteriorHead(input_dim=32, latent_dim=8)
    decoder = TabularDecoder(
        latent_dim=8,
        num_numeric_features=6,
        categorical_cardinalities=[3, 4],
        hidden_dims=(32,),
    )
    return MultimodalVAE(ts_encoder, tab_encoder, fusion, posterior, decoder)


def _make_batch(batch_size=4, seq_len=17):
    torch.manual_seed(0)
    return {
        "ts_values": torch.randn(batch_size, seq_len, 5),
        "ts_times": torch.sort(torch.rand(batch_size, seq_len), dim=1).values,
        "ts_mask": (torch.rand(batch_size, seq_len, 5) > 0.2).float(),
        "tab_num": torch.randn(batch_size, 6),
        "tab_cat": torch.stack([
            torch.randint(0, 3, (batch_size,)),
            torch.randint(0, 4, (batch_size,)),
        ], dim=1),
    }


def test_train_step_returns_finite_metrics_and_updates_params():
    model = _make_model()
    batch = _make_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = model.posterior.mu_head.weight.detach().clone()
    metrics = train_step(model, batch, optimizer)

    assert set(metrics.keys()) == {
        "num_reconstruction",
        "cat_reconstruction",
        "reconstruction",
        "kl",
        "balance",
        "total",
    }
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

    after = model.posterior.mu_head.weight.detach()
    assert not torch.allclose(before, after)


def test_evaluate_step_returns_finite_metrics():
    model = _make_model()
    batch = _make_batch()
    metrics = evaluate_step(model, batch)

    assert set(metrics.keys()) == {
        "num_reconstruction",
        "cat_reconstruction",
        "reconstruction",
        "kl",
        "balance",
        "total",
    }
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
