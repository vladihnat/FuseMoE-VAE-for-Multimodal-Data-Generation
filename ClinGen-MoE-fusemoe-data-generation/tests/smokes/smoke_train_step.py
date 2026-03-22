import sys
from pathlib import Path

CURRENT = Path(__file__).resolve()
for parent in [CURRENT.parent, *CURRENT.parents]:
    src = parent / "src"
    if src.exists():
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        break
else:
    raise FileNotFoundError("Could not find src directory")
import torch

from data.datasets import make_synthetic_ts_tab_dataloader
from models.decoders.tabular_decoder import TabularDecoder
from models.decoders.timeseries_decoder import TSIrregularDecoder
from models.encoders.tabular import TabularEncoder
from models.encoders.ts_irregular import TSIrregularEncoder
from models.fusion.sparse_moe import FuseMoEFusion
from models.latent.posterior import PosteriorHead
from models.multimodal_vae import MultimodalVAE
from training.engine import evaluate_step, train_step


loader = make_synthetic_ts_tab_dataloader(
    "data/processed/synthetic_ts_tab/train.pkl",
    batch_size=8,
    shuffle=False,
)
batch = next(iter(loader))

ts_encoder = TSIrregularEncoder(
    input_dim=batch["ts_values"].shape[-1],
    hidden_dim=32,
    embed_time=16,
    num_heads=4,
    num_query_steps=12,
)
tab_encoder = TabularEncoder(
    num_numeric_features=batch["tab_num"].shape[-1],
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

tabular_decoder = TabularDecoder(
    latent_dim=8,
    num_numeric_features=batch["tab_num"].shape[-1],
    categorical_cardinalities=[3, 4],
    hidden_dims=(32,),
)

# Initialize the new Time-Series Decoder
ts_decoder = TSIrregularDecoder(
    latent_dim=8,
    output_dim=batch["ts_values"].shape[-1],
    embed_time=16,
    hidden_dims=(32, 32),
    default_seq_len=batch["ts_values"].shape[1],
)

# Pass both decoders to the MultimodalVAE
model = MultimodalVAE(
    ts_encoder=ts_encoder,
    tabular_encoder=tab_encoder,
    fusion=fusion,
    posterior=posterior,
    tabular_decoder=tabular_decoder,
    ts_decoder=ts_decoder,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_metrics = train_step(model, batch, optimizer)
eval_metrics = evaluate_step(model, batch)

print("Train step metrics:")
for k, v in train_metrics.items():
    print(f"{k}: {v:.6f}")

print("\nEval step metrics:")
for k, v in eval_metrics.items():
    print(f"{k}: {v:.6f}")