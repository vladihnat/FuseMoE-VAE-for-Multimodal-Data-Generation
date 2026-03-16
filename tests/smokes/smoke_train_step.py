import torch

from src.data.datasets import make_synthetic_ts_tab_dataloader
from src.models.decoders.tabular_decoder import TabularDecoder
from src.models.encoders.tabular import TabularEncoder
from src.models.encoders.ts_irregular import TSIrregularEncoder
from src.models.fusion.sparse_moe import FuseMoEFusion
from src.models.latent.posterior import PosteriorHead
from src.models.multimodal_vae import MultimodalVAE
from src.training.engine import evaluate_step, train_step


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

model = MultimodalVAE(
    ts_encoder=ts_encoder,
    tabular_encoder=tab_encoder,
    fusion=fusion,
    posterior=posterior,
    tabular_decoder=tabular_decoder,
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
