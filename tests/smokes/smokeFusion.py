from src.data.datasets import make_synthetic_ts_tab_dataloader
from src.models.encoders.ts_irregular import TSIrregularEncoder
from src.models.encoders.tabular import TabularEncoder
from src.models.fusion.sparse_moe import FuseMoEFusion

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

ts_out = ts_encoder(
    values=batch["ts_values"],
    mask=batch["ts_mask"],
    times=batch["ts_times"],
)

tab_out = tab_encoder(
    x_num=batch["tab_num"],
    x_cat=batch["tab_cat"],
)

fusion = FuseMoEFusion(
    modality_dims={"ts": 32, "tab": 32},
    model_dim=32,
    num_experts=4,
    top_k=2,
    router_type="joint",
)

fused = fusion({
    "ts": ts_out["pooled"],
    "tab": tab_out["pooled"],
})

print(fused["pooled"].shape)
print(fused["modality_tokens"].shape)
print(fused["balance_loss"])