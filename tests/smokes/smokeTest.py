from src.data.datasets import make_synthetic_ts_tab_dataloader
from src.models.encoders.ts_irregular import TSIrregularEncoder
from src.models.encoders.tabular import TabularEncoder

# 1) Build dataloader
loader = make_synthetic_ts_tab_dataloader(
    "data/processed/synthetic_ts_tab/train.pkl",
    batch_size=8,
    shuffle=False,
    return_latent=False,
)

# 2) Get one batch
batch = next(iter(loader))

print("Batch shapes:")
print("ts_values:", batch["ts_values"].shape)
print("ts_times:", batch["ts_times"].shape)
print("ts_mask:", batch["ts_mask"].shape)
print("tab_num:", batch["tab_num"].shape)
print("tab_cat:", batch["tab_cat"].shape)

# 3) Create encoders
ts_encoder = TSIrregularEncoder(
    input_dim=batch["ts_values"].shape[-1],
    hidden_dim=32,
    embed_time=16,
    num_heads=4,
    num_query_steps=12,
)

tab_encoder = TabularEncoder(
    num_numeric_features=batch["tab_num"].shape[-1],
    categorical_cardinalities=[3, 4],   # from metadata.json
    hidden_dims=(32,),
    output_dim=32,
)

# 4) Forward pass
ts_out = ts_encoder(
    values=batch["ts_values"],
    mask=batch["ts_mask"],
    times=batch["ts_times"],
)

tab_out = tab_encoder(
    x_num=batch["tab_num"],
    x_cat=batch["tab_cat"],
)

print("\\nEncoder outputs:")
print("ts sequence:", ts_out["sequence"].shape)
print("ts pooled:", ts_out["pooled"].shape)
print("tab sequence:", tab_out["sequence"].shape)
print("tab pooled:", tab_out["pooled"].shape)
