from src.data.datasets import make_synthetic_ts_tab_dataloader
from src.models.encoders.ts_irregular import TSIrregularEncoder
from src.models.encoders.tabular import TabularEncoder
from src.models.fusion.sparse_moe import FuseMoEFusion
from src.models.latent.posterior import PosteriorHead
from src.models.decoders.tabular_decoder import TabularDecoder
from src.models.multimodal_vae import MultimodalVAE

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

posterior = PosteriorHead(
    input_dim=32,
    latent_dim=8,
)

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

out = model(batch)

print("z:", out["posterior"]["z"].shape)
print("tab_num_recon:", out["tabular_decoder"]["tab_num_recon"].shape)
print("num cat heads:", len(out["tabular_decoder"]["tab_cat_logits"]))
print("kl:", out["losses"]["kl"])
print("balance_loss:", out["losses"]["balance_loss"])