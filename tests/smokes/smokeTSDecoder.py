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
from models.encoders.ts_irregular import TSIrregularEncoder
from models.encoders.tabular import TabularEncoder
from models.fusion.sparse_moe import FuseMoEFusion
from models.latent.posterior import PosteriorHead
from models.decoders.timeseries_decoder import TSIrregularDecoder

print("Loading synthetic data...")
loader = make_synthetic_ts_tab_dataloader(
    "data/processed/synthetic_ts_tab/train.pkl",
    batch_size=8,
    shuffle=False,
)
batch = next(iter(loader))

print("Initializing modules...")
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

ts_decoder = TSIrregularDecoder(
    latent_dim=8,
    output_dim=batch["ts_values"].shape[-1],
    embed_time=16,
    hidden_dims=(32, 32),
    default_seq_len=batch["ts_values"].shape[1],
)

print("Running forward pass up to posterior...")
ts_out = ts_encoder(
    values=batch["ts_values"],
    mask=batch["ts_mask"],
    times=batch["ts_times"],
)

# Use positional arguments for TabularEncoder
tab_out = tab_encoder(batch["tab_num"], batch["tab_cat"])

# Extract the tensor from the tabular encoder's output dictionary
tab_tensor = tab_out["embedding"] if "embedding" in tab_out else tab_out["pooled"]

fused = fusion({"ts": ts_out["pooled"], "tab": tab_tensor})

# FIX: We now know the fusion module outputs the final representation under 'pooled'
fused_tensor = fused["pooled"]

post_out = posterior(fused_tensor)
z = post_out["z"]

print("Running TS Decoder...")
# Test 1: Let the decoder use its default internal target times
recon_default = ts_decoder(z=z)

# Test 2: Condition the decoder on the exact timesteps from the batch
recon_custom = ts_decoder(z=z, target_times=batch["ts_times"])

print("\n--- Output Shapes ---")
print("z:                 ", z.shape)
print("recon_default:     ", recon_default["ts_recon"].shape)
print("recon_custom:      ", recon_custom["ts_recon"].shape)
print("recon_target_times:", recon_custom["target_times"].shape)
print("Smoke test passed successfully!")