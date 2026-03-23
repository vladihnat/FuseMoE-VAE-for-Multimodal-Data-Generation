#!/usr/bin/env python3
"""
Generation script: load a trained FuseMoE VAE checkpoint and sample synthetic
MIMIC-IV tabular + irregular time-series data.

Run from the repository root after training:
    python generate_mimic.py

Outputs (written to outputs/):
    generated_samples.pkl    — list of dicts, reloadable by MIMICTsTabDataset
    generated_tabular.csv    — one row per sample, human-readable tabular features
    generated_ts.csv         — long format: sample_id, time_step, feature, value, observed
"""

import sys
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
src  = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import numpy as np
import pandas as pd
import torch

from models.encoders.ts_irregular import TSIrregularEncoder
from models.encoders.tabular import TabularEncoder
from models.fusion.sparse_moe import FuseMoEFusion
from models.latent.posterior import PosteriorHead
from models.decoders.tabular_decoder import TabularDecoder
from models.decoders.TS_decoder import IrregularTSDecoder
from models.multimodal_vae import MultimodalVAE

# ===========================================================================
# Configuration  (must match train_mimic.py exactly)
# ===========================================================================

TS_DIM         = 24
TAB_NUM_DIM    = 1
TAB_CAT_CARDS  = [2, 9, 3, 5]

D_MODEL    = 64
MODEL_DIM  = 64
LATENT_DIM = 32

N_SAMPLES       = 500          # number of synthetic patients to generate
QUERY_STEPS     = 48           # number of TS timesteps per generated sample

CHECKPOINT_PATH = ROOT / "checkpoints" / "mimic_vae.pt"
STATS_PATH      = ROOT / "data" / "processed" / "mimic_ts_tab" / "normalization_stats.json"
OUTPUT_DIR      = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Feature names — kept here to avoid importing preprocessing scripts
# ---------------------------------------------------------------------------
TS_FEATURE_NAMES = [
    "Heart_Rate", "Systolic_BP", "Diastolic_BP", "Mean_BP",
    "Respiratory_Rate", "O2_Saturation", "GCS_Verbal", "GCS_Eye", "GCS_Motor",
    "Glucose", "Potassium", "Sodium", "Chloride", "Creatinine",
    "Urea_Nitrogen", "Bicarbonate", "Anion_Gap", "Hemoglobin",
    "Hematocrit", "Magnesium", "Platelet_Count", "Phosphate",
    "White_Blood_Cells", "Calcium_Total",
]

# Inverse categorical maps for human-readable CSV
INV_GENDER          = {0: "M", 1: "F"}
INV_ADMISSION_TYPE  = {
    0: "AMBULATORY OBSERVATION", 1: "DIRECT EMER.", 2: "DIRECT OBSERVATION",
    3: "ELECTIVE", 4: "EMERGENCY", 5: "EU OBSERVATION",
    6: "OBSERVATION ADMIT", 7: "SURGICAL SAME DAY ADMISSION", 8: "URGENT",
}
INV_INSURANCE       = {0: "Medicare", 1: "Medicaid", 2: "Other"}
INV_MARITAL_STATUS  = {0: "DIVORCED", 1: "MARRIED", 2: "SINGLE", 3: "WIDOWED", 4: "UNKNOWN"}


# ===========================================================================
# Build model (same architecture as train_mimic.py)
# ===========================================================================
def build_model(device: torch.device) -> MultimodalVAE:
    ts_encoder = TSIrregularEncoder(
        input_dim=TS_DIM,
        hidden_dim=D_MODEL,
        embed_time=32,
        num_heads=4,
        num_query_steps=48,
        dropout=0.1,
    )
    tab_encoder = TabularEncoder(
        num_numeric_features=TAB_NUM_DIM,
        categorical_cardinalities=TAB_CAT_CARDS,
        hidden_dims=(64,),
        output_dim=D_MODEL,
        dropout=0.1,
    )
    fusion = FuseMoEFusion(
        modality_dims={"ts": D_MODEL, "tab": D_MODEL},
        model_dim=MODEL_DIM,
        num_experts=4,
        expert_hidden_dim=128,
        top_k=2,
        router_type="joint",
        dropout=0.1,
    )
    posterior = PosteriorHead(
        input_dim=MODEL_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    )
    tab_decoder = TabularDecoder(
        latent_dim=LATENT_DIM,
        num_numeric_features=TAB_NUM_DIM,
        categorical_cardinalities=TAB_CAT_CARDS,
        hidden_dims=(64, 64),
        dropout=0.1,
    )
    ts_decoder = IrregularTSDecoder(
        latent_dim=LATENT_DIM,
        output_dim=TS_DIM,
        hidden_dim=128,
        embed_time=32,
        hidden_layers=(128,),
        dropout=0.1,
        decode_mask=True,
    )
    return MultimodalVAE(
        ts_encoder=ts_encoder,
        tabular_encoder=tab_encoder,
        fusion=fusion,
        posterior=posterior,
        tabular_decoder=tab_decoder,
        ts_decoder=ts_decoder,
    ).to(device)


# ===========================================================================
# Save helpers
# ===========================================================================
def save_pkl(samples: list, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    print(f"  saved {len(samples)} samples -> {path}")


def save_tabular_csv(samples: list, path: Path) -> None:
    rows = []
    for s in samples:
        rows.append({
            "sample_id":      s["id"],
            "anchor_age":     float(s["tab_num"][0]),
            "gender":         INV_GENDER.get(int(s["tab_cat"][0]), s["tab_cat"][0]),
            "admission_type": INV_ADMISSION_TYPE.get(int(s["tab_cat"][1]), s["tab_cat"][1]),
            "insurance":      INV_INSURANCE.get(int(s["tab_cat"][2]), s["tab_cat"][2]),
            "marital_status": INV_MARITAL_STATUS.get(int(s["tab_cat"][3]), s["tab_cat"][3]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved tabular CSV ({len(rows)} rows) -> {path}")


def save_ts_csv(samples: list, path: Path) -> None:
    rows = []
    for s in samples:
        ts   = s["ts_values"]      # (T, 24)
        mask = s["ts_mask"]        # (T, 24)
        times = s["ts_times"]      # (T,)
        for t_idx in range(ts.shape[0]):
            for f_idx, feat in enumerate(TS_FEATURE_NAMES):
                rows.append({
                    "sample_id": s["id"],
                    "time_step": t_idx,
                    "time_value": float(times[t_idx]),
                    "feature":   feat,
                    "value":     float(ts[t_idx, f_idx]),
                    "observed":  int(mask[t_idx, f_idx]),
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved TS CSV ({len(rows)} rows) -> {path}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run 'python train_mimic.py' first."
        )

    print(f"\nLoading checkpoint from {CHECKPOINT_PATH}...")
    model = build_model(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("  Checkpoint loaded.")

    # -----------------------------------------------------------------------
    # Load normalization stats
    # -----------------------------------------------------------------------
    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {STATS_PATH}\n"
            "Run 'python data/preprocess_data.py' first."
        )
    with open(STATS_PATH) as f:
        stats = json.load(f)
    ts_mean  = np.array(stats["ts_mean"],      dtype=np.float32)   # (24,)
    ts_std   = np.array(stats["ts_std"],       dtype=np.float32)   # (24,)
    tab_mean = np.array(stats["tab_num_mean"], dtype=np.float32)   # (1,)
    tab_std  = np.array(stats["tab_num_std"],  dtype=np.float32)   # (1,)
    print("  Normalization stats loaded.")

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------
    print(f"\nGenerating {N_SAMPLES} synthetic samples...")
    generated = model.generate(n_samples=N_SAMPLES, device=device)

    # move all tensors to CPU numpy
    tab_num     = generated["tab_num"].cpu().numpy()          # (N, 1)
    tab_cat     = generated["tab_cat"].cpu().numpy()          # (N, 4)
    ts_recon    = generated["ts_recon"].cpu().numpy()         # (N, T, 24)
    ts_mask     = generated["ts_mask"].cpu().numpy()          # (N, T, 24)
    query_times = generated["query_times"].cpu().numpy()      # (N, T)

    # -----------------------------------------------------------------------
    # Denormalize back to original clinical scale
    # -----------------------------------------------------------------------
    tab_num  = tab_num  * tab_std  + tab_mean          # (N, 1)
    ts_recon = ts_recon * ts_std   + ts_mean           # (N, T, 24) broadcast over features
    print("  Outputs denormalized to original clinical scale.")

    print(f"  tab_num  shape : {tab_num.shape}")
    print(f"  tab_cat  shape : {tab_cat.shape}")
    print(f"  ts_recon shape : {ts_recon.shape}")

    # -----------------------------------------------------------------------
    # Build sample list (same format as MIMICTsTabDataset)
    # -----------------------------------------------------------------------
    samples = []
    for i in range(N_SAMPLES):
        samples.append({
            "id":        i,
            "ts_values": ts_recon[i].astype(np.float32),
            "ts_times":  query_times[i].astype(np.float32),
            "ts_mask":   ts_mask[i].astype(np.float32),
            "tab_num":   tab_num[i].astype(np.float32),
            "tab_cat":   tab_cat[i].astype(np.int64),
        })

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nSaving outputs to {OUTPUT_DIR}/...")

    save_pkl(samples,       OUTPUT_DIR / "generated_samples.pkl")
    save_tabular_csv(samples, OUTPUT_DIR / "generated_tabular.csv")
    save_ts_csv(samples,    OUTPUT_DIR / "generated_ts.csv")

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
