#!/usr/bin/env python3
"""
Training entry point for the FuseMoE VAE pipeline on MIMIC-IV demo data.

Run from the repository root:
    python train_mimic.py

Requires preprocessed data at:
    data/processed/mimic_ts_tab/{train,val,test}.pkl
    -> produced by: python data/preprocess_data.py
"""

import sys
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
import torch
import torch.optim as optim

from data.mimic_dataset import make_mimic_ts_tab_dataloader
from models.encoders.ts_irregular import TSIrregularEncoder
from models.encoders.tabular import TabularEncoder
from models.fusion.sparse_moe import FuseMoEFusion
from models.latent.posterior import PosteriorHead
from models.decoders.tabular_decoder import TabularDecoder
from models.decoders.TS_decoder import IrregularTSDecoder
from models.multimodal_vae import MultimodalVAE
from training.engine import train_one_epoch, evaluate_step

# ===========================================================================
# Hyperparameters
# ===========================================================================

# --- data (must match preprocess_data.py) ---
TS_DIM         = 24          # 9 vitals + 15 labs
TAB_NUM_DIM    = 1           # anchor_age
TAB_CAT_CARDS  = [2, 9, 3, 5]  # gender, admission_type, insurance, marital_status

# --- architecture ---
D_MODEL        = 64          # shared encoder output dim
MODEL_DIM      = 64          # fusion model dim  (= PosteriorHead input_dim)
LATENT_DIM     = 32

# --- training ---
BATCH_SIZE     = 16
NUM_EPOCHS     = 30
LR             = 1e-3
BETA           = 1.0         # KL weight
LAMBDA_BALANCE = 1e-2        # MoE load-balancing weight
GRAD_CLIP      = 1.0

DATA_ROOT = ROOT / "data" / "processed" / "mimic_ts_tab"

# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("\nLoading data...")
    train_loader = make_mimic_ts_tab_dataloader(
        DATA_ROOT / "train.pkl",
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = make_mimic_ts_tab_dataloader(
        DATA_ROOT / "val.pkl",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    print(f"  train batches : {len(train_loader)}")
    print(f"  val   batches : {len(val_loader)}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    print("\nBuilding model...")

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

    model = MultimodalVAE(
        ts_encoder=ts_encoder,
        tabular_encoder=tab_encoder,
        fusion=fusion,
        posterior=posterior,
        tabular_decoder=tab_decoder,
        ts_decoder=ts_decoder,
        kl_reduction="mean",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    print(f"{'epoch':>5}  {'train_total':>11}  {'train_recon':>11}  {'train_kl':>9}  "
          f"{'val_total':>9}  {'val_recon':>9}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=BETA,
            lambda_balance=LAMBDA_BALANCE,
            grad_clip_norm=GRAD_CLIP,
        )

        # val: average over all val batches
        val_totals = {}
        for batch in val_loader:
            m = evaluate_step(
                model=model,
                batch=batch,
                device=device,
                beta=BETA,
                lambda_balance=LAMBDA_BALANCE,
            )
            for k, v in m.items():
                val_totals[k] = val_totals.get(k, 0.0) + v
        val_metrics = {k: v / len(val_loader) for k, v in val_totals.items()}

        print(
            f"{epoch:>5}  "
            f"{train_metrics['total']:>11.4f}  "
            f"{train_metrics['reconstruction']:>11.4f}  "
            f"{train_metrics['kl']:>9.4f}  "
            f"{val_metrics['total']:>9.4f}  "
            f"{val_metrics['reconstruction']:>9.4f}"
        )

    # -----------------------------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------------------------
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "mimic_vae.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
