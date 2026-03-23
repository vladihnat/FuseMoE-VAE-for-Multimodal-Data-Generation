#!/usr/bin/env python3
"""
Preprocess MIMIC-IV Clinical Database Demo (v2.2) into the FuseMoE VAE pipeline format.

Extends preprocess_demo_data.py with:
1. Field names aligned to MIMICTsTabDataset (id, ts_values, ts_times, ts_mask, tab_num, tab_cat)
2. Tabular features extracted from patient demographics and admission metadata:
   - Numeric:      anchor_age
   - Categorical:  gender, admission_type, insurance, marital_status

Usage:
    python preprocess_data.py
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess_demo_data import (
    load_raw_data,
    build_item_maps,
    extract_stay_timeseries,
    ALL_FEATURE_NAMES,
    MIN_LOS_DAYS,
    NUM_FEATURES,
)

# ===========================================================================
# Configuration
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "mimic_ts_tab")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Tabular feature definitions
# ---------------------------------------------------------------------------

# Numeric features extracted from patients and admissions (in order)
TAB_NUMERIC_FEATURES = ["anchor_age"]
TAB_NUM_DIM = len(TAB_NUMERIC_FEATURES)

# Fixed categorical encoding maps (defined globally for consistency across splits)
GENDER_MAP = {
    "M": 0,
    "F": 1,
}

ADMISSION_TYPE_MAP = {
    "AMBULATORY OBSERVATION": 0,
    "DIRECT EMER.": 1,
    "DIRECT OBSERVATION": 2,
    "ELECTIVE": 3,
    "EMERGENCY": 4,
    "EU OBSERVATION": 5,
    "OBSERVATION ADMIT": 6,
    "SURGICAL SAME DAY ADMISSION": 7,
    "URGENT": 8,
}

INSURANCE_MAP = {
    "Medicare": 0,
    "Medicaid": 1,
    "Other": 2,
}

MARITAL_STATUS_MAP = {
    "DIVORCED": 0,
    "MARRIED": 1,
    "SINGLE": 2,
    "WIDOWED": 3,
    "UNKNOWN": 4,
}

# Cardinalities in the same column order as tab_cat
TAB_CAT_CARDINALITIES = [
    len(GENDER_MAP),            # 2
    len(ADMISSION_TYPE_MAP),    # 9
    len(INSURANCE_MAP),         # 3
    len(MARITAL_STATUS_MAP),    # 5
]

print(f"Tabular numeric features ({TAB_NUM_DIM}): {TAB_NUMERIC_FEATURES}")
print(f"Tabular categorical cardinalities: {TAB_CAT_CARDINALITIES}")


# ===========================================================================
# Step 1: Extract tabular features per stay
# ===========================================================================
def extract_tabular_features(stay, patients, admissions):
    """Extract numeric and categorical tabular features for one ICU stay.

    Args:
        stay:       row from the qualifying icustays DataFrame
        patients:   full patients DataFrame
        admissions: full admissions DataFrame

    Returns:
        tab_num: np.ndarray of shape (TAB_NUM_DIM,)  float32
        tab_cat: np.ndarray of shape (4,)             int64
    """
    subject_id = stay["subject_id"]
    hadm_id = stay["hadm_id"]

    # --- patient demographics ---
    pat_row = patients[patients["subject_id"] == subject_id]
    age = float(pat_row["anchor_age"].values[0]) if len(pat_row) > 0 else 0.0
    gender_raw = pat_row["gender"].values[0] if len(pat_row) > 0 else "M"

    tab_num = np.array([age], dtype=np.float32)

    # --- admission metadata ---
    adm_row = admissions[admissions["hadm_id"] == hadm_id]
    if len(adm_row) > 0:
        adm = adm_row.iloc[0]
        admission_type = adm["admission_type"] if pd.notna(adm.get("admission_type")) else "EMERGENCY"
        insurance = adm["insurance"] if pd.notna(adm.get("insurance")) else "Other"
        marital_raw = adm["marital_status"] if pd.notna(adm.get("marital_status")) else "UNKNOWN"
        marital_status = marital_raw.upper() if isinstance(marital_raw, str) else "UNKNOWN"
    else:
        admission_type = "EMERGENCY"
        insurance = "Other"
        marital_status = "UNKNOWN"

    # encode categoricals; unknown values fall back to the last index
    gender_idx = GENDER_MAP.get(gender_raw, 0)
    adm_type_idx = ADMISSION_TYPE_MAP.get(admission_type, len(ADMISSION_TYPE_MAP) - 1)
    insurance_idx = INSURANCE_MAP.get(insurance, len(INSURANCE_MAP) - 1)
    marital_idx = MARITAL_STATUS_MAP.get(marital_status, len(MARITAL_STATUS_MAP) - 1)

    tab_cat = np.array([gender_idx, adm_type_idx, insurance_idx, marital_idx], dtype=np.int64)

    return tab_num, tab_cat


# ===========================================================================
# Step 2: Build patient records
# ===========================================================================
def build_records(icustays, admissions, patients, vital_itemids, lab_itemids,
                  chartevents, labevents):
    print(f"\n=== Building patient records (LOS >= {MIN_LOS_DAYS} days) ===")

    qualifying = icustays[icustays["los"] >= MIN_LOS_DAYS].copy()
    print(f"Qualifying stays (LOS >= {MIN_LOS_DAYS}d): {len(qualifying)}")

    qualifying = qualifying.merge(
        admissions[["hadm_id", "hospital_expire_flag"]],
        on="hadm_id",
        how="left",
    )

    records = []
    skipped = 0

    for _, stay in qualifying.iterrows():
        ts_result = extract_stay_timeseries(
            stay, vital_itemids, lab_itemids, chartevents, labevents
        )

        if ts_result is None:
            skipped += 1
            continue

        irg_ts, irg_ts_mask, ts_tt = ts_result

        tab_num, tab_cat = extract_tabular_features(stay, patients, admissions)

        record = {
            "id":        int(stay["stay_id"]),
            "ts_values": irg_ts,
            "ts_times":  ts_tt,
            "ts_mask":   irg_ts_mask,
            "tab_num":   tab_num,
            "tab_cat":   tab_cat,
            "label":     int(stay["hospital_expire_flag"]),
        }
        records.append(record)

    print(f"Records built: {len(records)}, skipped (no TS data): {skipped}")

    labels = [r["label"] for r in records]
    print(f"Label distribution: {sum(labels)} died / {len(labels) - sum(labels)} survived")

    return records


# ===========================================================================
# Step 3: Normalization
# ===========================================================================
def compute_normalization_stats(train_records):
    """Compute per-feature mean and std from observed values in the training set only."""
    print("\n=== Computing normalization statistics from training set ===")

    ts_mean = np.zeros(NUM_FEATURES, dtype=np.float32)
    ts_std  = np.ones(NUM_FEATURES,  dtype=np.float32)

    for f in range(NUM_FEATURES):
        observed = []
        for r in train_records:
            mask_f = r["ts_mask"][:, f]
            vals_f = r["ts_values"][:, f]
            observed.extend(vals_f[mask_f == 1.0].tolist())
        if len(observed) > 1:
            ts_mean[f] = float(np.mean(observed))
            ts_std[f]  = float(max(np.std(observed), 1e-6))
        print(f"  {ALL_FEATURE_NAMES[f]:>35s}: mean={ts_mean[f]:8.3f}  std={ts_std[f]:8.3f}  "
              f"n_obs={len(observed)}")

    ages = [float(r["tab_num"][0]) for r in train_records]
    tab_num_mean = [float(np.mean(ages))]
    tab_num_std  = [float(max(np.std(ages), 1e-6))]
    print(f"  {'anchor_age':>35s}: mean={tab_num_mean[0]:8.3f}  std={tab_num_std[0]:8.3f}")

    return {
        "ts_mean":      ts_mean.tolist(),
        "ts_std":       ts_std.tolist(),
        "tab_num_mean": tab_num_mean,
        "tab_num_std":  tab_num_std,
    }


def apply_normalization(records, stats):
    """Apply z-score normalization in place. Unobserved TS positions remain zero."""
    ts_mean  = np.array(stats["ts_mean"],      dtype=np.float32)
    ts_std   = np.array(stats["ts_std"],       dtype=np.float32)
    tab_mean = np.array(stats["tab_num_mean"], dtype=np.float32)
    tab_std  = np.array(stats["tab_num_std"],  dtype=np.float32)

    for r in records:
        # normalize only observed positions; unobserved stay at 0
        r["ts_values"] = (r["ts_values"] - ts_mean) / ts_std * r["ts_mask"]
        r["tab_num"]   = (r["tab_num"] - tab_mean) / tab_std

    return records


# ===========================================================================
# Step 4: Split and save
# ===========================================================================
def split_and_save(records):
    print(f"\n=== Splitting and saving to {OUTPUT_DIR} ===")

    labels = [r["label"] for r in records]
    n_positive = sum(labels)

    if n_positive >= 3 and len(records) >= 10:
        train_data, temp_data = train_test_split(
            records, test_size=0.4, random_state=42, stratify=labels
        )
        temp_labels = [r["label"] for r in temp_data]
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42, stratify=temp_labels
        )
    else:
        print("  NOTE: Too few positive samples for stratified split, using random split")
        train_data, temp_data = train_test_split(records, test_size=0.4, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # --- normalization: fit on train, apply to all splits ---
    stats = compute_normalization_stats(train_data)
    for split_data in [train_data, val_data, test_data]:
        apply_normalization(split_data, stats)

    # --- save normalization stats ---
    stats_path = os.path.join(OUTPUT_DIR, "normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Normalization stats saved -> {stats_path}")

    splits = {"train": train_data, "val": val_data, "test": test_data}

    for split_name, split_data in splits.items():
        filepath = os.path.join(OUTPUT_DIR, f"{split_name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(split_data, f)

        split_labels = [r["label"] for r in split_data]
        print(
            f"  {split_name}: {len(split_data)} records "
            f"({sum(split_labels)} pos / {len(split_labels) - sum(split_labels)} neg) "
            f"-> {filepath}"
        )

    return splits


# ===========================================================================
# Step 4: Print summary statistics
# ===========================================================================
def print_summary(splits):
    print("\n=== Data Summary ===")
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            print(f"  {split_name}: EMPTY")
            continue

        n_timesteps = [r["ts_values"].shape[0] for r in split_data]
        fill_rates = [r["ts_mask"].mean() for r in split_data]
        sample = split_data[0]

        print(f"  {split_name}:")
        print(f"    Samples:         {len(split_data)}")
        print(f"    TS features:     {sample['ts_values'].shape[1]}")
        print(f"    Timesteps:       min={min(n_timesteps)}, max={max(n_timesteps)}, "
              f"mean={np.mean(n_timesteps):.1f}")
        print(f"    Mask fill rate:  {np.mean(fill_rates):.3f}")
        print(f"    tab_num shape:   {sample['tab_num'].shape}  features: {TAB_NUMERIC_FEATURES}")
        print(f"    tab_cat shape:   {sample['tab_cat'].shape}  cardinalities: {TAB_CAT_CARDINALITIES}")
        print(f"    Sample keys:     {sorted(sample.keys())}")
        print(f"    Sample label:    {sample['label']}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("MIMIC-IV Demo Preprocessor — FuseMoE VAE format")
    print("=" * 60)

    icustays, admissions, patients, d_items, d_labitems, chartevents, labevents = load_raw_data()

    vital_itemids, lab_itemids = build_item_maps(d_items, d_labitems)

    records = build_records(
        icustays, admissions, patients, vital_itemids, lab_itemids, chartevents, labevents
    )

    if len(records) == 0:
        print("\nERROR: No records produced. Check data paths and filters.")
        return

    splits = split_and_save(records)

    print_summary(splits)

    print("\n✅ Preprocessing complete!")
    print(f"   Files saved to: {OUTPUT_DIR}")
    print(f"   Numeric features:         {TAB_NUMERIC_FEATURES}")
    print(f"   Categorical cardinalities: {TAB_CAT_CARDINALITIES}")


if __name__ == "__main__":
    main()
