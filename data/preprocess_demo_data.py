#!/usr/bin/env python3
"""
Preprocess MIMIC-IV Clinical Database Demo (v2.2) into FuseMoE's expected .pkl format.

This script:
1. Reads raw CSVs (icustays, admissions, patients, chartevents, labevents, d_items)
2. Extracts vital signs and lab values for the first 48 hours of each ICU stay
3. Creates irregular time-series matrices (irg_ts, irg_ts_mask, ts_tt)
4. Sets missing modality flags for text, CXR, and ECG (not available in demo)
5. Determines in-hospital mortality (IHM) labels
6. Splits data into train/val/test and saves .pkl files

Usage:
    conda activate fusemoe_research
    python preprocess_demo_data.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ===========================================================================
# Configuration
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(BASE_DIR, "mimic-iv-clinical-database-demo-2.2")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "ihm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# First 48 hours of ICU stay
OBSERVATION_WINDOW_HRS = 48
# Minimum LOS in days to qualify for IHM-48 task
MIN_LOS_DAYS = 2.0

# Output filename pattern (matches what load_data expects)
# Format: {mode}_ihm-48-cxr-notes-ecg_stays.pkl
TASK_SUFFIX = "ihm-48-cxr-notes-ecg"

# ---------------------------------------------------------------------------
# Vital signs from chartevents (labels → d_items.csv)
# Based on the project's ts_irregular.ipynb reference
# ---------------------------------------------------------------------------
VITAL_LABELS = [
    'Heart Rate',
    'Non Invasive Blood Pressure systolic',
    'Non Invasive Blood Pressure diastolic',
    'Non Invasive Blood Pressure mean',
    'Respiratory Rate',
    'O2 saturation pulseoxymetry',
    'GCS - Verbal Response',
    'GCS - Eye Opening',
    'GCS - Motor Response',
]

# Rename to shorter feature names
VITAL_RENAME = {
    'Non Invasive Blood Pressure systolic': 'Systolic_BP',
    'Non Invasive Blood Pressure diastolic': 'Diastolic_BP',
    'Non Invasive Blood Pressure mean': 'Mean_BP',
    'O2 saturation pulseoxymetry': 'O2_Saturation',
    'Heart Rate': 'Heart_Rate',
    'Respiratory Rate': 'Respiratory_Rate',
    'GCS - Verbal Response': 'GCS_Verbal',
    'GCS - Eye Opening': 'GCS_Eye',
    'GCS - Motor Response': 'GCS_Motor',
}

# ---------------------------------------------------------------------------
# Lab items from labevents (labels → d_labitems.csv)
# These are looked up by label from chartevents in the original notebook,
# but the demo stores labs in hosp/labevents with different item IDs.
# We'll use d_labitems.csv item IDs matched by label.
# ---------------------------------------------------------------------------
LAB_LABELS = [
    'Glucose',
    'Potassium',
    'Sodium',
    'Chloride',
    'Creatinine',
    'Urea Nitrogen',
    'Bicarbonate',
    'Anion Gap',
    'Hemoglobin',
    'Hematocrit',
    'Magnesium',
    'Platelet Count',
    'Phosphate',
    'White Blood Cells',
    'Calcium, Total',
]

# All feature names (vitals + labs), in stable order
ALL_FEATURE_NAMES = (
    [VITAL_RENAME.get(v, v) for v in VITAL_LABELS] +
    [l.replace(', ', '_').replace(' ', '_') for l in LAB_LABELS]
)
NUM_FEATURES = len(ALL_FEATURE_NAMES)

print(f"Total features: {NUM_FEATURES}")
print(f"Features: {ALL_FEATURE_NAMES}")


# ===========================================================================
# Step 1: Load raw CSVs
# ===========================================================================
def load_raw_data():
    print("\n=== Loading raw CSVs ===")
    
    # ICU stays
    icustays = pd.read_csv(os.path.join(DEMO_DIR, "icu", "icustays.csv"))
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])
    print(f"ICU stays: {len(icustays)}")
    
    # Admissions (for hospital_expire_flag = IHM label)
    admissions = pd.read_csv(os.path.join(DEMO_DIR, "hosp", "admissions.csv"))
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    admissions['deathtime'] = pd.to_datetime(admissions['deathtime'])
    print(f"Admissions: {len(admissions)}")
    
    # Patients
    patients = pd.read_csv(os.path.join(DEMO_DIR, "hosp", "patients.csv"))
    print(f"Patients: {len(patients)}")
    
    # d_items (for mapping vital itemid → label)
    d_items = pd.read_csv(os.path.join(DEMO_DIR, "icu", "d_items.csv"))
    
    # d_labitems (for mapping lab itemid → label)
    d_labitems = pd.read_csv(os.path.join(DEMO_DIR, "hosp", "d_labitems.csv"))
    
    # Chartevents (vitals)
    chartevents = pd.read_csv(os.path.join(DEMO_DIR, "icu", "chartevents.csv"),
                               low_memory=False)
    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
    print(f"Chart events: {len(chartevents)}")
    
    # Labevents
    labevents = pd.read_csv(os.path.join(DEMO_DIR, "hosp", "labevents.csv"),
                             low_memory=False)
    labevents['charttime'] = pd.to_datetime(labevents['charttime'])
    labevents = labevents.dropna(subset=['hadm_id'])
    print(f"Lab events (with hadm_id): {len(labevents)}")
    
    return icustays, admissions, patients, d_items, d_labitems, chartevents, labevents


# ===========================================================================
# Step 2: Build item ID lookup maps
# ===========================================================================
def build_item_maps(d_items, d_labitems):
    print("\n=== Building item ID maps ===")
    
    # Vitals: label -> itemid from d_items
    vital_itemids = {}
    for label in VITAL_LABELS:
        matches = d_items[d_items['label'] == label]
        if len(matches) > 0:
            vital_itemids[matches.iloc[0]['itemid']] = VITAL_RENAME.get(label, label)
            print(f"  Vital: {label} -> itemid {matches.iloc[0]['itemid']}")
        else:
            print(f"  WARNING: Vital '{label}' not found in d_items!")
    
    # Labs: label -> itemid from d_labitems (blood only)
    lab_itemids = {}
    for label in LAB_LABELS:
        # Match label exactly, prefer Blood fluid
        matches = d_labitems[d_labitems['label'] == label]
        blood_matches = matches[matches['fluid'] == 'Blood']
        if len(blood_matches) > 0:
            item_id = blood_matches.iloc[0]['itemid']
        elif len(matches) > 0:
            item_id = matches.iloc[0]['itemid']
        else:
            print(f"  WARNING: Lab '{label}' not found in d_labitems!")
            continue
        feature_name = label.replace(', ', '_').replace(' ', '_')
        lab_itemids[item_id] = feature_name
        print(f"  Lab: {label} -> itemid {item_id}")
    
    return vital_itemids, lab_itemids


# ===========================================================================
# Step 3: Extract time series per ICU stay
# ===========================================================================
def extract_stay_timeseries(stay, vital_itemids, lab_itemids,
                            chartevents, labevents):
    """
    For a single ICU stay, extract vital and lab observations within the
    first OBSERVATION_WINDOW_HRS hours.
    
    Returns: (irg_ts, irg_ts_mask, ts_tt) or None if no data
    """
    stay_id = stay['stay_id']
    subject_id = stay['subject_id']
    hadm_id = stay['hadm_id']
    icu_intime = stay['intime']
    window_end = icu_intime + pd.Timedelta(hours=OBSERVATION_WINDOW_HRS)
    
    # ---- Vitals from chartevents ----
    stay_charts = chartevents[
        (chartevents['stay_id'] == stay_id) &
        (chartevents['charttime'] >= icu_intime) &
        (chartevents['charttime'] <= window_end) &
        (chartevents['itemid'].isin(vital_itemids.keys())) &
        (chartevents['valuenum'].notna())
    ].copy()
    
    if len(stay_charts) > 0:
        stay_charts['feature'] = stay_charts['itemid'].map(vital_itemids)
        stay_charts['time_hrs'] = (stay_charts['charttime'] - icu_intime).dt.total_seconds() / 3600
        stay_charts = stay_charts[['time_hrs', 'feature', 'valuenum']].copy()
    
    # ---- Labs from labevents ----
    stay_labs = labevents[
        (labevents['subject_id'] == subject_id) &
        (labevents['hadm_id'] == hadm_id) &
        (labevents['charttime'] >= icu_intime) &
        (labevents['charttime'] <= window_end) &
        (labevents['itemid'].isin(lab_itemids.keys())) &
        (labevents['valuenum'].notna())
    ].copy()
    
    if len(stay_labs) > 0:
        stay_labs['feature'] = stay_labs['itemid'].map(lab_itemids)
        stay_labs['time_hrs'] = (stay_labs['charttime'] - icu_intime).dt.total_seconds() / 3600
        stay_labs = stay_labs[['time_hrs', 'feature', 'valuenum']].copy()
    
    # ---- Combine ----
    combined = pd.concat([stay_charts, stay_labs], ignore_index=True)
    
    if len(combined) == 0:
        return None
    
    # Get unique timestamps, sorted
    unique_times = sorted(combined['time_hrs'].unique())
    n_timesteps = len(unique_times)
    
    # Build matrices
    irg_ts = np.zeros((n_timesteps, NUM_FEATURES), dtype=np.float32)
    irg_ts_mask = np.zeros((n_timesteps, NUM_FEATURES), dtype=np.float32)
    ts_tt = np.array(unique_times, dtype=np.float32)
    
    # Feature name → column index
    feat_to_idx = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
    
    for _, row in combined.iterrows():
        t_idx = unique_times.index(row['time_hrs'])
        f_idx = feat_to_idx.get(row['feature'])
        if f_idx is not None:
            irg_ts[t_idx, f_idx] = row['valuenum']
            irg_ts_mask[t_idx, f_idx] = 1.0
    
    return irg_ts, irg_ts_mask, ts_tt


# ===========================================================================
# Step 4: Build patient records
# ===========================================================================
def build_records(icustays, admissions, vital_itemids, lab_itemids,
                  chartevents, labevents):
    print(f"\n=== Building patient records (LOS >= {MIN_LOS_DAYS} days) ===")
    
    # Filter ICU stays by minimum LOS
    qualifying = icustays[icustays['los'] >= MIN_LOS_DAYS].copy()
    print(f"Qualifying stays (LOS >= {MIN_LOS_DAYS}d): {len(qualifying)}")
    
    # Merge with admissions to get IHM label
    qualifying = qualifying.merge(
        admissions[['hadm_id', 'hospital_expire_flag']],
        on='hadm_id',
        how='left'
    )
    
    records = []
    skipped = 0
    
    for idx, stay in qualifying.iterrows():
        result = extract_stay_timeseries(
            stay, vital_itemids, lab_itemids, chartevents, labevents
        )
        
        if result is None:
            skipped += 1
            continue
        
        irg_ts, irg_ts_mask, ts_tt = result
        
        # Build record dict matching TSNote_Irg expectations
        record = {
            'name': int(stay['stay_id']),
            
            # Time series
            'irg_ts': irg_ts,
            'irg_ts_mask': irg_ts_mask,
            'ts_tt': ts_tt,
            'reg_ts': np.zeros((OBSERVATION_WINDOW_HRS, NUM_FEATURES), dtype=np.float32),
            
            # Text (missing)
            'text_data': [],
            'text_embeddings': np.zeros((0, 768), dtype=np.float32),
            'text_time_to_end': [],
            'text_missing': 1,
            
            # CXR (missing)
            'cxr_feats': np.zeros((0, 1024), dtype=np.float32),
            'cxr_time': np.array([], dtype=np.float32),
            'cxr_missing': 1,
            
            # ECG (missing)
            'ecg_feats': np.zeros((0, 256), dtype=np.float32),
            'ecg_time': np.array([], dtype=np.float32),
            'ecg_missing': 1,
            
            # Label
            'label': int(stay['hospital_expire_flag']),
        }
        
        records.append(record)
    
    print(f"Records built: {len(records)}, skipped (no data): {skipped}")
    
    # Label distribution
    labels = [r['label'] for r in records]
    print(f"Label distribution: {sum(labels)} died / {len(labels) - sum(labels)} survived")
    
    return records


# ===========================================================================
# Step 5: Split and save
# ===========================================================================
def split_and_save(records):
    print(f"\n=== Splitting and saving to {OUTPUT_DIR} ===")
    
    labels = [r['label'] for r in records]
    
    # Stratified split: 60% train, 20% val, 20% test
    # If too few positives, fall back to random split
    n_positive = sum(labels)
    
    if n_positive >= 3 and len(records) >= 10:
        # Enough for stratified split
        train_data, temp_data = train_test_split(
            records, test_size=0.4, random_state=42,
            stratify=labels
        )
        temp_labels = [r['label'] for r in temp_data]
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42,
            stratify=temp_labels
        )
    else:
        # Random split
        print("  NOTE: Too few positive samples for stratified split, using random split")
        train_data, temp_data = train_test_split(
            records, test_size=0.4, random_state=42
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42
        )
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
    }
    
    for split_name, split_data in splits.items():
        filename = f"{split_name}_{TASK_SUFFIX}_stays.pkl"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(split_data, f)
        
        split_labels = [r['label'] for r in split_data]
        print(f"  {split_name}: {len(split_data)} records "
              f"({sum(split_labels)} pos / {len(split_labels) - sum(split_labels)} neg) "
              f"-> {filepath}")
    
    return splits


# ===========================================================================
# Step 6: Print summary statistics
# ===========================================================================
def print_summary(splits):
    print("\n=== Data Summary ===")
    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            print(f"  {split_name}: EMPTY")
            continue
        
        ts_shapes = [r['irg_ts'].shape for r in split_data]
        n_timesteps = [s[0] for s in ts_shapes]
        
        print(f"  {split_name}:")
        print(f"    Samples: {len(split_data)}")
        print(f"    Features per timestep: {ts_shapes[0][1]}")
        print(f"    Timesteps: min={min(n_timesteps)}, max={max(n_timesteps)}, "
              f"mean={np.mean(n_timesteps):.1f}")
        
        # Check mask fill rate
        fill_rates = [r['irg_ts_mask'].mean() for r in split_data]
        print(f"    Mask fill rate: {np.mean(fill_rates):.3f}")
        
        sample = split_data[0]
        print(f"    Sample keys: {sorted(sample.keys())}")
        print(f"    Sample irg_ts shape: {sample['irg_ts'].shape}")
        print(f"    Sample ts_tt length: {len(sample['ts_tt'])}")
        print(f"    Sample label: {sample['label']}")
        print(f"    Sample text_missing: {sample['text_missing']}")
        print(f"    Sample cxr_missing: {sample['cxr_missing']}")
        print(f"    Sample ecg_missing: {sample['ecg_missing']}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("MIMIC-IV Demo Data Preprocessor for FuseMoE")
    print("=" * 60)
    
    # Load data
    icustays, admissions, patients, d_items, d_labitems, chartevents, labevents = load_raw_data()
    
    # Build item maps
    vital_itemids, lab_itemids = build_item_maps(d_items, d_labitems)
    
    # Build records
    records = build_records(icustays, admissions, vital_itemids, lab_itemids,
                           chartevents, labevents)
    
    if len(records) == 0:
        print("\nERROR: No records produced. Check data paths and filters.")
        return
    
    # Split and save
    splits = split_and_save(records)
    
    # Print summary
    print_summary(splits)
    
    print("\n✅ Preprocessing complete!")
    print(f"   Files saved to: {OUTPUT_DIR}")
    print(f"   Feature names: {ALL_FEATURE_NAMES}")


if __name__ == "__main__":
    main()
