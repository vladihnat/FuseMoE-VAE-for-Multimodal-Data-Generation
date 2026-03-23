# Generation Pipeline

---

## Data download

TODO

---

## Required folder structure

All three scripts must be run from the **repository root**. The raw data must be placed at the exact path below before running the preprocessor.

```
ClinGen-MoE/
├── data/
│   ├── preprocess_data.py
│   ├── preprocess_demo_data.py
│   ├── mimic-iv-clinical-database-demo-2.2/   ← raw data goes here
│   │   ├── icu/
│   │   │   ├── icustays.csv
│   │   │   ├── chartevents.csv
│   │   │   └── d_items.csv
│   │   └── hosp/
│   │       ├── admissions.csv
│   │       ├── patients.csv
│   │       ├── labevents.csv
│   │       └── d_labitems.csv
│   └── processed/
│       └── mimic_ts_tab/                      ← created by preprocess_data.py
│           ├── train.pkl
│           ├── val.pkl
│           ├── test.pkl
│           └── normalization_stats.json
├── checkpoints/
│   └── mimic_vae.pt                           ← created by train_mimic.py
├── outputs/
│   ├── generated_samples.pkl                  ← created by generate_mimic.py
│   ├── generated_tabular.csv
│   └── generated_ts.csv
├── train_mimic.py
└── generate_mimic.py
```

---

## `data/preprocess_data.py`

Reads raw MIMIC-IV CSVs and produces the `.pkl` splits used for training.

### Input — CSV files used

| File | Used for |
|---|---|
| `icu/icustays.csv` | stay_id, admission time, LOS (filter: LOS ≥ 2 days) |
| `icu/chartevents.csv` | vital sign measurements |
| `icu/d_items.csv` | mapping item_id → vital label |
| `hosp/labevents.csv` | lab measurements |
| `hosp/d_labitems.csv` | mapping item_id → lab label |
| `hosp/admissions.csv` | IHM label + tabular features (admission_type, insurance, marital_status) |
| `hosp/patients.csv` | tabular features (age, gender) |

### How multiple measurements are merged into one matrix

For each ICU stay, all vitals and labs recorded in the **first 48 hours** are collected.
Every distinct timestamp across both sources becomes one row. The result is a sparse
`(T, 24)` matrix where:

- **T** = number of unique timestamps for that stay (varies per patient — this is what makes it irregular)
- **24 columns** = 9 vitals + 15 labs, always in the same fixed order
- A value is written at `[time_idx, feature_idx]` only if that feature was measured at that time
- The mask `(T, 24)` records which cells were actually observed (1) vs left as zero (0)

If two measurements of the same feature occur at the exact same timestamp, the last one written is kept (no aggregation).

### Output

`data/processed/mimic_ts_tab/{train, val, test}.pkl` — stratified 60/20/20 split.

Each record:

| Key | Shape | Description |
|---|---|---|
| `id` | scalar | ICU stay_id |
| `ts_values` | `(T, 24)` | z-scored vital/lab values (observed positions only) |
| `ts_times` | `(T,)` | timestamps in hours since ICU admission |
| `ts_mask` | `(T, 24)` | binary observation mask |
| `tab_num` | `(1,)` | z-scored `[anchor_age]` |
| `tab_cat` | `(4,)` | `[gender, admission_type, insurance, marital_status]` encoded as integers |
| `label` | scalar | in-hospital mortality (0/1) |

`data/processed/mimic_ts_tab/normalization_stats.json` — per-feature mean and std computed from the training set, used to denormalize generated outputs.

### Run

```bash
python data/preprocess_data.py
```

---

## `train_mimic.py`

Trains the multimodal VAE on the preprocessed MIMIC data.

### Input

`data/processed/mimic_ts_tab/{train, val, test}.pkl`

### Output

`checkpoints/mimic_vae.pt` — saved model weights after the final epoch.

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `D_MODEL` | 64 | encoder embedding dimension |
| `LATENT_DIM` | 32 | VAE latent space dimension |
| `NUM_EXPERTS` | 4 | number of MoE experts |
| `NUM_EPOCHS` | 150 | number of training epochs |
| `BATCH_SIZE` | 16 | samples per batch |
| `LR` | 1e-3 | Adam learning rate |
| `BETA` | 1.0 | final KL weight (reached after warmup) |
| `KL_WARMUP_EPOCHS` | 40 | epochs over which beta linearly increases 0 → BETA |
| `LAMBDA_BALANCE` | 1e-2 | weight of the MoE load-balancing term |

### Run

```bash
python train_mimic.py
```

---

## `generate_mimic.py`

Loads the trained checkpoint and generates synthetic ICU stay data.

### Input

- `checkpoints/mimic_vae.pt`
- `data/processed/mimic_ts_tab/normalization_stats.json`

### Output

| File | Format | Content |
|---|---|---|
| `outputs/generated_samples.pkl` | list of dicts | same format as the preprocessed data — can be reloaded with `MIMICTsTabDataset` |
| `outputs/generated_tabular.csv` | one row per sample | `sample_id, anchor_age, gender, admission_type, insurance, marital_status` |
| `outputs/generated_ts.csv` | long format | `sample_id, time_step, time_value, feature_name, value, observed` |

Generated values are denormalized back to the original clinical scale using the saved normalization stats.

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `N_SAMPLES` | 500 | number of synthetic samples to generate |
| `QUERY_STEPS` | 48 | number of time steps per generated TS sample |

### Run

```bash
python generate_mimic.py
```

---

## Full sequence

```bash
python data/preprocess_data.py
python train_mimic.py
python generate_mimic.py
```
