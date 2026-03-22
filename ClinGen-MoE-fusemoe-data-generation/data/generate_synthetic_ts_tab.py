from pathlib import Path
import json
import math
import pickle
import numpy as np


def _one_sample(
    rng,
    sample_id,
    latent_dim,
    ts_dim,
    tab_num_dim,
    tab_cat_cardinalities,
    min_seq_len,
    max_seq_len,
    min_missing_prob,
    max_missing_prob,
):
    z = rng.normal(size=(latent_dim,)).astype(np.float32)

    seq_len = int(rng.integers(min_seq_len, max_seq_len + 1))
    times = np.sort(rng.uniform(0.0, 1.0, size=(seq_len,)).astype(np.float32))

    # Sample-specific parameters derived from the shared latent vector.
    W_amp = rng.normal(scale=0.6, size=(ts_dim, latent_dim))
    W_freq = rng.normal(scale=0.4, size=(ts_dim, latent_dim))
    W_phase = rng.normal(scale=0.5, size=(ts_dim, latent_dim))
    W_trend = rng.normal(scale=0.3, size=(ts_dim, latent_dim))
    W_bias = rng.normal(scale=0.5, size=(ts_dim, latent_dim))

    amps = 0.8 + np.exp(0.3 * (W_amp @ z))
    freqs = 1.0 + np.abs(W_freq @ z)
    phases = W_phase @ z
    trends = 0.3 * (W_trend @ z)
    biases = 0.5 * (W_bias @ z)

    values = np.zeros((seq_len, ts_dim), dtype=np.float32)
    for d in range(ts_dim):
        t = times
        seasonal = amps[d] * np.sin(2.0 * math.pi * freqs[d] * t + phases[d])
        drift = trends[d] * t
        curvature = 0.15 * np.cos(math.pi * (d + 1) * t + 0.2 * z[d % latent_dim])
        noise = rng.normal(scale=0.05, size=seq_len)
        values[:, d] = (seasonal + drift + curvature + biases[d] + noise).astype(np.float32)

    missing_prob = float(rng.uniform(min_missing_prob, max_missing_prob))
    mask = (rng.uniform(size=(seq_len, ts_dim)) > missing_prob).astype(np.float32)

    for i in range(seq_len):
        if mask[i].sum() == 0:
            mask[i, int(rng.integers(0, ts_dim))] = 1.0
    if mask.sum() == 0:
        mask[0, 0] = 1.0

    W_num = rng.normal(scale=0.8, size=(tab_num_dim, latent_dim))
    b_num = rng.normal(scale=0.2, size=(tab_num_dim,))
    tab_num = (W_num @ z + b_num + rng.normal(scale=0.08, size=(tab_num_dim,))).astype(np.float32)

    tab_cat = []
    for card in tab_cat_cardinalities:
        logits_w = rng.normal(scale=0.8, size=(card, latent_dim))
        logits_b = rng.normal(scale=0.2, size=(card,))
        logits = logits_w @ z + logits_b + rng.normal(scale=0.05, size=(card,))
        tab_cat.append(int(np.argmax(logits)))
    tab_cat = np.array(tab_cat, dtype=np.int64)

    return {
        "id": sample_id,
        "ts_values": values,
        "ts_times": times,
        "ts_mask": mask,
        "tab_num": tab_num,
        "tab_cat": tab_cat,
        "latent_z": z,
    }


def generate_dataset(
    out_dir,
    train_size=800,
    val_size=100,
    test_size=100,
    seed=42,
    latent_dim=4,
    ts_dim=5,
    tab_num_dim=6,
    tab_cat_cardinalities=(3, 4),
    min_seq_len=15,
    max_seq_len=35,
    min_missing_prob=0.10,
    max_missing_prob=0.30,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    def make_split(n, start_id):
        return [
            _one_sample(
                rng=rng,
                sample_id=start_id + i,
                latent_dim=latent_dim,
                ts_dim=ts_dim,
                tab_num_dim=tab_num_dim,
                tab_cat_cardinalities=tab_cat_cardinalities,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                min_missing_prob=min_missing_prob,
                max_missing_prob=max_missing_prob,
            )
            for i in range(n)
        ]

    train = make_split(train_size, 0)
    val = make_split(val_size, train_size)
    test = make_split(test_size, train_size + val_size)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(split, f)

    metadata = {
        "name": "synthetic_ts_tab",
        "description": "Synthetic paired irregular time-series + tabular dataset with shared latent factors.",
        "seed": seed,
        "split_sizes": {"train": train_size, "val": val_size, "test": test_size},
        "latent_dim": latent_dim,
        "ts_dim": ts_dim,
        "tab_num_dim": tab_num_dim,
        "tab_cat_cardinalities": list(tab_cat_cardinalities),
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
        "min_missing_prob": min_missing_prob,
        "max_missing_prob": max_missing_prob,
        "sample_schema": {
            "id": "int",
            "ts_values": "(seq_len, ts_dim) float32",
            "ts_times": "(seq_len,) float32",
            "ts_mask": "(seq_len, ts_dim) float32",
            "tab_num": "(tab_num_dim,) float32",
            "tab_cat": "(n_tab_cat,) int64",
            "latent_z": "(latent_dim,) float32",
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return out_dir


if __name__ == "__main__":
    out = generate_dataset("data/processed/synthetic_ts_tab")
    print(f"Generated synthetic dataset in: {out}")
