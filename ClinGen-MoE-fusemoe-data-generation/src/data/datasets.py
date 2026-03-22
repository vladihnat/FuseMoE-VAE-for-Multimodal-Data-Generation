
from pathlib import Path
from typing import Any, Dict, Sequence, Union
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PathLike = Union[str, Path]


class SyntheticTsTabDataset(Dataset):
    """Minimal dataset for synthetic irregular TS + tabular samples stored in pickle files."""

    def __init__(self, pkl_path: PathLike, return_latent: bool = False) -> None:
        self.pkl_path = Path(pkl_path)
        self.return_latent = return_latent

        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {self.pkl_path}")

        with open(self.pkl_path, "rb") as f:
            self.samples = pickle.load(f)

        if not isinstance(self.samples, list):
            raise ValueError(f"Expected pickle file to contain a list, got {type(self.samples)}")
        if len(self.samples) == 0:
            raise ValueError(f"Dataset is empty: {self.pkl_path}")

        self._validate_sample(self.samples[0])

    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        required = {"id", "ts_values", "ts_times", "ts_mask", "tab_num", "tab_cat"}
        missing = required.difference(sample.keys())
        if missing:
            raise ValueError(f"Sample is missing required keys: {sorted(missing)}")

        ts_values = np.asarray(sample["ts_values"])
        ts_times = np.asarray(sample["ts_times"])
        ts_mask = np.asarray(sample["ts_mask"])
        tab_num = np.asarray(sample["tab_num"])
        tab_cat = np.asarray(sample["tab_cat"])

        if ts_values.ndim != 2:
            raise ValueError(f"ts_values must be 2D, got shape {ts_values.shape}")
        if ts_times.ndim != 1:
            raise ValueError(f"ts_times must be 1D, got shape {ts_times.shape}")
        if ts_mask.shape != ts_values.shape:
            raise ValueError(
                f"ts_mask must have same shape as ts_values, got {ts_mask.shape} vs {ts_values.shape}"
            )
        if ts_values.shape[0] != ts_times.shape[0]:
            raise ValueError(
                f"ts_times length must match ts_values seq_len, got {ts_times.shape[0]} vs {ts_values.shape[0]}"
            )
        if tab_num.ndim != 1:
            raise ValueError(f"tab_num must be 1D, got shape {tab_num.shape}")
        if tab_cat.ndim != 1:
            raise ValueError(f"tab_cat must be 1D, got shape {tab_cat.shape}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        out = {
            "id": int(sample["id"]),
            "ts_values": torch.as_tensor(sample["ts_values"], dtype=torch.float32),
            "ts_times": torch.as_tensor(sample["ts_times"], dtype=torch.float32),
            "ts_mask": torch.as_tensor(sample["ts_mask"], dtype=torch.float32),
            "tab_num": torch.as_tensor(sample["tab_num"], dtype=torch.float32),
            "tab_cat": torch.as_tensor(sample["tab_cat"], dtype=torch.long),
        }
        if self.return_latent and "latent_z" in sample:
            out["latent_z"] = torch.as_tensor(sample["latent_z"], dtype=torch.float32)
        return out


def synthetic_ts_tab_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length irregular time series and stack tabular features."""
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(batch)
    lengths = torch.tensor([item["ts_values"].size(0) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    ts_dim = batch[0]["ts_values"].size(1)

    ts_values = torch.zeros(batch_size, max_len, ts_dim, dtype=torch.float32)
    ts_mask = torch.zeros(batch_size, max_len, ts_dim, dtype=torch.float32)
    ts_times = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        seq_len = item["ts_values"].size(0)
        if item["ts_values"].size(1) != ts_dim:
            raise ValueError("All samples in a batch must have the same ts_dim")
        ts_values[i, :seq_len] = item["ts_values"]
        ts_mask[i, :seq_len] = item["ts_mask"]
        ts_times[i, :seq_len] = item["ts_times"]

    out = {
        "ids": torch.tensor([item["id"] for item in batch], dtype=torch.long),
        "lengths": lengths,
        "ts_values": ts_values,
        "ts_times": ts_times,
        "ts_mask": ts_mask,
        "tab_num": torch.stack([item["tab_num"] for item in batch], dim=0),
        "tab_cat": torch.stack([item["tab_cat"] for item in batch], dim=0),
    }

    if all("latent_z" in item for item in batch):
        out["latent_z"] = torch.stack([item["latent_z"] for item in batch], dim=0)

    return out


def make_synthetic_ts_tab_dataloader(
    pkl_path: PathLike,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    return_latent: bool = False,
) -> DataLoader:
    dataset = SyntheticTsTabDataset(pkl_path=pkl_path, return_latent=return_latent)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=synthetic_ts_tab_collate_fn,
    )


__all__ = [
    "SyntheticTsTabDataset",
    "synthetic_ts_tab_collate_fn",
    "make_synthetic_ts_tab_dataloader",
]
