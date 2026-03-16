
import importlib.util
import pickle
from pathlib import Path

import torch


def _load_datasets_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "data" / "datasets.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find datasets module at: {module_path}")

    spec = importlib.util.spec_from_file_location("datasets_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


datasets_mod = _load_datasets_module()
SyntheticTsTabDataset = datasets_mod.SyntheticTsTabDataset
synthetic_ts_tab_collate_fn = datasets_mod.synthetic_ts_tab_collate_fn
make_synthetic_ts_tab_dataloader = datasets_mod.make_synthetic_ts_tab_dataloader


def _write_tiny_pickle(tmp_path: Path):
    samples = [
        {
            "id": 0,
            "ts_values": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "ts_times": [0.1, 0.4, 0.9],
            "ts_mask": [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            "tab_num": [0.5, -1.2, 0.7],
            "tab_cat": [1, 0],
            "latent_z": [0.1, -0.2],
        },
        {
            "id": 1,
            "ts_values": [[1.0, 1.1], [1.2, 1.3]],
            "ts_times": [0.2, 0.8],
            "ts_mask": [[1.0, 1.0], [1.0, 1.0]],
            "tab_num": [1.5, 0.2, -0.7],
            "tab_cat": [0, 1],
            "latent_z": [0.4, 0.3],
        },
    ]
    pkl_path = tmp_path / "tiny.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(samples, f)
    return pkl_path


def test_dataset_reads_one_sample(tmp_path):
    pkl_path = _write_tiny_pickle(tmp_path)
    dataset = SyntheticTsTabDataset(pkl_path, return_latent=True)

    item = dataset[0]

    assert len(dataset) == 2
    assert item["id"] == 0
    assert item["ts_values"].shape == (3, 2)
    assert item["ts_times"].shape == (3,)
    assert item["ts_mask"].shape == (3, 2)
    assert item["tab_num"].shape == (3,)
    assert item["tab_cat"].shape == (2,)
    assert item["latent_z"].shape == (2,)
    assert item["ts_values"].dtype == torch.float32
    assert item["tab_cat"].dtype == torch.long


def test_collate_pads_variable_length_sequences(tmp_path):
    pkl_path = _write_tiny_pickle(tmp_path)
    dataset = SyntheticTsTabDataset(pkl_path, return_latent=True)

    batch = synthetic_ts_tab_collate_fn([dataset[0], dataset[1]])

    assert batch["ids"].shape == (2,)
    assert batch["lengths"].tolist() == [3, 2]
    assert batch["ts_values"].shape == (2, 3, 2)
    assert batch["ts_times"].shape == (2, 3)
    assert batch["ts_mask"].shape == (2, 3, 2)
    assert batch["tab_num"].shape == (2, 3)
    assert batch["tab_cat"].shape == (2, 2)
    assert batch["latent_z"].shape == (2, 2)

    # Second sample is padded at the last step.
    assert torch.allclose(batch["ts_values"][1, 2], torch.zeros(2))
    assert torch.allclose(batch["ts_mask"][1, 2], torch.zeros(2))
    assert torch.allclose(batch["ts_times"][1, 2], torch.tensor(0.0))


def test_dataloader_returns_expected_batch(tmp_path):
    pkl_path = _write_tiny_pickle(tmp_path)
    loader = make_synthetic_ts_tab_dataloader(
        pkl_path,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        return_latent=True,
    )

    batch = next(iter(loader))

    assert batch["ts_values"].shape == (2, 3, 2)
    assert batch["ts_times"].shape == (2, 3)
    assert batch["ts_mask"].shape == (2, 3, 2)
    assert batch["tab_num"].shape == (2, 3)
    assert batch["tab_cat"].shape == (2, 2)
    assert batch["latent_z"].shape == (2, 2)
    assert torch.equal(batch["ids"], torch.tensor([0, 1]))
