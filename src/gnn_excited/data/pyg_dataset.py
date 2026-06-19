from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from gnn_excited.data.qcdge import read_molecule_arrays

try:
    import torch
    from torch_geometric.data import Dataset, Data
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in environments without PyG.
    torch = None
    Dataset = object
    Data = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class QCDGES1Dataset(Dataset):
    """PyG dataset for QCDGE S1 energy and oscillator-strength targets."""

    def __init__(self, hdf5_path: str | Path, manifest_path: str | Path, molecule_keys: Sequence[str] | None = None):
        if _IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "QCDGES1Dataset requires torch and torch_geometric. Install the ML environment first."
            ) from _IMPORT_ERROR
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.rows = self._load_rows(Path(manifest_path), molecule_keys)

    @staticmethod
    def _load_rows(manifest_path: Path, molecule_keys: Sequence[str] | None) -> list[dict[str, str]]:
        allowed = set(molecule_keys) if molecule_keys is not None else None
        rows: list[dict[str, str]] = []
        with manifest_path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") != "ok":
                    continue
                if allowed is not None and row["molecule_key"] not in allowed:
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(f"No usable rows found in manifest {manifest_path}")
        return rows

    def len(self) -> int:
        return len(self.rows)

    def get(self, idx: int):
        row = self.rows[idx]
        z_np, pos_np = read_molecule_arrays(self.hdf5_path, row["molecule_key"])
        y = torch.tensor(
            [[float(row["S1_eV"]), math.log1p(float(row["S1_f"]))]],
            dtype=torch.float32,
        )
        return Data(
            z=torch.as_tensor(z_np, dtype=torch.long),
            pos=torch.as_tensor(np.asarray(pos_np), dtype=torch.float32),
            y=y,
            molecule_key=row["molecule_key"],
        )


def deterministic_split(rows: Sequence[dict[str, str]], seed: int, train_fraction: float, val_fraction: float):
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Expected train_fraction > 0, val_fraction >= 0, and train + val < 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    n_train = int(len(indices) * train_fraction)
    n_val = int(len(indices) * val_fraction)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
