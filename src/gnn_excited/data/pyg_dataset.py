from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

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

DEFAULT_TARGET_COLUMNS = ('S1_eV', 'log1p_S1_f')


class QCDGES1Dataset(Dataset):
    '''PyG dataset for QCDGE excited-state targets.'''

    def __init__(
        self,
        hdf5_path: str | Path,
        manifest_path: str | Path,
        molecule_keys: Sequence[str] | None = None,
        target_columns: Sequence[str] | None = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                'QCDGES1Dataset requires torch and torch_geometric. Install the ML environment first.'
            ) from _IMPORT_ERROR
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.target_columns = tuple(target_columns or DEFAULT_TARGET_COLUMNS)
        if not self.target_columns:
            raise ValueError('At least one target column is required')
        self.rows = self._load_rows(Path(manifest_path), molecule_keys, self.target_columns)
        self._hdf5_handle: h5py.File | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hdf5_handle'] = None
        return state

    def __del__(self):
        handle = getattr(self, '_hdf5_handle', None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    @staticmethod
    def _load_rows(
        manifest_path: Path,
        molecule_keys: Sequence[str] | None,
        target_columns: Sequence[str],
    ) -> list[dict[str, str]]:
        allowed = set(molecule_keys) if molecule_keys is not None else None
        rows: list[dict[str, str]] = []
        with manifest_path.open('r', newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            missing = [column for column in target_columns if column not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f'Manifest {manifest_path} is missing target columns: {missing}')
            for row in reader:
                if row.get('status') != 'ok':
                    continue
                if allowed is not None and row['molecule_key'] not in allowed:
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(f'No usable rows found in manifest {manifest_path}')
        return rows

    def _handle(self) -> h5py.File:
        if self._hdf5_handle is None:
            self._hdf5_handle = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_handle

    def _read_molecule_arrays(self, molecule_key: str) -> tuple[np.ndarray, np.ndarray]:
        group = self._handle()[str(molecule_key)]['ground_state']
        z = np.asarray(group['labels'][()]).reshape(-1).astype(np.int64, copy=False)
        pos = np.asarray(group['coords'][()], dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f'Molecule {molecule_key} has invalid coords shape {pos.shape}')
        if z.shape[0] != pos.shape[0]:
            raise ValueError(f'Molecule {molecule_key} labels/coords atom-count mismatch: {z.shape[0]} vs {pos.shape[0]}')
        return z, pos

    def len(self) -> int:
        return len(self.rows)

    def get(self, idx: int):
        row = self.rows[idx]
        z_np, pos_np = self._read_molecule_arrays(row['molecule_key'])
        y_values = [float(row[column]) for column in self.target_columns]
        y = torch.tensor([y_values], dtype=torch.float32)
        return Data(
            z=torch.as_tensor(z_np, dtype=torch.long),
            pos=torch.as_tensor(pos_np, dtype=torch.float32),
            y=y,
            molecule_key=row['molecule_key'],
        )


def deterministic_split(rows: Sequence[dict[str, str]], seed: int, train_fraction: float, val_fraction: float):
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError('Expected train_fraction > 0, val_fraction >= 0, and train + val < 1')
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    n_train = int(len(indices) * train_fraction)
    n_val = int(len(indices) * val_fraction)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
