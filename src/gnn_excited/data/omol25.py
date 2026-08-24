"""PyG dataset for OMol25 orbital-gap pretraining (schema omol25-gap-v1)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

try:
    import torch
    from torch_geometric.data import Dataset, Data
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without PyG.
    torch = None
    Dataset = object
    Data = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEFAULT_TARGET_COLUMNS = ('gap',)


class Omol25GapDataset(Dataset):
    '''PyG dataset over OMol25 HDF5 groups keyed by molecule_key.'''

    def __init__(
        self,
        hdf5_path: str | Path,
        manifest_path: str | Path,
        molecule_keys: Sequence[str] | None = None,
        target_columns: Sequence[str] | None = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                'Omol25GapDataset requires torch and torch_geometric. Install the ML environment first.'
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

    def len(self) -> int:
        return len(self.rows)

    def get(self, idx: int):
        row = self.rows[idx]
        group = self._handle()['m'][str(row['molecule_key'])]
        z = np.asarray(group['numbers'][()]).reshape(-1).astype(np.int64, copy=False)
        pos = np.asarray(group['positions'][()], dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"Molecule {row['molecule_key']} has invalid positions shape {pos.shape}")
        if z.shape[0] != pos.shape[0]:
            raise ValueError(
                f"Molecule {row['molecule_key']} numbers/positions mismatch: {z.shape[0]} vs {pos.shape[0]}"
            )
        y_values = [float(row[column]) for column in self.target_columns]
        y = torch.tensor([y_values], dtype=torch.float32)
        return Data(
            z=torch.as_tensor(z, dtype=torch.long),
            pos=torch.as_tensor(pos, dtype=torch.float32),
            y=y,
            molecule_key=row['molecule_key'],
        )
