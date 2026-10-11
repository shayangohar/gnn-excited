"""PyG dataset over packed PubChemQC TD-DFT singlets (schema pubchemqc-st-packed-v1).

Written by scripts/pack_pubchemqc.py. Molecule i occupies the contiguous atom
slice atom_offsets[i]:atom_offsets[i+1]; S_eV holds S1..S10 in true eV. The
arrays (~1.5 GB for 3.2M molecules) load once per process and are shared
copy-on-write with forked DataLoader workers.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

try:
    import torch
    from torch_geometric.data import Data, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without PyG.
    torch = None
    Dataset = object
    Data = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

PACKED_SCHEMA = 'pubchemqc-st-packed-v1'
N_STATES = 10
DEFAULT_TARGET_COLUMNS = ('S1_eV', 'S2_eV', 'S3_eV', 'S4_eV', 'S5_eV')
_ARRAY_CACHE: dict[str, dict] = {}


def state_column_indices(target_columns: Sequence[str]) -> np.ndarray:
    '''Map S<k>_eV target columns to S_eV column indices (PubChemQC has singlets only).'''
    indices = []
    for column in target_columns:
        match = re.fullmatch(r'S(\d+)_eV', str(column))
        if match is None or not 1 <= int(match.group(1)) <= N_STATES:
            raise ValueError(f'PubChemQC provides S1_eV..S{N_STATES}_eV only, got {column!r}')
        indices.append(int(match.group(1)) - 1)
    if not indices:
        raise ValueError('At least one target column is required')
    return np.asarray(indices, dtype=np.int64)


def load_packed_arrays(hdf5_path: str | Path) -> dict:
    key = str(Path(hdf5_path).resolve())
    cached = _ARRAY_CACHE.get(key)
    if cached is None:
        with h5py.File(hdf5_path, 'r') as handle:
            schema = handle.attrs.get('schema')
            if str(schema) != PACKED_SCHEMA:
                raise ValueError(f'{hdf5_path}: schema {schema!r} != {PACKED_SCHEMA}')
            keys = handle['molecule_key'][()]
            cached = {
                'numbers': handle['numbers'][()],
                'positions': handle['positions'][()],
                'atom_offsets': handle['atom_offsets'][()],
                'S_eV': handle['S_eV'][()],
                'molecule_key': keys,
                'index': {value.decode(): i for i, value in enumerate(keys)},
            }
        _ARRAY_CACHE[key] = cached
    return cached


class PubChemQCPackedDataset(Dataset):
    '''PyG dataset over the packed PubChemQC HDF5, keyed by manifest molecule_key.'''

    def __init__(
        self,
        hdf5_path: str | Path,
        manifest_path: str | Path,
        molecule_keys: Sequence[str] | None = None,
        target_columns: Sequence[str] | None = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                'PubChemQCPackedDataset requires torch and torch_geometric. Install the ML environment first.'
            ) from _IMPORT_ERROR
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.target_columns = tuple(target_columns or DEFAULT_TARGET_COLUMNS)
        self._columns = state_column_indices(self.target_columns)
        if molecule_keys is None:
            with Path(manifest_path).open('r', newline='', encoding='utf-8') as stream:
                molecule_keys = [row['molecule_key'] for row in csv.DictReader(stream) if row.get('status') == 'ok']
        index = load_packed_arrays(self.hdf5_path)['index']
        missing = [key for key in molecule_keys if key not in index]
        if missing:
            raise ValueError(f'{len(missing)} molecule keys missing from {self.hdf5_path}; first={missing[0]}')
        self._pack_rows = np.fromiter((index[key] for key in molecule_keys), dtype=np.int64, count=len(molecule_keys))
        if self._pack_rows.size == 0:
            raise ValueError(f'No usable rows found in manifest {manifest_path}')

    def len(self) -> int:
        return int(self._pack_rows.size)

    def get(self, idx: int):
        arrays = load_packed_arrays(self.hdf5_path)
        i = int(self._pack_rows[idx])
        start, end = int(arrays['atom_offsets'][i]), int(arrays['atom_offsets'][i + 1])
        y = arrays['S_eV'][i, self._columns].astype(np.float32)
        return Data(
            z=torch.from_numpy(arrays['numbers'][start:end].astype(np.int64)),
            pos=torch.from_numpy(np.array(arrays['positions'][start:end], dtype=np.float32)),
            y=torch.from_numpy(y).view(1, -1),
            molecule_key=arrays['molecule_key'][i].decode(),
        )
