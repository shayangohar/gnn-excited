from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

_EV_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class S1Target:
    molecule_key: str
    atom_count: int
    s1_ev: float
    s1_f: float
    state_index: int

    @property
    def log1p_s1_f(self) -> float:
        return math.log1p(self.s1_f)


def parse_energy_ev(value: Any) -> float:
    """Parse QCDGE energy strings such as '7.7710 eV' into eV floats."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = _EV_RE.search(str(value))
    if match is None:
        raise ValueError(f"Could not parse excitation energy from {value!r}")
    return float(match.group(0))


def decode_excited_state_info(raw: Any) -> dict[str, Any]:
    """Decode the JSON object stored in Info_of_AllExcitedStates."""
    value = raw[()]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected one JSON payload, found shape {value.shape}")
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"Unsupported excited-state payload type {type(value)!r}")
    return json.loads(value)


def extract_lowest_singlet(info: dict[str, Any]) -> tuple[int, float, float]:
    """Return (state_index, excitation_e_eV, oscillator_strength) for S1."""
    singlets: list[tuple[int, float, float]] = []
    for key, record in info.items():
        if str(record.get("state_type", "")).lower() != "singlet":
            continue
        state_index = int(record.get("state", key))
        energy = parse_energy_ev(record["excitation_e_eV"])
        osc = float(record.get("oscillator_trength", 0.0))
        if osc < 0:
            raise ValueError(f"Negative oscillator strength for state {state_index}: {osc}")
        singlets.append((state_index, energy, osc))
    if not singlets:
        raise ValueError("No singlet excited states found")
    return min(singlets, key=lambda item: item[1])


def _flatten_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    return labels.astype(np.int64, copy=False)


def read_molecule_arrays(hdf5_path: str | Path, molecule_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Read atomic numbers and coordinates for one molecule."""
    with h5py.File(hdf5_path, "r") as handle:
        group = handle[str(molecule_key)]["ground_state"]
        z = _flatten_labels(group["labels"][()])
        pos = np.asarray(group["coords"][()], dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected coords with shape (n_atoms, 3), found {pos.shape}")
    if len(z) != pos.shape[0]:
        raise ValueError(f"labels/coords atom count mismatch: {len(z)} vs {pos.shape[0]}")
    return z, pos


def read_s1_target(hdf5_path: str | Path, molecule_key: str) -> S1Target:
    """Read the v1 target for one molecule without loading unrelated molecules."""
    with h5py.File(hdf5_path, "r") as handle:
        return s1_target_from_group(str(molecule_key), handle[str(molecule_key)])


def s1_target_from_group(molecule_key: str, mol: h5py.Group) -> S1Target:
    """Extract the v1 target from an already-open molecule group."""
    z = _flatten_labels(mol["ground_state"]["labels"][()])
    info = decode_excited_state_info(mol["excited_state"]["Info_of_AllExcitedStates"])
    state_index, s1_ev, s1_f = extract_lowest_singlet(info)
    return S1Target(str(molecule_key), len(z), s1_ev, s1_f, state_index)


def iter_molecule_keys(hdf5_path: str | Path, max_count: int | None = None) -> Iterable[str]:
    """Yield molecule keys lazily from the HDF5 root group."""
    with h5py.File(hdf5_path, "r") as handle:
        for idx, key in enumerate(handle.keys()):
            if max_count is not None and idx >= max_count:
                break
            yield str(key)


def build_manifest(
    hdf5_path: str | Path,
    output_path: str | Path,
    max_count: int | None = None,
    progress_every: int = 25,
) -> dict[str, int]:
    """Build a CSV manifest with parse status for each attempted molecule."""
    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "error": 0}
    fieldnames = [
        "molecule_key",
        "atom_count",
        "S1_eV",
        "S1_f",
        "log1p_S1_f",
        "s1_state_index",
        "status",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        stream.flush()
        with h5py.File(hdf5_path, "r") as handle:
            for seen, key in enumerate(handle.keys(), start=1):
                if max_count is not None and seen > max_count:
                    break
                try:
                    target = s1_target_from_group(str(key), handle[str(key)])
                    writer.writerow(
                        {
                            "molecule_key": target.molecule_key,
                            "atom_count": target.atom_count,
                            "S1_eV": target.s1_ev,
                            "S1_f": target.s1_f,
                            "log1p_S1_f": target.log1p_s1_f,
                            "s1_state_index": target.state_index,
                            "status": "ok",
                            "error": "",
                        }
                    )
                    counts["ok"] += 1
                except Exception as exc:  # noqa: BLE001 - manifest should record parse failures.
                    writer.writerow(
                        {
                            "molecule_key": key,
                            "atom_count": "",
                            "S1_eV": "",
                            "S1_f": "",
                            "log1p_S1_f": "",
                            "s1_state_index": "",
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                    counts["error"] += 1
                if progress_every and seen % progress_every == 0:
                    print(f"processed {seen} molecules ({counts['ok']} ok, {counts['error']} errors)", flush=True)
                stream.flush()
    return counts
