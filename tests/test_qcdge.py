from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from gnn_excited.data.qcdge import build_manifest, extract_lowest_singlet, parse_energy_ev, read_molecule_arrays, read_s1_target


def _write_test_hdf5(path: Path) -> None:
    info = {
        "1": {"state": 1, "state_type": "Triplet", "oscillator_trength": 0.0, "excitation_e_eV": "2.5 eV"},
        "2": {"state": 2, "state_type": "Singlet", "oscillator_trength": 0.12, "excitation_e_eV": "4.1 eV"},
        "3": {"state": 3, "state_type": "Singlet", "oscillator_trength": 0.02, "excitation_e_eV": "3.7 eV"},
    }
    with h5py.File(path, "w") as handle:
        mol = handle.create_group("10")
        gs = mol.create_group("ground_state")
        gs.create_dataset("labels", data=np.array([[6, 1, 1, 1]], dtype=np.int64))
        gs.create_dataset("coords", data=np.zeros((4, 3), dtype=np.float64))
        es = mol.create_group("excited_state")
        dt = h5py.string_dtype(encoding="utf-8")
        es.create_dataset("Info_of_AllExcitedStates", data=np.array([json.dumps(info)], dtype=dt))


def test_parse_energy_ev_from_qcdge_string() -> None:
    assert parse_energy_ev("7.7710 eV") == pytest.approx(7.7710)


def test_extract_lowest_singlet() -> None:
    state, energy, osc = extract_lowest_singlet(
        {
            "1": {"state": 1, "state_type": "Singlet", "excitation_e_eV": "5.0 eV", "oscillator_trength": 0.4},
            "2": {"state": 2, "state_type": "Triplet", "excitation_e_eV": "2.0 eV", "oscillator_trength": 0.0},
            "3": {"state": 3, "state_type": "Singlet", "excitation_e_eV": "4.0 eV", "oscillator_trength": 0.1},
        }
    )
    assert state == 3
    assert energy == pytest.approx(4.0)
    assert osc == pytest.approx(0.1)


def test_read_s1_target_and_arrays_from_synthetic_hdf5(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "sample.hdf5"
    _write_test_hdf5(hdf5_path)

    target = read_s1_target(hdf5_path, "10")
    assert target.molecule_key == "10"
    assert target.atom_count == 4
    assert target.s1_ev == pytest.approx(3.7)
    assert target.s1_f == pytest.approx(0.02)

    z, pos = read_molecule_arrays(hdf5_path, "10")
    assert z.tolist() == [6, 1, 1, 1]
    assert pos.shape == (4, 3)


def test_build_manifest_records_parse_status(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "sample.hdf5"
    manifest_path = tmp_path / "manifest.csv"
    _write_test_hdf5(hdf5_path)

    counts = build_manifest(hdf5_path, manifest_path, max_count=1)
    text = manifest_path.read_text(encoding="utf-8")
    assert counts == {"ok": 1, "error": 0}
    assert "S1_eV" in text
    assert "3.7" in text


def test_real_a9_known_key_if_present() -> None:
    hdf5_path = Path("data/A_9.hdf5")
    if not hdf5_path.exists():
        pytest.skip("A_9.hdf5 is not installed")
    target = read_s1_target(hdf5_path, "10")
    assert target.s1_ev > 0
    assert target.s1_f >= 0
