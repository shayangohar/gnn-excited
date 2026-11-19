"""Tests for the wave G pieces (TASK-320): mixed-geometry packing, S-free shot draws, conformer selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _packed(keys, atoms, split, offset=0.0):
    counts = np.array(atoms)
    n = int(counts.sum())
    return {"numbers": np.full(n, 6, dtype=np.uint8), "positions": np.arange(3 * n, dtype=np.float32).reshape(n, 3) + offset,
            "atom_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            "molecule_key": np.array(keys), "split": np.array(split),
            "targets": np.arange(len(keys) * 2, dtype=np.float32).reshape(len(keys), 2),
            "target_columns": np.array(["S1_eV", "S2_eV"])}


def test_mixed_packing_keeps_labels_and_splits():
    from build_mixed_packed import merge

    b3 = _packed(["1", "2", "3"], [2, 3, 1], ["train", "val", "test"])
    xtb = _packed(["1", "3"], [2, 1], ["train", "test"], offset=100.0)
    xtb["targets"] = b3["targets"][[0, 2]]
    xtb["source_row"] = np.array([0, 2])
    out = merge(b3, xtb)
    assert out["molecule_key"].tolist() == ["1", "2", "3", "1:xtb", "3:xtb"]
    assert out["split"].tolist() == ["train", "val", "test", "train", "test"]
    assert out["geometry_source"].tolist() == ["b3lyp"] * 3 + ["xtb"] * 2
    assert out["atom_offsets"].tolist() == [0, 2, 5, 6, 8, 9]
    assert np.allclose(out["positions"][6:8], xtb["positions"][0:2])
    assert np.allclose(out["targets"][3], out["targets"][0]) and np.allclose(out["targets"][4], out["targets"][2])
    assert "source_row" not in out and out["target_columns"].tolist() == ["S1_eV", "S2_eV"]


def test_exclusion_draws_never_use_the_element():
    pytest.importorskip("yaml")
    from build_oe62h_shots import exclusion_draws

    keys = [f"m{i}" for i in range(40)]
    elements = {k: ({"C", "S"} if i % 2 else {"C", "Cl"}) for i, k in enumerate(keys)}
    draws = exclusion_draws(keys, elements, "S", [5, 10], draws=3, seed=0)
    assert len(draws) == 3
    for d in draws:
        assert all("S" not in elements[k] for k in d["shots"][10])
        assert set(d["shots"][5]) <= set(d["shots"][10])
        assert set(d["held_out"]) == set(keys) - set(d["shots"][10])
        assert all(k in d["held_out"] for k in keys if "S" in elements[k])
    assert draws[0]["shots"][10] != draws[1]["shots"][10]


def test_conformer_selection():
    from conformer_report import KT_EH, select

    confs = [{"j": 0, "energy": -10.0, "rmsd": 0.9, "pred": np.array([5.0, 6.0])},
             {"j": 1, "energy": -10.0 - 10 * KT_EH, "rmsd": 0.2, "pred": np.array([4.0, 5.0])},
             {"j": 2, "energy": -9.0, "rmsd": None, "pred": np.array([7.0, 8.0])}]
    assert select(confs, "c0").tolist() == [5.0, 6.0]
    assert select(confs, "min_energy").tolist() == [4.0, 5.0]
    assert select(confs, "oracle_rmsd").tolist() == [4.0, 5.0]
    assert np.allclose(select(confs, "mean"), [16 / 3, 19 / 3])
    assert np.allclose(select(confs, "boltzmann"), [4.0, 5.0], atol=1e-3)  # j=1 dominates at 10 kT lower
