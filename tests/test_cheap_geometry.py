"""Tests for SMILES-to-geometry helpers, the rCEG comparison builder and per-molecule charge/spin (TASK-291 E3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("rdkit")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cheap_geometry import charge_and_multiplicity, embed, heavy_atom_rmsd  # noqa: E402


def test_charge_and_multiplicity_from_smiles():
    assert charge_and_multiplicity("C[N+](C)(C)C") == (1, 1)
    assert charge_and_multiplicity("[CH3]") == (0, 2)
    assert charge_and_multiplicity("not a smiles") is None


def test_embed_and_rmsd():
    numbers, positions = embed("CCO", seed=0)
    assert sorted(numbers.tolist()) == [1, 1, 1, 1, 1, 1, 6, 6, 8] and np.isfinite(positions).all()
    assert heavy_atom_rmsd(numbers, positions, numbers, positions) == pytest.approx(0.0, abs=1e-6)
    moved = positions + np.random.default_rng(0).normal(0, 0.02, positions.shape)
    assert 0 < heavy_atom_rmsd(numbers, positions, numbers, moved) < 0.1
    uff = embed("CCO", seed=0, forcefield="UFF")
    assert uff is not None and uff[0].size == 9


def test_canonical_ignores_stereo():
    from build_rceg_comparison import canonical

    assert canonical("C/C=C/C") == canonical("C/C=C\\C") == canonical("CC=CC")
    assert canonical("not a smiles") is None


def test_charge_spin_defaults_to_neutral_singlets():
    pytest.importorskip("torch")
    from train_uma_qm9gwbse import _charge_spin

    charge, spin = _charge_spin({}, 3)
    assert charge.tolist() == [0, 0, 0] and spin.tolist() == [1, 1, 1]
    charge, spin = _charge_spin({"charge": np.array([1, 0]), "spin": np.array([2, 1])}, 2)
    assert charge.tolist() == [1, 0] and spin.tolist() == [2, 1]
