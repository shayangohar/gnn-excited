"""Tests for scripts/make_qm9gwbse_subsets.py and scripts/calibrate_uncertainty.py (TASK-250)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_uncertainty import blend, conformal_quantile, spearman  # noqa: E402
from make_qm9gwbse_subsets import heavy_atom_counts, subsample_exclusions  # noqa: E402


def test_heavy_atom_counts_per_molecule():
    numbers = np.array([6, 1, 1, 1, 1, 8, 1, 1, 7, 6])
    offsets = np.array([0, 5, 8, 10])
    assert heavy_atom_counts(numbers, offsets).tolist() == [1, 1, 2]


def test_subsample_exclusions_keeps_exact_count():
    rng = np.random.default_rng(0)
    idx = np.arange(100, 200)
    drop = subsample_exclusions(idx, 25, rng)
    assert drop.size == 75 and set(drop) <= set(idx)
    with pytest.raises(ValueError):
        subsample_exclusions(idx, 101, rng)


def test_conformal_quantile_gives_finite_sample_coverage():
    rng = np.random.default_rng(1)
    cal, new = np.abs(rng.normal(size=2000)), np.abs(rng.normal(size=20000))
    q = conformal_quantile(cal, 0.9)
    assert 0.885 <= (new <= q).mean() <= 0.92


def test_blend_sorts_energies_and_uses_all_members_for_sigma():
    a = [np.array([[2.0, 1.0]]), np.array([[2.2, 1.2]])]
    b = [np.array([[1.8, 1.4]])]
    mean, sigma = blend([a, b], [0.5, 0.5])
    assert mean[0, 0] <= mean[0, 1]
    assert sigma.shape == (1, 2) and sigma[0, 0] == pytest.approx(np.std([2.0, 2.2, 1.8]))
    assert spearman(np.array([1, 2, 3.0]), np.array([10, 20, 30.0])) == pytest.approx(1.0)
