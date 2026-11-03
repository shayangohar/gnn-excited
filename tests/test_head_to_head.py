"""Tests for the ViSNetGWBSE head-to-head helpers (TASK-271)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_external_testsets import connectivity_key, read_xyz, set_is_hartree, visnetgwbse_split  # noqa: E402
from head_to_head_report import blend_sorted, guarded_blend, paired_bootstrap  # noqa: E402


def test_visnetgwbse_split_matches_their_global_seed_shuffle():
    split = visnetgwbse_split()
    sizes = {name: sum(1 for s in split.values() if s == name) for name in ("train", "val", "test")}
    assert sizes == {"train": 120000, "val": 10000, "test": 3885}
    idx = list(range(1, 133886))
    random.seed(42)  # data/qm9gwbse.py seeds the module RNG, then shuffles once
    random.shuffle(idx)
    assert [split[i] for i in idx[:3]] == ["train"] * 3
    assert all(split[i] == "test" for i in idx[130000:])
    assert all(split[i] == "val" for i in idx[120000:130000])


def test_read_xyz_plain_and_extended(tmp_path):
    ext = tmp_path / "a.xyz"
    ext.write_text('2\nProperties=species:S:1:pos:R:3 pbc="F F F"\nC 0.0 0.0 0.0\nO 0.0 0.0 1.2\n')
    plain = tmp_path / "b.xyz"
    plain.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    symbols, pos = read_xyz(ext)
    assert symbols == ["C", "O"] and pos.shape == (2, 3) and pos[1, 2] == pytest.approx(1.2)
    assert read_xyz(plain)[0] == ["H", "H"]


def test_set_units_decided_per_set():
    assert set_is_hartree([np.array([0.15, 0.2, 0.22]), np.array([0.18, 0.19])])
    # the QM9 set is eV but contains one S1 below 2 eV
    assert not set_is_hartree([np.array([5.43]), np.array([1.57]), np.array([6.1])])


def test_connectivity_key_ignores_small_geometry_changes():
    pytest.importorskip("rdkit")
    numbers = np.array([6, 8, 1, 1])
    pos = np.array([[0, 0, 0], [0, 0, 1.205], [0, 0.936, -0.579], [0, -0.936, -0.579]], dtype=float)
    key = connectivity_key(numbers, pos)
    assert key is not None
    assert connectivity_key(numbers, pos + np.random.default_rng(0).normal(0, 0.01, pos.shape)) == key
    methanol = connectivity_key(np.array([6, 8, 1, 1, 1, 1]),
                                np.array([[0, 0, 0], [1.43, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89],
                                          [-0.36, -0.51, -0.89], [1.75, 0.9, 0]], dtype=float))
    assert methanol != key


def test_checkpoint_name_parsing():
    pytest.importorskip("torch")
    from eval_visnetgwbse import parse_checkpoint_name

    assert parse_checkpoint_name("gapbse05M128502532") == {
        "prop": "gapbse", "n_pre_millions": 0.5, "hidden_channels": 128, "cutoff": 5.0, "num_rbf": 25,
        "num_layers": 3, "lmax": 2}
    assert parse_checkpoint_name("homo10M216502532")["hidden_channels"] == 216
    assert parse_checkpoint_name("bse0M128502532")["n_pre_millions"] == 0.0


def test_blend_sorted_and_bootstrap():
    a = {"m": np.array([3.0, 1.0, 2.0, 4.0, 5.0])}
    b = {"m": np.array([1.0, 1.0, 2.0, 4.0, 5.0])}
    out = blend_sorted([[a], [b]], [0.5, 0.5], ["m"])
    assert out[0].tolist() == [1.0, 2.0, 2.0, 4.0, 5.0]
    lo, hi = paired_bootstrap(np.full(50, 0.1), np.full(50, 0.3))
    assert lo == pytest.approx(-0.2) and hi == pytest.approx(-0.2)


def test_guarded_blend_drops_a_failed_family_per_molecule():
    good = {"a": np.array([3.0, 4.0, 5.0, 6.0, 7.0]), "b": np.array([3.0, 4.0, 5.0, 6.0, 7.0])}
    bad = {"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "b": np.array([1e9, 2.0, 3.0, 4.0, 5.0])}
    out, guarded = guarded_blend([[good], [bad]], [0.6, 0.4], ["a", "b"])
    assert guarded == 1
    assert out[0].tolist() == pytest.approx([0.6 * 3 + 0.4 * 1, 0.6 * 4 + 0.4 * 2, 0.6 * 5 + 0.4 * 3,
                                             0.6 * 6 + 0.4 * 4, 0.6 * 7 + 0.4 * 5])
    assert out[1].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]  # "b" falls back to the good family alone
    _, none = guarded_blend([[bad]], [1.0], ["b"])
    assert none == 1
