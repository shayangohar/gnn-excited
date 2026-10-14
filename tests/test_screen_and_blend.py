"""Tests for scripts/screen_pubchemqc.py and scripts/blend_predictions.py (TASK-244)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
pytest.importorskip("scipy")
yaml = pytest.importorskip("yaml")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from blend_predictions import ENERGY, OSCILLATOR, metrics, simplex_grid  # noqa: E402
from screen_pubchemqc import flag_rows, molecule_geometry_stats  # noqa: E402

WATER = (np.array([8, 1, 1]), np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]))


def test_connected_molecule_has_one_graph_and_one_covalent_fragment() -> None:
    graph_parts, covalent_parts, min_dist = molecule_geometry_stats(*WATER, cutoff=5.0)
    assert (graph_parts, covalent_parts) == (1, 1)
    assert min_dist == pytest.approx(0.96, abs=1e-6)


def test_far_fragments_are_graph_disconnected_and_close_ones_only_covalently_split() -> None:
    numbers = np.concatenate([WATER[0], [1, 1]])
    far = np.vstack([WATER[1], [[8.0, 0.0, 0.0], [8.74, 0.0, 0.0]]])  # H2 6.3 A away
    near = np.vstack([WATER[1], [[3.0, 0.0, 0.0], [3.74, 0.0, 0.0]]])  # H2 within the 5 A cutoff
    assert molecule_geometry_stats(numbers, far, cutoff=5.0)[:2] == (2, 2)
    assert molecule_geometry_stats(numbers, near, cutoff=5.0)[:2] == (1, 2)


def test_flags_cover_short_contacts_single_atoms_and_s1_outliers() -> None:
    stats = np.array([
        [3, 1, 1, 0.96],   # clean
        [3, 1, 1, 0.30],   # short contact
        [1, 1, 1, np.inf],  # single atom
        [5, 2, 2, 0.74],   # disconnected
    ])
    s1 = np.array([4.0, 4.0, 4.0, 12.0])
    flags = flag_rows(stats, s1, min_distance=0.5, s1_min=0.5, s1_max=10.0)
    assert flags["short_contact"].tolist() == [False, True, False, False]
    assert flags["single_atom"].tolist() == [False, False, True, False]
    assert flags["graph_disconnected"].tolist() == [False, False, False, True]
    assert flags["s1_outlier"].tolist() == [False, False, False, True]


def test_screen_cli_writes_combined_exclusions(tmp_path: Path) -> None:
    molecules = [WATER, (np.array([1, 1]), np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]))]
    numbers = np.concatenate([m[0] for m in molecules]).astype("u1")
    positions = np.vstack([m[1] for m in molecules]).astype("f4")
    offsets = np.array([0, 3, 5], dtype="i8")
    packed = tmp_path / "packed.h5"
    with h5py.File(packed, "w") as h5:
        h5["numbers"], h5["positions"], h5["atom_offsets"] = numbers, positions, offsets
        h5["S_eV"] = np.tile(np.linspace(4.0, 9.0, 10, dtype="f4"), (2, 1))
        h5["molecule_key"] = np.array([b"pqc_1", b"pqc_2"])
    base = tmp_path / "base.csv"
    base.write_text("molecule_key,inchikey\npqc_9,X\n", encoding="utf-8")
    out = tmp_path / "excl.csv"
    subprocess.run([sys.executable, str(SCRIPTS / "screen_pubchemqc.py"), "--packed-h5", str(packed),
                    "--base-exclusions", str(base), "--exclusions-out", str(out),
                    "--flagged-out", str(tmp_path / "flagged.csv"), "--report-out", str(tmp_path / "report.json"),
                    "--workers", "1", "--chunk", "1"], check=True, capture_output=True)
    with out.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["molecule_key"] for row in rows] == ["pqc_9", "pqc_2"]
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["flag_counts"]["graph_disconnected"] == 1
    assert report["total_exclusion_rows"] == 2


def test_simplex_grid_sums_to_one_and_includes_corners() -> None:
    grid = list(simplex_grid(3, 0.5))
    assert all(abs(w.sum() - 1.0) < 1e-12 for w in grid)
    assert len(grid) == 6
    assert any(np.allclose(w, [0.0, 0.0, 1.0]) for w in grid)


def test_sorting_predictions_never_increases_mae_for_sorted_targets() -> None:
    rng = np.random.default_rng(0)
    target = np.sort(rng.uniform(3, 9, size=(200, 5)), axis=1)
    pred = target + rng.normal(0, 0.3, size=target.shape)
    result = metrics(pred, target)
    assert result["S_agg_sorted"] <= result["S_agg"] + 1e-12
    assert result["adjacent_inversion_rate"] > 0


def _write_member(path: Path, keys, target, pred) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", *[f"{p}_{c}" for c in ENERGY + OSCILLATOR for p in ("target", "prediction")]])
        for k, t, p in zip(keys, target, pred):
            values = []
            for j in range(5):
                values += [t[j], p[j]]
            for _ in range(5):
                values += [0.1, 0.1]
            writer.writerow([k, *values])


def test_blend_picks_weights_on_val_and_beats_members(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    keys = [f"m{i:03d}" for i in range(300)]
    spec = {"grid_step": 0.1, "members": {}, "blends": [["a", "b"]]}
    for split in ("val", "test"):
        target = np.sort(rng.uniform(3, 9, size=(300, 5)), axis=1)
        for name in ("a", "b"):
            path = tmp_path / f"{name}_{split}.csv"
            _write_member(path, keys, target, target + rng.normal(0, 0.2, size=target.shape))
            spec["members"].setdefault(name, {})[split] = str(path)
    spec["members"]["missing"] = {"val": str(tmp_path / "nope.csv"), "test": str(tmp_path / "nope.csv")}
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    out = tmp_path / "blend.json"
    subprocess.run([sys.executable, str(SCRIPTS / "blend_predictions.py"), "--spec", str(spec_path), "--out", str(out)],
                   check=True, capture_output=True)
    report = json.loads(out.read_text(encoding="utf-8"))
    blend = report["blends"][0]
    assert 0.2 <= blend["val_chosen_weights"][0] <= 0.8
    best_single = min(report["single"][n]["test"]["S_agg"] for n in ("a", "b"))
    assert blend["val_weighted_test"]["S_agg"] < best_single
    assert any("missing" in note for note in report["notes"])
