"""Tests for the DOC-327 audit follow-ups (TASK-328): dev/test lock, recalibration arms, CP2K output parsing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _rows():
    rows = []
    for i in range(40):
        rows.append({"molecule_key": f"OE62H:M{i}", "set": "OE62H", "heavy_atoms": str(10 + i % 7),
                     "elements": "C-H-S" if i % 3 else "C-H-Cl" if i % 5 else "C-H-Se"})
    for s in ("QM9", "PC9", "OE62L"):
        rows += [{"molecule_key": f"{s}:{i}", "set": s, "heavy_atoms": str(3 + i % 9), "elements": "C-H-O"} for i in range(31)]
    return rows


def test_split_is_deterministic_balanced_and_stratified():
    from lock_external_split import assign, strata

    rows = _rows()
    groups = strata(rows, min_stratum=4)
    a, b = assign(groups, seed=1), assign(groups, seed=1)
    assert a == b and set(a) == {r["molecule_key"] for r in rows}
    for name, members in groups.items():
        halves = [a[r["molecule_key"]][0] for r in members]
        assert abs(halves.count("dev") - halves.count("test")) <= 1, name
    assert any(name.startswith("OE62H:S") for name in groups)
    assert assign(groups, seed=2) != a


def test_recalibration_arms():
    from fewshot_baselines import fit_predict

    rng = np.random.default_rng(0)
    p_shot = np.sort(rng.normal(5, 1, (30, 5)), axis=1)
    y_shot = p_shot - 0.2
    p_out = np.sort(rng.normal(5, 1, (10, 5)), axis=1)
    assert np.allclose(fit_predict("shift_global", p_shot, y_shot, p_out), p_out - 0.2)
    assert np.allclose(fit_predict("shift_per_state", p_shot, y_shot, p_out), p_out - 0.2)
    y_aff = 0.9 * p_shot + 0.3
    assert np.allclose(fit_predict("affine_per_state", p_shot, np.sort(y_aff, axis=1), p_out), np.sort(0.9 * p_out + 0.3, axis=1), atol=1e-8)
    pytest.importorskip("sklearn")
    x_shot = np.column_stack([np.arange(30) % 2, np.ones(30)])
    y_el = p_shot + np.where(x_shot[:, :1] > 0, 0.5, 0.0)
    out = fit_predict("element_offsets", p_shot, y_el, p_out[:2], x_shot, np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert out[0, 0] - p_out[0, 0] > out[1, 0] - p_out[1, 0] + 0.3


def test_cp2k_parse():
    from cp2k_gwbse import parse

    text = """ BSE| Excitation energies from solving the BSE without the TDA:
 BSE|     Excitation n   Spin Config       TDA/ABBA   Excitation energy Ω^n (eV)
 BSE|                1       Singlet         -ABBA-                      12.0361
 BSE|                2       Singlet         -ABBA-                      13.0327
 BSE|  Excitation n  TDA/ABBA        d_x^n     d_y^n     d_z^n Osc. strength f^n
 BSE|             1    -ABBA-        0.000    -0.000    -1.191             0.418
 BSE|             2    -ABBA-        0.000     0.000     0.000             0.000
 BSE| Total peak memory estimate from BSE [GB]                         8.725E+02
  **** **** ******  **  PROGRAM ENDED AT                 2026-11-19 10:00:00.000
"""
    out = parse(text)
    assert out["energies_eV"] == [12.0361, 13.0327] and out["osc"] == [0.418, 0.0]
    assert out["bse_memory_GB"] == pytest.approx(872.5) and out["ok"]
