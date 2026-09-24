"""Parser tests for scripts/extract_pubchemqc.py (numpy only)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from extract_pubchemqc import CM_TO_EV, parse_record


def _record(**overrides):
    base = {
        "atoms": {
            "coords": {"3d": [0.0, 0.0, 0.0, 1.4, 0.0, 0.0]},
            "elements": {"number": [6, 1]},
        },
        "properties": {"charge": 0, "multiplicity": 1},
        "transitions": {"electronic transitions": [23950.0 + 1000.0 * i for i in range(10)]},
        "inchikey": "ABC-DEF",
        "smiles": "CC",
        "formula": "CH4",
    }
    base.update(overrides)
    return base


def test_parse_record_converts_cm_to_ev():
    np = pytest.importorskip("numpy")
    parsed, reason = parse_record(42, _record())
    assert reason is None
    assert abs(parsed["S_eV"][0] - 23950.0 * CM_TO_EV) < 1e-9
    assert (np.diff(parsed["S_eV"]) > 0).all()
    assert parsed["numbers"].tolist() == [6, 1]


def test_parse_record_rejects_charged_and_multiplet():
    props = {"charge": 1, "multiplicity": 1}
    parsed, reason = parse_record(1, _record(properties=props))
    assert parsed is None and reason == "charged"
    props = {"charge": 0, "multiplicity": 2}
    parsed, reason = parse_record(1, _record(properties=props))
    assert parsed is None and reason == "non_singlet"


def test_parse_record_rejects_short_transitions():
    rec = _record()
    rec["transitions"] = {"electronic transitions": [1.0, 2.0]}
    parsed, reason = parse_record(1, rec)
    assert parsed is None and reason == "bad_transitions"
