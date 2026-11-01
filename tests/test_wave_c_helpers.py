"""Tests for wave C helpers: joint distillation set, ORCA pilot parsing, S/T metric split (TASK-262)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("h5py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_joint_distill_dataset import qm9_entries  # noqa: E402
from parse_orca_pilot import parse_steom  # noqa: E402

ORCA_SNIPPET = """
 CIS-type guess
IROOT=  1:  0.100000 au     2.721 eV   21947.5 cm**-1
----------------------
STEOM-CCSD RESULTS (SINGLETS)
----------------------
IROOT=  1:  0.207000 au     5.633 eV   45431.2 cm**-1
IROOT=  2:  0.230000 au     6.259 eV   50479.1 cm**-1
IROOT=  3:  0.241000 au     6.558 eV   52893.3 cm**-1
IROOT=  4:  0.250000 au     6.803 eV   54868.6 cm**-1
IROOT=  5:  0.262000 au     7.129 eV   57502.3 cm**-1
"""


def test_parse_steom_takes_the_block_after_the_steom_header():
    assert parse_steom(ORCA_SNIPPET) == [5.633, 6.259, 6.558, 6.803, 7.129]
    assert parse_steom("no results here") is None
    assert parse_steom(ORCA_SNIPPET, nroots=6) is None


def test_qm9_entries_repeat_train_and_keep_val_test_once():
    entries = qm9_entries(np.array(["train", "val", "test", "train"]), true_repeat=2)
    assert [(r, s) for r, _, s in entries] == [(0, "train"), (0, "train"), (1, "val"), (2, "test"), (3, "train"), (3, "train")]
    assert {suffix for r, suffix, s in entries if s == "train"} == {"#r0", "#r1"}


def test_batch_mae_splits_singlet_and_triplet_aggregates():
    torch = pytest.importorskip("torch")
    from gnn_excited.metrics import batch_mae

    cols = ("S1_eV", "S2_eV", "T1_eV")
    pred = torch.tensor([[1.0, 2.0, 0.5]])
    target = torch.tensor([[1.1, 2.3, 0.0]])
    m = batch_mae(pred, target, cols)
    assert m["singlet_energy_eV_mae"] == pytest.approx(0.2)
    assert m["triplet_energy_eV_mae"] == pytest.approx(0.5)
    assert "singlet_energy_eV_mae" not in batch_mae(pred[:, :2], target[:, :2], cols[:2])
