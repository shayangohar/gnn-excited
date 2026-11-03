"""Tests for wave C helpers: joint distillation set, ORCA pilot parsing, S/T metric split (TASK-262)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("h5py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_joint_distill_dataset import qm9_entries  # noqa: E402
from parse_orca_pilot import failure_cause, is_reliable, parse_steom, parse_steom_roots  # noqa: E402

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
    assert not is_reliable(parse_steom_roots(ORCA_SNIPPET))  # no active character printed


# Shape of a real ORCA 6.1.1 STEOM block (QM9 581): duplicated roots that ORCA itself removes,
# and roots below 98% active character.
ORCA_DUPLICATES = """
STEOM-CCSD RESULTS (SINGLETS)
IROOT=  1:  0.139512 au     3.796 eV   30619.4 cm**-1 <S**2> =   0.000000
Percentage Active Character     97.90
IROOT=  2:  0.139722 au     3.802 eV   30665.4 cm**-1 <S**2> =   0.000000
Percentage Active Character     97.93
IROOT=  3:  0.283976 au     7.727 eV   62325.6 cm**-1 <S**2> =   0.000000
Percentage Active Character     99.18
IROOT=  4:  0.284020 au     7.729 eV   62335.3 cm**-1 <S**2> =   0.000000
Percentage Active Character     99.16
IROOT=  5:  0.306685 au     8.345 eV   67309.6 cm**-1 <S**2> =   0.000000
Percentage Active Character     86.93
PAIRWISE REDUNDANCY FOUND (L1=1.999979E+00,L2=2.075650E-05) AND REMOVED FOR IROOT=1 and IROOT=2!
PAIRWISE REDUNDANCY FOUND (L1=1.998889E+00,L2=1.111016E-03) AND REMOVED FOR IROOT=3 and IROOT=4!
IROOT=  1:     3.796    1.099490    1.180526    0.286923    4.164862
"""


def test_parse_steom_drops_orca_duplicates_and_flags_low_active_character():
    roots = parse_steom_roots(ORCA_DUPLICATES)
    assert [r["redundant"] for r in roots] == [False, True, False, True, False]
    assert [r["active"] for r in roots] == [97.90, 97.93, 99.18, 99.16, 86.93]
    assert parse_steom(ORCA_DUPLICATES, nroots=3) == [3.796, 7.727, 8.345]
    assert parse_steom(ORCA_DUPLICATES) is None
    assert not is_reliable(roots, nroots=3)
    clean = ORCA_DUPLICATES.replace("97.90", "98.50").replace("86.93", "99.00")
    assert is_reliable(parse_steom_roots(clean), nroots=3)
    assert failure_cause("... Solution of the IP-EOM-DLPNO-CCSD equations failed ...") == "IP-EOM"
    assert failure_cause("partial output") == "incomplete (killed or timed out)"


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
