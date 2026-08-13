from __future__ import annotations

import csv
import math
import zipfile
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")
from gnn_excited.data.qm9gwbse import build_qm9gwbse, qcdge_pretrain_exclusion_rows, read_qm9gwbse_arrays


def _write_archive(path: Path, values: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for molecule_id, content in values.items():
            archive.writestr(f"mol_{molecule_id}.dat", content)


def test_build_qm9gwbse_streams_zip_members_and_validates_targets(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    raw.mkdir()
    xyz = {
        "1": "2\nwater\nO 0 0 0\nH 0 0 1\n",
        "2": "2\nwater\nO 0 0 0\nH 0 1 0\n",
    }
    _write_archive(raw / "xyz.zip", xyz)
    _write_archive(raw / "eexc_ss.zip", {key: "1 2 3 4 5\n" for key in xyz})
    _write_archive(raw / "eexc_st.zip", {key: "2 3 4 5 6\n" for key in xyz})
    _write_archive(raw / "osc_stren.zip", {key: "0 0.1 0.2 0.3 0.4\n" for key in xyz})
    _write_archive(raw / "trans_dip_mom.zip", {key: "0 0 0\n" * 5 for key in xyz})

    result = build_qm9gwbse(raw, out, seed=9, train_fraction=0.5, val_fraction=0.0, validate_md5=False, nonconverged_ids=())
    assert result["ok"] == 2
    labels, coords = read_qm9gwbse_arrays(result["hdf5"], "1")
    assert labels.tolist() == [8, 1]
    assert coords.shape == (2, 3)
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["S1_eV"] == "1.0"
    assert float(rows[0]["log1p_S2_f"]) == pytest.approx(math.log1p(0.1))
    assert {row["random_split"] for row in rows} == {"train", "test"}
    assert (out / "qcdge_audit.csv").exists() is False


def test_qcdge_pretrain_exclusions_only_include_qm9_val_test_identities() -> None:
    rows = [
        {"qm9_id": "1", "canonical_smiles": "C", "random_split": "train"},
        {"qm9_id": "2", "canonical_smiles": "O", "random_split": "val"},
        {"qm9_id": "3", "canonical_smiles": "N", "random_split": "test"},
    ]
    exclusions = qcdge_pretrain_exclusion_rows(
        rows, {"qcdge-train": "C", "qcdge-val": "O", "qcdge-test": "N", "qcdge-unmatched": "CC"}
    )
    assert [row["qcdge_molecule_key"] for row in exclusions] == ["qcdge-test", "qcdge-val"]
    assert {row["high_level_splits"] for row in exclusions} == {"val", "test"}
