from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gnn_excited.data.pyg_dataset import explicit_split
from gnn_excited.data.audit import (
    MoleculeIdentity,
    analyze_duplicate_target_differences,
    canonicalize_identity,
    assert_no_group_leakage,
    deduplicate_identities,
    grouped_assignments,
    parse_sha512_file,
    random_assignments,
    read_identity_csv,
    scaffold_keys,
    verify_sha512_files,
)


def _identity(key: str, smiles: str, inchi_key: str) -> MoleculeIdentity:
    return MoleculeIdentity(key, smiles, "InChI=1S/test", inchi_key, 2, 0, "test")


def test_deduplicate_identities_uses_composite_smiles_and_inchi() -> None:
    records = [
        _identity("A", "CC", "KEY-1"),
        _identity("B", "CC", "KEY-1"),
        _identity("C", "CC", "KEY-2"),
    ]
    selected, report = deduplicate_identities(records)
    assert [record.molecule_key for record in selected] == ["A", "C"]
    assert report["duplicate_identity_rows"] == 1
    assert report["smiles_with_multiple_inchi_keys"] == 1


def test_random_assignments_are_deterministic_and_complete() -> None:
    keys = [f"mol-{index}" for index in range(20)]
    first = random_assignments(keys, seed=17, fractions=(0.8, 0.1, 0.1))
    second = random_assignments(keys, seed=17, fractions=(0.8, 0.1, 0.1))
    assert first == second
    assert set(first) == set(keys)
    assert list(first.values()).count("train") == 16
    assert list(first.values()).count("val") == 2
    assert list(first.values()).count("test") == 2


def test_grouped_assignments_prevent_group_leakage() -> None:
    groups = {"a": "ring-1", "b": "ring-1", "c": "ring-2", "d": "ring-3", "e": "ring-3"}
    assignments = grouped_assignments(groups, seed=17, fractions=(0.6, 0.2, 0.2))
    assert_no_group_leakage(assignments, groups)
    assert assignments["a"] == assignments["b"]
    assert assignments["d"] == assignments["e"]


def test_scaffold_keys_use_topology_fallback_for_acyclic_molecules() -> None:
    pytest.importorskip("rdkit")
    scaffold, core = scaffold_keys("CCO")
    assert scaffold.startswith("acyclic:")
    assert core == scaffold
    cyclic_scaffold, cyclic_core = scaffold_keys("c1ccccc1O")
    assert cyclic_scaffold == "c1ccccc1"
    assert cyclic_core == "C1CCCCC1"


def test_identity_csv_uses_pybel_when_rdkit_columns_contain_sentinel(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    csv_path = tmp_path / "identities.csv"
    csv_path.write_text(
        "Index,HeavyAtomCount,RingNumber,CompoundType,Smiles_pybel,InchI_pybel,"
        "Smiles_rdkit,InchI_rdkit,Smiles_rdkit_can\n"
        "Aa1,3,0,test,CCO,InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3,1,1,1\n",
        encoding="utf-8",
    )
    records, warnings = read_identity_csv(csv_path)
    assert len(records) == 1
    assert records[0].canonical_smiles == "CCO"
    assert records[0].inchi_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert "Pybel fallback" in warnings[0]["error"]


def test_identity_keeps_valid_smiles_when_inchi_is_invalid() -> None:
    pytest.importorskip("rdkit")
    smiles, inchi, inchi_key = canonicalize_identity("CCO", "not-an-inchi")
    assert smiles == "CCO"
    assert inchi == ""
    assert inchi_key == ""


def test_duplicate_target_analysis_reports_large_disagreement(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "molecule_key,S1_eV,S1_f,status,error\n"
        "A,4.0,0.1,ok,\n"
        "B,4.1,0.2,ok,\n",
        encoding="utf-8",
    )
    output = tmp_path / "differences.csv"
    report = analyze_duplicate_target_differences(
        manifest,
        [{"molecule_key": "B", "representative_key": "A", "identity": "same"}],
        output,
    )
    assert report["pairs_compared"] == 1
    assert report["pairs_with_any_energy_difference_over_chemical_accuracy"] == 1
    assert output.exists()


def test_sha512_parser_and_verifier(tmp_path: Path) -> None:
    payload = b"qcdge"
    data_path = tmp_path / "sample.hdf5"
    data_path.write_bytes(payload)
    expected = hashlib.sha512(payload).hexdigest()
    checksum_path = tmp_path / "SHA512SUM"
    checksum_path.write_text(f"{expected}  sample.hdf5\n", encoding="utf-8")
    assert parse_sha512_file(checksum_path) == {"sample.hdf5": expected}
    result = verify_sha512_files(checksum_path)
    assert result["sample.hdf5"]["matches"] is True


def test_explicit_split_resolves_manifest_rows(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.csv"
    split_path.write_text(
        "molecule_key,random_split,scaffold_split\n"
        "A,train,test\n"
        "B,val,train\n"
        "C,test,val\n",
        encoding="utf-8",
    )
    rows = [{"molecule_key": key} for key in ("A", "B", "C")]
    assert explicit_split(rows, split_path, "random_split") == ([0], [1], [2])
    assert explicit_split(rows, split_path, "scaffold_split") == ([1], [2], [0])


def test_explicit_split_rejects_missing_manifest_keys(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.csv"
    split_path.write_text(
        "molecule_key,random_split\nA,train\nB,val\nC,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing 1 manifest keys"):
        explicit_split([{"molecule_key": key} for key in ("A", "B", "C", "D")], split_path, "random_split")
