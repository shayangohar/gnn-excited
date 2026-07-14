from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gnn_excited.data.audit import (
    MoleculeIdentity,
    assert_no_group_leakage,
    deduplicate_identities,
    grouped_assignments,
    parse_sha512_file,
    random_assignments,
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
