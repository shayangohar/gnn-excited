"""Compact, streaming QM9GWBSE archive builder and PyG dataset.

The published data is a collection of ZIPs containing one tiny member per
molecule.  This module reads those members directly and writes one compact
HDF5, so callers never need to materialise hundreds of thousands of files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

SOURCE_DOI = "10.5281/zenodo.17902233"
EXPECTED_MD5 = {
    "xyz.zip": "02e0f8da2c3960305f7a8ef3c2b26821",
    "eexc_ss.zip": "df566f4e86e10c55f3e3e4511836b011",
    "eexc_st.zip": "09538dc984d3c01e823f6e49be145cf9",
    "osc_stren.zip": "8307098b9c103e5f5f91c94ccf4b06cb",
    "trans_dip_mom.zip": "d88a7471fa51fbac1084c39f6329e330",
    "README": "4165b3f296bddd592cb5a17cd3c4af12",
}
PROPERTY_ARCHIVES = {
    "eexc_ss": "eexc_ss.zip",
    "eexc_st": "eexc_st.zip",
    "osc_stren": "osc_stren.zip",
    "trans_dip_mom": "trans_dip_mom.zip",
}
NONCONVERGED_IDS = {"37992", "133858"}
STATE_COUNT = 5

_ID_RE = re.compile(r"(?:^|/)(?:mol[_-])?(\d+)(?:\.[^/]*)?$", re.IGNORECASE)
_ELEMENTS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


def _member_id(name: str) -> str | None:
    match = _ID_RE.search(name.replace("\\", "/"))
    return match.group(1) if match else None


def _index_zip(archive: zipfile.ZipFile, path: Path | str) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        molecule_id = _member_id(info.filename)
        if molecule_id is None:
            continue
        if molecule_id in result:
            raise ValueError(f"Duplicate molecule ID {molecule_id} in {path}")
        result[molecule_id] = info
    return result


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    return archive.read(info)


def _parse_xyz(raw: bytes, molecule_id: str) -> tuple[np.ndarray, np.ndarray]:
    lines = raw.decode("utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError(f"QM9 ID {molecule_id}: XYZ has fewer than two header lines")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"QM9 ID {molecule_id}: invalid XYZ atom count {lines[0]!r}") from exc
    if atom_count <= 0 or len(lines) < atom_count + 2:
        raise ValueError(f"QM9 ID {molecule_id}: XYZ atom count/content mismatch")
    labels: list[int] = []
    coords: list[list[float]] = []
    for line in lines[2 : atom_count + 2]:
        parts = line.split()
        if len(parts) < 4 or parts[0] not in _ELEMENTS:
            raise ValueError(f"QM9 ID {molecule_id}: invalid XYZ atom line {line!r}")
        labels.append(_ELEMENTS[parts[0]])
        coords.append([float(value) for value in parts[1:4]])
    positions = np.asarray(coords, dtype=np.float32)
    if not np.isfinite(positions).all():
        raise ValueError(f"QM9 ID {molecule_id}: non-finite XYZ coordinate")
    return np.asarray(labels, dtype=np.int16), positions


def _parse_values(raw: bytes, molecule_id: str, name: str, width: int = STATE_COUNT) -> np.ndarray:
    values: list[float] = []
    for token in raw.decode("utf-8-sig").split():
        values.append(float(token))
    if len(values) != width:
        raise ValueError(f"QM9 ID {molecule_id}: {name} expected {width} values, found {len(values)}")
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"QM9 ID {molecule_id}: {name} contains non-finite values")
    return result


def _parse_dipoles(raw: bytes, molecule_id: str) -> np.ndarray:
    values = _parse_values(raw, molecule_id, "trans_dip_mom", width=STATE_COUNT * 3)
    return values.reshape(STATE_COUNT, 3)


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_smiles(value: str) -> tuple[str, str]:
    if not str(value).strip():
        return "", "unmapped"
    try:
        from rdkit import Chem, RDLogger
    except ModuleNotFoundError:
        return "", "rdkit_unavailable"
    if not getattr(_canonical_smiles, "_rdkit_log_disabled", False):
        RDLogger.DisableLog("rdApp.error")
        _canonical_smiles._rdkit_log_disabled = True
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return "", "invalid_smiles"
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True), "ok"


def _identity_status_counts(statuses: Mapping[str, str]) -> dict[str, int]:
    return {
        "total": len(statuses),
        "canonical": sum(status == "ok" for status in statuses.values()),
        "invalid": sum(status == "invalid_smiles" for status in statuses.values()),
        "unmapped": sum(status == "unmapped" for status in statuses.values()),
        "rdkit_unavailable": sum(status == "rdkit_unavailable" for status in statuses.values()),
    }


def _load_identity_mapping(path: str | Path | None) -> tuple[dict[str, str], str]:
    if path is None:
        return {}, "not_requested"
    mapping: dict[str, str] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        id_column = next((name for name in ("qm9_id", "qm9_index") if name in fields), None)
        smiles_column = next((name for name in ("smiles", "canonical_smiles", "Smiles_rdkit_can") if name in fields), None)
        if id_column is None or smiles_column is None:
            raise ValueError("Identity mapping requires qm9_id/qm9_index and smiles/canonical_smiles columns")
        for row in reader:
            molecule_id = (row.get(id_column) or "").strip()
            smiles = (row.get(smiles_column) or "").strip()
            if molecule_id:
                mapping[molecule_id] = smiles
    return mapping, "provided"


def _load_qcdge_mapping(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        id_column = next((name for name in ("qcdge_molecule_key", "molecule_key", "Index") if name in fields), None)
        smiles_column = next((name for name in ("canonical_smiles", "smiles", "Smiles_rdkit_can") if name in fields), None)
        if id_column is None or smiles_column is None:
            raise ValueError("QCDGE identity mapping requires molecule_key/Index and smiles/canonical_smiles columns")
        for row in reader:
            molecule_key = (row.get(id_column) or "").strip()
            smiles = (row.get(smiles_column) or "").strip()
            if molecule_key:
                mapping[molecule_key] = smiles
    return mapping


def _split_assignments(
    ids: Sequence[str],
    seed: int,
    train_fraction: float,
    val_fraction: float,
    identity_keys: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Expected train_fraction > 0, val_fraction >= 0, and train + val < 1")
    groups: dict[str, list[str]] = {}
    for molecule_id in sorted(set(str(value) for value in ids)):
        identity = (identity_keys or {}).get(molecule_id) or f"qm9_id:{molecule_id}"
        groups.setdefault(str(identity), []).append(molecule_id)
    group_names = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(group_names)
    n_train = int(len(group_names) * train_fraction)
    n_val = int(len(group_names) * val_fraction)
    assignments: dict[str, str] = {}
    for split, names in (("train", group_names[:n_train]), ("val", group_names[n_train : n_train + n_val]), ("test", group_names[n_train + n_val :])):
        for name in names:
            assignments.update({molecule_id: split for molecule_id in groups[str(name)]})
    return assignments


def qcdge_pretrain_exclusion_rows(
    qm9_rows: Iterable[Mapping[str, str]],
    qcdge_canonical_map: Mapping[str, str],
) -> list[dict[str, str]]:
    """Map QM9 validation/test identities to explicit QCDGE molecule keys."""
    qm9_by_identity: dict[str, dict[str, set[str]]] = {}
    for row in qm9_rows:
        canonical = (row.get("canonical_smiles") or "").strip()
        split = (row.get("random_split") or "").strip().lower()
        qm9_id = (row.get("qm9_id") or "").strip()
        if canonical and qm9_id and split in {"val", "test"}:
            identity = qm9_by_identity.setdefault(canonical, {"qm9_ids": set(), "splits": set()})
            identity["qm9_ids"].add(qm9_id)
            identity["splits"].add(split)
    exclusions: list[dict[str, str]] = []
    for qcdge_key, canonical in sorted(qcdge_canonical_map.items()):
        identity = qm9_by_identity.get((canonical or "").strip())
        if not identity:
            continue
        exclusions.append({
            "qcdge_molecule_key": str(qcdge_key),
            "canonical_smiles": canonical,
            "qm9_ids": ";".join(sorted(identity["qm9_ids"])),
            "high_level_splits": ";".join(sorted(identity["splits"])),
        })
    return exclusions


def _validate_properties(
    molecule_id: str,
    ss: np.ndarray,
    st: np.ndarray,
    osc: np.ndarray,
    dip: np.ndarray,
) -> None:
    if ss.shape != (STATE_COUNT,) or st.shape != (STATE_COUNT,) or osc.shape != (STATE_COUNT,):
        raise ValueError(f"QM9 ID {molecule_id}: excitation/oscillator arrays must have shape (5,)")
    if dip.shape != (STATE_COUNT, 3):
        raise ValueError(f"QM9 ID {molecule_id}: transition dipoles must have shape (5, 3)")
    if np.any(osc < 0):
        raise ValueError(f"QM9 ID {molecule_id}: oscillator strengths must be non-negative")
    if not np.all(np.diff(ss) >= 0) or not np.all(np.diff(st) >= 0):
        raise ValueError(f"QM9 ID {molecule_id}: excitation energies must be ascending")


def build_qm9gwbse(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 17,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    identity_csv: str | Path | None = None,
    qcdge_identity_csv: str | Path | None = None,
    max_count: int | None = None,
    validate_md5: bool = True,
    nonconverged_ids: Iterable[str] = NONCONVERGED_IDS,
) -> dict[str, Any]:
    """Build HDF5, manifest, deterministic splits and provenance metadata."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = raw_dir / "xyz.zip"
    archive_paths = {name: raw_dir / filename for name, filename in PROPERTY_ARCHIVES.items()}
    missing_archives = [str(path) for path in (xyz_path, *archive_paths.values()) if not path.exists()]
    if missing_archives:
        raise FileNotFoundError(f"QM9GWBSE archive(s) missing: {', '.join(missing_archives)}")
    if validate_md5:
        checksum_paths = {"xyz.zip": xyz_path, **{path.name: path for path in archive_paths.values()}}
        readme_path = raw_dir / "README"
        if readme_path.exists():
            checksum_paths["README"] = readme_path
        mismatches = []
        for name, path in checksum_paths.items():
            actual = _archive_md5(path)
            expected = EXPECTED_MD5[name]
            if actual != expected:
                mismatches.append(f"{name}: expected {expected}, got {actual}")
        if mismatches:
            raise ValueError("QM9GWBSE MD5 validation failed: " + "; ".join(mismatches))

    archives = {"xyz": zipfile.ZipFile(xyz_path), **{name: zipfile.ZipFile(path) for name, path in archive_paths.items()}}
    indexes = {name: _index_zip(archive, path) for (name, archive), path in zip(archives.items(), (xyz_path, *archive_paths.values()))}
    ids = sorted(indexes["xyz"])
    if max_count is not None:
        ids = ids[: int(max_count)]
    excluded = {str(value) for value in nonconverged_ids}
    identity_map, identity_source = _load_identity_mapping(identity_csv)
    qcdge_map = _load_qcdge_mapping(qcdge_identity_csv) if qcdge_identity_csv is not None else {}
    canonical_map: dict[str, str] = {}
    qcdge_canonical_map: dict[str, str] = {}
    identity_status: dict[str, str] = {}
    for molecule_id, smiles in identity_map.items():
        canonical, status = _canonical_smiles(smiles)
        canonical_map[molecule_id] = canonical
        identity_status[molecule_id] = status
    qcdge_identity_status: dict[str, str] = {}
    for molecule_key, smiles in qcdge_map.items():
        canonical, status = _canonical_smiles(smiles)
        qcdge_canonical_map[molecule_key] = canonical
        qcdge_identity_status[molecule_key] = status

    hdf5_path = output_dir / "qm9gwbse.h5"
    manifest_path = output_dir / "manifest.csv"
    splits_path = output_dir / "splits.csv"
    audit_path = output_dir / "qcdge_audit.csv"
    pretrain_exclusions_path = output_dir / "qcdge_pretrain_exclusions.csv"
    if pretrain_exclusions_path.exists():
        pretrain_exclusions_path.unlink()
    provenance_path = output_dir / "provenance.json"
    fields = [
        "molecule_key", "qm9_id", "atom_count", "canonical_smiles", "identity_key",
        *[field for state in range(1, STATE_COUNT + 1) for field in (f"S{state}_eV", f"S{state}_f", f"log1p_S{state}_f", f"T{state}_eV")],
        "status", "error", "eexc_ss_unit", "eexc_st_unit", "osc_stren_unit", "trans_dip_mom_unit",
        "random_split",
    ]
    rows: list[dict[str, Any]] = []
    written_ids: list[str] = []
    with h5py.File(hdf5_path, "w") as handle, manifest_path.open("w", newline="", encoding="utf-8") as stream:
        handle.attrs.update({"schema_version": "qm9gwbse-v1", "source_doi": SOURCE_DOI, "length_units": "angstrom", "energy_units": "eV", "oscillator_units": "dimensionless", "dipole_units": "Debye"})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for molecule_id in ids:
            row: dict[str, Any] = {"molecule_key": molecule_id, "qm9_id": molecule_id, "canonical_smiles": canonical_map.get(molecule_id, ""), "identity_key": canonical_map.get(molecule_id) or f"qm9_id:{molecule_id}", "status": "ok", "error": "", "eexc_ss_unit": "eV", "eexc_st_unit": "eV", "osc_stren_unit": "dimensionless", "trans_dip_mom_unit": "Debye"}
            if molecule_id in excluded:
                row.update(status="excluded_nonconverged", error="listed non-converged in source metadata")
                writer.writerow(row)
                continue
            try:
                missing_properties = [name for name, index in indexes.items() if name != "xyz" and molecule_id not in index]
                if missing_properties:
                    raise ValueError(f"QM9 ID {molecule_id}: missing property member(s): {', '.join(missing_properties)}")
                labels, coords = _parse_xyz(_read_member(archives["xyz"], indexes["xyz"][molecule_id]), molecule_id)
                ss = _parse_values(_read_member(archives["eexc_ss"], indexes["eexc_ss"][molecule_id]), molecule_id, "eexc_ss")
                st = _parse_values(_read_member(archives["eexc_st"], indexes["eexc_st"][molecule_id]), molecule_id, "eexc_st")
                osc = _parse_values(_read_member(archives["osc_stren"], indexes["osc_stren"][molecule_id]), molecule_id, "osc_stren")
                dip = _parse_dipoles(_read_member(archives["trans_dip_mom"], indexes["trans_dip_mom"][molecule_id]), molecule_id)
                _validate_properties(molecule_id, ss, st, osc, dip)
                group = handle.create_group(molecule_id)
                group.attrs["qm9_id"] = molecule_id
                ground = group.create_group("ground_state")
                ground.create_dataset("labels", data=labels, compression="gzip")
                ground.create_dataset("coords", data=coords, compression="gzip")
                excited = group.create_group("excited_state")
                excited.create_dataset("eexc_ss_eV", data=ss, compression="gzip")
                excited.create_dataset("eexc_st_eV", data=st, compression="gzip")
                excited.create_dataset("osc_stren", data=osc, compression="gzip")
                excited.create_dataset("trans_dip_mom_D", data=dip, compression="gzip")
                row["atom_count"] = int(labels.shape[0])
                for state in range(STATE_COUNT):
                    row[f"S{state + 1}_eV"] = float(ss[state])
                    row[f"S{state + 1}_f"] = float(osc[state])
                    row[f"log1p_S{state + 1}_f"] = math.log1p(float(osc[state]))
                    row[f"T{state + 1}_eV"] = float(st[state])
                written_ids.append(molecule_id)
            except Exception as exc:  # Preserve a row-level audit trail.
                row.update(status="error", error=str(exc))
            writer.writerow(row)

    assignments = _split_assignments(written_ids, seed, train_fraction, val_fraction, {key: canonical_map.get(key, "") for key in written_ids})
    with manifest_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["random_split"] = assignments.get(row["qm9_id"], "")
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with splits_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["molecule_key", "qm9_id", "identity_key", "random_split"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows if row["status"] == "ok")

    exclusion_rows: list[dict[str, str]] = []
    audit: dict[str, Any] = {"status": "not_mapped", "reason": "No explicit QM9 identity mapping was supplied", "mapped_rows": 0}
    if identity_csv is not None and identity_map:
        if not any(canonical_map.values()):
            audit = {"status": "unavailable", "reason": "RDKit unavailable or supplied identities invalid", "mapped_rows": 0}
        else:
            # A QCDGE overlap audit is only meaningful with explicit molecule identities.
            qcdge_by_identity: dict[str, list[str]] = {}
            for qcdge_key, canonical in qcdge_canonical_map.items():
                if canonical:
                    qcdge_by_identity.setdefault(canonical, []).append(qcdge_key)
            overlap_count = sum(bool(qcdge_by_identity.get(row["canonical_smiles"])) for row in rows if row["status"] == "ok")
            audit = {"status": "mapped", "reason": "Explicit QM9 identity mapping supplied", "mapped_rows": sum(bool(canonical_map.get(key)) for key in written_ids), "qcdge_identity_rows": len(qcdge_map), "qcdge_overlap_rows": overlap_count}
            if qcdge_map:
                with audit_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["qm9_id", "canonical_smiles", "identity_status", "random_split", "qcdge_matches"])
                    writer.writeheader()
                    for row in rows:
                        if row["status"] == "ok" and row["canonical_smiles"]:
                            writer.writerow({"qm9_id": row["qm9_id"], "canonical_smiles": row["canonical_smiles"], "identity_status": identity_status.get(row["qm9_id"], "ok"), "random_split": row["random_split"], "qcdge_matches": ";".join(sorted(qcdge_by_identity.get(row["canonical_smiles"], [])))})
                exclusion_rows = qcdge_pretrain_exclusion_rows(rows, qcdge_canonical_map)
                if exclusion_rows:
                    with pretrain_exclusions_path.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["qcdge_molecule_key", "canonical_smiles", "qm9_ids", "high_level_splits"])
                        writer.writeheader()
                        writer.writerows(exclusion_rows)

    counts = {"candidates": len(ids), "ok": sum(row["status"] == "ok" for row in rows), "errors": sum(row["status"] == "error" for row in rows), "excluded_nonconverged": sum(row["status"] == "excluded_nonconverged" for row in rows), "missing_from_any_archive": sum(1 for key in ids if any(key not in indexes[name] for name in archive_paths))}
    archive_records = {"xyz.zip": {"sha256": _archive_sha256(xyz_path), "md5": _archive_md5(xyz_path), "expected_md5": EXPECTED_MD5["xyz.zip"], "members": len(indexes["xyz"])} }
    archive_records.update({path.name: {"sha256": _archive_sha256(path), "md5": _archive_md5(path), "expected_md5": EXPECTED_MD5[path.name], "members": len(indexes[name])} for name, path in archive_paths.items()})
    readme_path = raw_dir / "README"
    if readme_path.exists():
        archive_records["README"] = {"sha256": _archive_sha256(readme_path), "md5": _archive_md5(readme_path), "expected_md5": EXPECTED_MD5["README"], "matches_expected_md5": _archive_md5(readme_path) == EXPECTED_MD5["README"]}
    for record in archive_records.values():
        if "expected_md5" in record:
            record["matches_expected_md5"] = record["md5"] == record["expected_md5"]
    provenance = {"schema_version": "qm9gwbse-v1", "source_doi": SOURCE_DOI, "archives": archive_records, "counts": counts, "nonconverged_ids": sorted(excluded), "identity": {"source": identity_source, **audit, "qm9_identity_counts": _identity_status_counts(identity_status), "qcdge_identity_counts": _identity_status_counts(qcdge_identity_status)}, "qcdge_pretrain_exclusions": {"path": str(pretrain_exclusions_path) if pretrain_exclusions_path.exists() else None, "count": len(exclusion_rows)}, "split": {"seed": int(seed), "train_fraction": train_fraction, "val_fraction": val_fraction, "key": "qm9_id or explicit canonical identity"}}
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for archive in archives.values():
        archive.close()
    exclusion_count = provenance["qcdge_pretrain_exclusions"]["count"]
    return {"hdf5": str(hdf5_path), "manifest": str(manifest_path), "splits": str(splits_path), "provenance": str(provenance_path), "audit": str(audit_path) if audit_path.exists() else None, "qcdge_pretrain_exclusions_path": str(pretrain_exclusions_path) if pretrain_exclusions_path.exists() else None, "qcdge_pretrain_exclusions_count": exclusion_count, **counts}


def read_qm9gwbse_arrays(hdf5_path: str | Path, molecule_key: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as handle:
        group = handle[str(molecule_key)]["ground_state"]
        return np.asarray(group["labels"][()], dtype=np.int64), np.asarray(group["coords"][()], dtype=np.float32)


try:
    import torch
    from torch_geometric.data import Data, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - optional ML dependency.
    torch = None
    Data = None
    Dataset = object
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


class QM9GWBSEDataset(Dataset):
    """Lazy PyG view of the compact QM9GWBSE HDF5 and manifest."""

    def __init__(self, hdf5_path: str | Path, manifest_path: str | Path, molecule_keys: Sequence[str] | None = None, target_columns: Sequence[str] | None = None):
        if _PYG_IMPORT_ERROR is not None:
            raise ModuleNotFoundError("QM9GWBSEDataset requires torch and torch_geometric") from _PYG_IMPORT_ERROR
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.target_columns = tuple(target_columns or ("S1_eV", "log1p_S1_f"))
        allowed = None if molecule_keys is None else {str(value) for value in molecule_keys}
        with Path(manifest_path).open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = set(self.target_columns).difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Manifest is missing target columns: {sorted(missing)}")
            self.rows = [row for row in reader if row.get("status") == "ok" and (allowed is None or row["molecule_key"] in allowed)]
        if not self.rows:
            raise ValueError(f"No usable rows found in manifest {manifest_path}")
        self._handle_obj: h5py.File | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle_obj"] = None
        return state

    def __del__(self):
        if getattr(self, "_handle_obj", None) is not None:
            self._handle_obj.close()

    def _handle(self) -> h5py.File:
        if self._handle_obj is None:
            self._handle_obj = h5py.File(self.hdf5_path, "r")
        return self._handle_obj

    def len(self) -> int:
        return len(self.rows)

    def get(self, idx: int):
        row = self.rows[idx]
        molecule = self._handle()[row["molecule_key"]]
        ground = molecule["ground_state"]
        z = torch.as_tensor(np.asarray(ground["labels"][()], dtype=np.int64), dtype=torch.long).view(-1)
        pos = torch.as_tensor(np.asarray(ground["coords"][()], dtype=np.float32), dtype=torch.float32)
        transition_dipole = torch.as_tensor(
            np.asarray(molecule["excited_state"]["trans_dip_mom_D"][()], dtype=np.float32),
            dtype=torch.float32,
        )
        y = torch.tensor([[float(row[column]) for column in self.target_columns]], dtype=torch.float32)
        return Data(
            z=z,
            pos=pos,
            y=y,
            transition_dipole=transition_dipole,
            molecule_key=row["molecule_key"],
            qm9_id=row["qm9_id"],
        )
