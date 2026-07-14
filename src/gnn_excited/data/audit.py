from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

import h5py
import numpy as np


@dataclass(frozen=True)
class MoleculeIdentity:
    molecule_key: str
    canonical_smiles: str
    canonical_inchi: str
    inchi_key: str
    heavy_atom_count: int | None
    ring_count: int | None
    compound_type: str

    @property
    def identity_key(self) -> tuple[str, ...]:
        if self.canonical_smiles and self.inchi_key:
            return ("smiles+inchi", self.canonical_smiles, self.inchi_key)
        if self.canonical_smiles:
            return ("smiles", self.canonical_smiles)
        if self.inchi_key:
            return ("inchi", self.inchi_key)
        if self.canonical_inchi:
            return ("inchi-text", self.canonical_inchi)
        return ("molecule-key", self.molecule_key)


def _require_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "QCDGE identity auditing requires RDKit. Install the project's chem extra: "
            "python -m pip install -e '.[chem]'"
        ) from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem, MurckoScaffold


def _optional_int(value: str | None) -> int | None:
    text = (value or "").strip()
    return int(text) if text else None


def _normalise_inchi(value: str | None) -> str:
    return "".join((value or "").strip().split())


def _identity_candidates(values: Iterable[str | None]) -> list[str]:
    candidates: list[str] = []
    for value in values:
        text = (value or "").strip()
        if not text or text.lower() in {"1", "nan", "none", "null"} or text in candidates:
            continue
        candidates.append(text)
    return candidates


def _canonicalize_identity_candidates(
    smiles_values: Iterable[str | None], inchi_values: Iterable[str | None]
) -> tuple[str, str, str, list[str]]:
    Chem, _ = _require_rdkit()
    warnings: list[str] = []
    canonical_smiles = ""
    for smiles_text in _identity_candidates(smiles_values):
        smiles_mol = Chem.MolFromSmiles(smiles_text)
        if smiles_mol is None:
            warnings.append(f"RDKit could not parse SMILES {smiles_text!r}")
            continue
        canonical_smiles = Chem.MolToSmiles(smiles_mol, canonical=True, isomericSmiles=True)
        break

    canonical_inchi = ""
    inchi_key = ""
    for inchi_value in _identity_candidates(inchi_values):
        inchi_text = _normalise_inchi(inchi_value)
        inchi_mol = Chem.MolFromInchi(inchi_text)
        if inchi_mol is None:
            warnings.append(f"RDKit could not parse InChI {inchi_text!r}")
            continue
        canonical_inchi = inchi_text
        inchi_key = Chem.MolToInchiKey(inchi_mol)
        if not canonical_smiles:
            canonical_smiles = Chem.MolToSmiles(inchi_mol, canonical=True, isomericSmiles=True)
        break

    if not canonical_smiles and not inchi_key:
        detail = "; ".join(warnings) or "all identity fields are empty or sentinel values"
        raise ValueError(f"No usable SMILES or InChI identity: {detail}")
    return canonical_smiles, canonical_inchi, inchi_key, warnings


def canonicalize_identity(smiles: str | None, inchi: str | None) -> tuple[str, str, str]:
    """Return the usable canonical identity even when one representation is invalid."""
    canonical_smiles, canonical_inchi, inchi_key, _ = _canonicalize_identity_candidates(
        [smiles], [inchi]
    )
    return canonical_smiles, canonical_inchi, inchi_key


def read_identity_csv(path: str | Path) -> tuple[list[MoleculeIdentity], list[dict[str, str]]]:
    """Read and validate the identity fields published in final_all.csv."""
    records: list[MoleculeIdentity] = []
    errors: list[dict[str, str]] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"Index", "Smiles_rdkit_can", "InchI_rdkit"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Identity CSV is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            molecule_key = (row.get("Index") or "").strip()
            if not molecule_key:
                errors.append({"row": str(row_number), "molecule_key": "", "error": "missing Index"})
                continue
            try:
                smiles, inchi, inchi_key, row_warnings = _canonicalize_identity_candidates(
                    [row.get("Smiles_rdkit_can"), row.get("Smiles_rdkit"), row.get("Smiles_pybel")],
                    [row.get("InchI_rdkit"), row.get("InchI_pybel")],
                )
                records.append(
                    MoleculeIdentity(
                        molecule_key=molecule_key,
                        canonical_smiles=smiles,
                        canonical_inchi=inchi,
                        inchi_key=inchi_key,
                        heavy_atom_count=_optional_int(row.get("HeavyAtomCount")),
                        ring_count=_optional_int(row.get("RingNumber")),
                        compound_type=(row.get("CompoundType") or "").strip(),
                    )
                )
                if row_warnings or (row.get("Smiles_rdkit_can") or "").strip() == "1":
                    warning_parts = list(row_warnings)
                    if (row.get("Smiles_rdkit_can") or "").strip() == "1":
                        warning_parts.insert(0, "used Pybel fallback because RDKit identity columns contain sentinel 1")
                    errors.append(
                        {
                            "row": str(row_number),
                            "molecule_key": molecule_key,
                            "error": "; ".join(warning_parts),
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - audit must preserve every failure.
                records.append(
                    MoleculeIdentity(
                        molecule_key=molecule_key,
                        canonical_smiles="",
                        canonical_inchi="",
                        inchi_key="",
                        heavy_atom_count=_optional_int(row.get("HeavyAtomCount")),
                        ring_count=_optional_int(row.get("RingNumber")),
                        compound_type=(row.get("CompoundType") or "").strip(),
                    )
                )
                errors.append({"row": str(row_number), "molecule_key": molecule_key, "error": str(exc)})
            if (row_number - 1) % 10000 == 0:
                print(
                    f"canonicalized {row_number - 1} identity rows "
                    f"({len(records)} ok, {len(errors)} errors)",
                    flush=True,
                )
    return records, errors


def deduplicate_identities(
    records: Sequence[MoleculeIdentity],
) -> tuple[list[MoleculeIdentity], dict[str, object]]:
    """Deduplicate on the composite canonical SMILES/InChI identity."""
    seen_ids: set[str] = set()
    seen_identity: dict[tuple[str, ...], str] = {}
    selected: list[MoleculeIdentity] = []
    duplicate_ids: list[str] = []
    duplicate_identity_rows: list[dict[str, str]] = []
    smiles_to_inchi: dict[str, set[str]] = defaultdict(set)
    inchi_to_smiles: dict[str, set[str]] = defaultdict(set)

    for record in records:
        if record.molecule_key in seen_ids:
            duplicate_ids.append(record.molecule_key)
            continue
        seen_ids.add(record.molecule_key)
        if record.canonical_smiles and record.inchi_key:
            smiles_to_inchi[record.canonical_smiles].add(record.inchi_key)
            inchi_to_smiles[record.inchi_key].add(record.canonical_smiles)
        representative = seen_identity.get(record.identity_key)
        if representative is not None:
            duplicate_identity_rows.append(
                {
                    "molecule_key": record.molecule_key,
                    "representative_key": representative,
                    "identity": "|".join(record.identity_key),
                }
            )
            continue
        seen_identity[record.identity_key] = record.molecule_key
        selected.append(record)

    report: dict[str, object] = {
        "input_rows": len(records),
        "unique_molecule_ids": len(seen_ids),
        "selected_unique_identities": len(selected),
        "duplicate_id_rows": len(duplicate_ids),
        "duplicate_identity_rows": len(duplicate_identity_rows),
        "smiles_with_multiple_inchi_keys": sum(len(values) > 1 for values in smiles_to_inchi.values()),
        "inchi_keys_with_multiple_smiles": sum(len(values) > 1 for values in inchi_to_smiles.values()),
        "duplicate_ids": duplicate_ids,
        "duplicate_identities": duplicate_identity_rows,
    }
    return selected, report


def scaffold_keys(canonical_smiles: str) -> tuple[str, str]:
    """Return exact Murcko and generic-core keys, with an acyclic topology fallback."""
    Chem, MurckoScaffold = _require_rdkit()
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse canonical SMILES {canonical_smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        generic = MurckoScaffold.MakeScaffoldGeneric(molecule)
        fallback = Chem.MolToSmiles(generic, canonical=True, isomericSmiles=False)
        return f"acyclic:{fallback}", f"acyclic:{fallback}"
    exact = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    core = Chem.MolToSmiles(generic, canonical=True, isomericSmiles=False)
    return exact, core


def _decode_attribute(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _attribute_value(attributes: Mapping[str, object], *names: str) -> str:
    normalized = {"".join(character for character in key.lower() if character.isalnum()): value for key, value in attributes.items()}
    for name in names:
        value = normalized.get("".join(character for character in name.lower() if character.isalnum()))
        if value is not None:
            return _decode_attribute(value)
    return ""


def inspect_hdf5_only_identities(
    hdf5_path: str | Path,
    hdf5_only_ids: Sequence[str],
    official_records: Sequence[MoleculeIdentity],
    output_path: str | Path,
) -> tuple[dict[str, object], list[MoleculeIdentity]]:
    """Classify HDF5-only groups against published identities using group attributes."""
    composite_to_ids: dict[tuple[str, ...], list[str]] = defaultdict(list)
    smiles_to_ids: dict[str, list[str]] = defaultdict(list)
    inchi_to_ids: dict[str, list[str]] = defaultdict(list)
    for record in official_records:
        composite_to_ids[record.identity_key].append(record.molecule_key)
        if record.canonical_smiles:
            smiles_to_ids[record.canonical_smiles].append(record.molecule_key)
        if record.inchi_key:
            inchi_to_ids[record.inchi_key].append(record.molecule_key)

    rows: list[dict[str, str]] = []
    identities: Counter[tuple[str, ...]] = Counter()
    match_counts: Counter[str] = Counter()
    parsed_records: list[MoleculeIdentity] = []
    with h5py.File(hdf5_path, "r") as handle:
        for index, molecule_key in enumerate(hdf5_only_ids, start=1):
            attributes = dict(handle[molecule_key].attrs.items())
            smiles_values = [
                _attribute_value(attributes, "Smiles_rdkit_can"),
                _attribute_value(attributes, "Smiles_rdkit"),
                _attribute_value(attributes, "canonical_smiles"),
                _attribute_value(attributes, "Smiles_pybel"),
            ]
            inchi_values = [
                _attribute_value(attributes, "InchI_rdkit"),
                _attribute_value(attributes, "InChI"),
                _attribute_value(attributes, "InchI_pybel"),
            ]
            row = {
                "molecule_key": molecule_key,
                "canonical_smiles": "",
                "canonical_inchi": "",
                "inchi_key": "",
                "match_type": "",
                "matched_official_ids": "",
                "error": "",
            }
            try:
                smiles, inchi, inchi_key, identity_warnings = _canonicalize_identity_candidates(
                    smiles_values, inchi_values
                )
                identity = MoleculeIdentity(molecule_key, smiles, inchi, inchi_key, None, None, "")
                parsed_records.append(identity)
                identities[identity.identity_key] += 1
                exact_ids = composite_to_ids.get(identity.identity_key, [])
                smiles_ids = smiles_to_ids.get(smiles, [])
                inchi_ids = inchi_to_ids.get(inchi_key, [])
                if exact_ids:
                    match_type, matched = "exact_smiles+inchi", exact_ids
                elif smiles_ids and inchi_ids:
                    match_type, matched = "conflicting_smiles_inchi", sorted(set(smiles_ids + inchi_ids))
                elif smiles_ids:
                    match_type, matched = "smiles_only", smiles_ids
                elif inchi_ids:
                    match_type, matched = "inchi_only", inchi_ids
                else:
                    match_type, matched = "unmatched", []
                row.update(
                    {
                        "canonical_smiles": smiles,
                        "canonical_inchi": inchi,
                        "inchi_key": inchi_key,
                        "match_type": match_type,
                        "matched_official_ids": ";".join(matched),
                        "error": "; ".join(identity_warnings),
                    }
                )
                match_counts[match_type] += 1
            except Exception as exc:  # noqa: BLE001 - audit must preserve every failure.
                row["match_type"] = "attribute_error"
                row["error"] = str(exc)
                match_counts["attribute_error"] += 1
            rows.append(row)
            if index % 5000 == 0:
                print(f"inspected attributes for {index} HDF5-only groups", flush=True)

    _write_csv(
        Path(output_path),
        [
            "molecule_key",
            "canonical_smiles",
            "canonical_inchi",
            "inchi_key",
            "match_type",
            "matched_official_ids",
            "error",
        ],
        rows,
    )
    return (
        {
            "match_counts": dict(match_counts),
            "unique_hdf5_only_identities": len(identities),
            "duplicate_hdf5_only_rows": sum(count - 1 for count in identities.values() if count > 1),
        },
        parsed_records,
    )


def random_assignments(keys: Sequence[str], seed: int, fractions: Sequence[float]) -> dict[str, str]:
    _validate_fractions(fractions)
    shuffled = np.asarray(list(keys), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    boundaries = _split_boundaries(len(shuffled), fractions)
    assignments: dict[str, str] = {}
    for split, start, end in zip(("train", "val", "test"), (0, *boundaries[:-1]), boundaries):
        assignments.update((str(key), split) for key in shuffled[start:end])
    return assignments


def grouped_assignments(
    key_to_group: Mapping[str, str], seed: int, fractions: Sequence[float]
) -> dict[str, str]:
    """Greedily balance whole groups across train/validation/test."""
    _validate_fractions(fractions)
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, group in key_to_group.items():
        grouped[group].append(key)

    def tie_breaker(group: str) -> str:
        return hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()

    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), tie_breaker(item[0])))
    split_names = ("train", "val", "test")
    targets = dict(zip(split_names, _split_sizes(len(key_to_group), fractions)))
    counts = dict.fromkeys(split_names, 0)
    assignments: dict[str, str] = {}
    for _, members in ordered:
        split = max(
            split_names,
            key=lambda name: ((targets[name] - counts[name]) / max(targets[name], 1), -counts[name]),
        )
        for key in members:
            assignments[key] = split
        counts[split] += len(members)
    return assignments


def _validate_fractions(fractions: Sequence[float]) -> None:
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("Expected three positive train/validation/test fractions")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train/validation/test fractions must sum to 1")


def _split_sizes(total: int, fractions: Sequence[float]) -> tuple[int, int, int]:
    n_train = int(total * fractions[0])
    n_val = int(total * fractions[1])
    return n_train, n_val, total - n_train - n_val


def _split_boundaries(total: int, fractions: Sequence[float]) -> tuple[int, int, int]:
    sizes = _split_sizes(total, fractions)
    return sizes[0], sizes[0] + sizes[1], total


def assert_no_group_leakage(assignments: Mapping[str, str], key_to_group: Mapping[str, str]) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for key, split in assignments.items():
        group_splits[key_to_group[key]].add(split)
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise AssertionError(f"Found {len(leaking)} groups spanning multiple splits")


def parse_sha512_file(path: str | Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2 or len(parts[0]) != 128:
                raise ValueError(f"Invalid SHA-512 record on line {line_number}: {stripped!r}")
            expected[parts[1].lstrip("*")] = parts[0].lower()
    return expected


def sha512_stream(stream: BinaryIO, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha512()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def verify_sha512_files(checksum_path: str | Path) -> dict[str, dict[str, object]]:
    checksum_path = Path(checksum_path)
    results: dict[str, dict[str, object]] = {}
    for name, expected in parse_sha512_file(checksum_path).items():
        path = checksum_path.parent / name
        if not path.exists():
            results[name] = {"expected": expected, "actual": None, "matches": False, "error": "missing file"}
            continue
        with path.open("rb") as stream:
            actual = sha512_stream(stream)
        results[name] = {
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
            "size_bytes": path.stat().st_size,
        }
    return results


def verify_gzip_decompressed_sha512(gzip_path: str | Path, expected: str) -> dict[str, object]:
    path = Path(gzip_path)
    try:
        with gzip.open(path, "rb") as stream:
            actual = sha512_stream(stream)
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        return {"expected": expected, "actual": None, "matches": False, "error": str(exc)}
    return {
        "expected": expected,
        "actual": actual,
        "matches": actual == expected,
        "compressed_size_bytes": path.stat().st_size,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _filter_manifest(
    source_path: Path, output_path: Path, selected_ids: set[str]
) -> tuple[set[str], set[str], dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    usable: set[str] = set()
    counts = {"source_rows": 0, "selected_rows": 0, "selected_ok_rows": 0, "selected_error_rows": 0}
    with source_path.open("r", newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "molecule_key" not in reader.fieldnames:
            raise ValueError(f"Manifest {source_path} has no molecule_key column")
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            counts["source_rows"] += 1
            key = row["molecule_key"]
            if key not in selected_ids:
                continue
            found.add(key)
            counts["selected_rows"] += 1
            if row.get("status") == "ok":
                counts["selected_ok_rows"] += 1
                usable.add(key)
            else:
                counts["selected_error_rows"] += 1
            writer.writerow(row)
    return found, usable, counts


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def analyze_duplicate_target_differences(
    manifest_path: str | Path,
    duplicate_rows: Sequence[Mapping[str, str]],
    output_path: str | Path,
) -> dict[str, object]:
    """Compare all available state targets for canonical-identity duplicate pairs."""
    pair_keys = {
        key
        for pair in duplicate_rows
        for key in (pair["molecule_key"], pair["representative_key"])
    }
    rows_by_key: dict[str, dict[str, str]] = {}
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        energy_columns = [column for column in fieldnames if re.fullmatch(r"[ST]\d+_eV", column)]
        oscillator_columns = [column for column in fieldnames if re.fullmatch(r"[ST]\d+_f", column)]
        target_columns = energy_columns + oscillator_columns
        for row in reader:
            if row.get("status") == "ok" and row.get("molecule_key") in pair_keys:
                rows_by_key[row["molecule_key"]] = row

    output_rows: list[dict[str, object]] = []
    max_energy_differences: list[float] = []
    max_oscillator_differences: list[float] = []
    missing_pairs = 0
    for pair in duplicate_rows:
        duplicate = rows_by_key.get(pair["molecule_key"])
        representative = rows_by_key.get(pair["representative_key"])
        if duplicate is None or representative is None:
            missing_pairs += 1
            continue
        row: dict[str, object] = {
            "molecule_key": pair["molecule_key"],
            "representative_key": pair["representative_key"],
            "identity": pair["identity"],
        }
        energy_differences: list[float] = []
        oscillator_differences: list[float] = []
        for column in target_columns:
            difference = abs(float(duplicate[column]) - float(representative[column]))
            row[f"{column}_abs_diff"] = difference
            if column in energy_columns:
                energy_differences.append(difference)
            else:
                oscillator_differences.append(difference)
        max_energy = max(energy_differences, default=0.0)
        max_oscillator = max(oscillator_differences, default=0.0)
        row["max_energy_abs_diff_eV"] = max_energy
        row["max_oscillator_abs_diff"] = max_oscillator
        output_rows.append(row)
        max_energy_differences.append(max_energy)
        max_oscillator_differences.append(max_oscillator)

    output_fields = ["molecule_key", "representative_key", "identity"]
    output_fields.extend(f"{column}_abs_diff" for column in target_columns)
    output_fields.extend(["max_energy_abs_diff_eV", "max_oscillator_abs_diff"])
    _write_csv(Path(output_path), output_fields, output_rows)
    return {
        "pairs_requested": len(duplicate_rows),
        "pairs_compared": len(output_rows),
        "pairs_missing_targets": missing_pairs,
        "pairs_with_any_energy_difference_over_chemical_accuracy": sum(
            value > 0.043 for value in max_energy_differences
        ),
        "pairs_with_numerically_identical_targets": sum(
            energy <= 1e-6 and oscillator <= 1e-8
            for energy, oscillator in zip(max_energy_differences, max_oscillator_differences)
        ),
        "max_energy_abs_diff_eV": _numeric_summary(max_energy_differences),
        "max_oscillator_abs_diff": _numeric_summary(max_oscillator_differences),
    }


def analyze_legacy_random_split_leakage(
    manifest_path: str | Path,
    identity_records: Sequence[MoleculeIdentity],
    output_path: str | Path,
    seed: int,
    fractions: Sequence[float],
) -> dict[str, object]:
    """Replay the old row-random split and count strict identity groups crossing partitions."""
    manifest_keys: list[str] = []
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("status") == "ok":
                manifest_keys.append(row["molecule_key"])
    assignments = random_assignments(manifest_keys, seed, fractions)
    identity_by_key = {
        record.molecule_key: record.identity_key
        for record in identity_records
        if record.identity_key[0] != "molecule-key"
    }
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for key in manifest_keys:
        identity = identity_by_key.get(key)
        if identity is not None:
            groups[identity].append(key)

    duplicate_groups = {identity: keys for identity, keys in groups.items() if len(keys) > 1}
    rows: list[dict[str, object]] = []
    leaking_groups = 0
    leaking_rows = 0
    train_test_groups = 0
    test_rows_exposed_to_train = 0
    exposed_by_subset: Counter[str] = Counter()
    for identity, keys in duplicate_groups.items():
        splits = {assignments[key] for key in keys}
        leaking = len(splits) > 1
        train_test = "train" in splits and "test" in splits
        if leaking:
            leaking_groups += 1
            leaking_rows += len(keys)
        if train_test:
            train_test_groups += 1
            exposed_test_keys = [key for key in keys if assignments[key] == "test"]
            test_rows_exposed_to_train += len(exposed_test_keys)
            exposed_by_subset.update(key[:2] for key in exposed_test_keys)
        rows.append(
            {
                "identity": "|".join(identity),
                "member_count": len(keys),
                "splits": ";".join(sorted(splits)),
                "leaking": str(leaking).lower(),
                "train_test_leak": str(train_test).lower(),
                "molecule_keys": ";".join(keys),
            }
        )
    _write_csv(
        Path(output_path),
        ["identity", "member_count", "splits", "leaking", "train_test_leak", "molecule_keys"],
        rows,
    )
    return {
        "policy": "strict canonical SMILES+InChI identity; attribute-unparseable rows are excluded",
        "seed": seed,
        "manifest_ok_rows": len(manifest_keys),
        "rows_with_parsed_identity": sum(key in identity_by_key for key in manifest_keys),
        "duplicate_identity_groups": len(duplicate_groups),
        "rows_in_duplicate_identity_groups": sum(len(keys) for keys in duplicate_groups.values()),
        "identity_groups_crossing_splits": leaking_groups,
        "rows_in_cross_split_identity_groups": leaking_rows,
        "identity_groups_crossing_train_test": train_test_groups,
        "test_rows_with_identity_in_train": test_rows_exposed_to_train,
        "test_rows_with_identity_in_train_by_subset": dict(sorted(exposed_by_subset.items())),
    }


def run_qcdge_audit(
    csv_path: str | Path,
    hdf5_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    deduplicated_manifest_path: str | Path,
    checksum_path: str | Path | None = None,
    compressed_hdf5_path: str | Path | None = None,
    seed: int = 17,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[str, object]:
    """Audit full QCDGE identities, integrity, manifests, and leakage-safe splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, identity_warnings = read_identity_csv(csv_path)
    unparsed_identity_ids = {
        record.molecule_key for record in records if record.identity_key[0] == "molecule-key"
    }
    identity_errors = [
        warning for warning in identity_warnings if warning["molecule_key"] in unparsed_identity_ids
    ]
    selected, dedup_report = deduplicate_identities(records)
    selected_ids = {record.molecule_key for record in selected}
    duplicate_rows = dedup_report["duplicate_identities"]

    with h5py.File(hdf5_path, "r") as handle:
        hdf5_ids = set(map(str, handle.keys()))
    csv_ids = {record.molecule_key for record in records}
    csv_missing_hdf5 = sorted(csv_ids - hdf5_ids)
    hdf5_only = sorted(hdf5_ids - csv_ids)
    hdf5_only_report, hdf5_only_records = inspect_hdf5_only_identities(
        hdf5_path,
        hdf5_only,
        records,
        output_dir / "hdf5_only_identity.csv",
    )

    found_manifest_ids, usable_manifest_ids, manifest_report = _filter_manifest(
        Path(source_manifest_path), Path(deduplicated_manifest_path), selected_ids & hdf5_ids
    )
    eligible = [record for record in selected if record.molecule_key in usable_manifest_ids]
    duplicate_target_report = analyze_duplicate_target_differences(
        source_manifest_path,
        duplicate_rows,
        output_dir / "duplicate_target_differences.csv",
    )
    legacy_leakage_report = analyze_legacy_random_split_leakage(
        source_manifest_path,
        [*records, *hdf5_only_records],
        output_dir / "legacy_random_split_identity_groups.csv",
        seed,
        fractions,
    )

    scaffold_by_key: dict[str, str] = {}
    core_by_key: dict[str, str] = {}
    scaffold_errors: list[dict[str, str]] = []
    for index, record in enumerate(eligible, start=1):
        try:
            scaffold, core = scaffold_keys(record.canonical_smiles)
            scaffold_by_key[record.molecule_key] = scaffold
            core_by_key[record.molecule_key] = core
        except Exception as exc:  # noqa: BLE001 - audit must preserve every failure.
            scaffold_errors.append({"molecule_key": record.molecule_key, "error": str(exc)})
            fallback = f"unparsed:{record.molecule_key}"
            scaffold_by_key[record.molecule_key] = fallback
            core_by_key[record.molecule_key] = fallback
        if index % 10000 == 0:
            print(f"computed scaffold/core keys for {index} molecules", flush=True)

    split_ids = [record.molecule_key for record in eligible]
    random_split = random_assignments(split_ids, seed, fractions)
    scaffold_split = grouped_assignments({key: scaffold_by_key[key] for key in split_ids}, seed, fractions)
    core_split = grouped_assignments({key: core_by_key[key] for key in split_ids}, seed, fractions)
    assert_no_group_leakage(scaffold_split, scaffold_by_key)
    assert_no_group_leakage(core_split, core_by_key)

    _write_csv(
        output_dir / "identity_table.csv",
        [
            "molecule_key",
            "canonical_smiles",
            "canonical_inchi",
            "inchi_key",
            "heavy_atom_count",
            "ring_count",
            "compound_type",
            "murcko_scaffold",
            "generic_core",
        ],
        (
            {
                **asdict(record),
                "murcko_scaffold": scaffold_by_key.get(record.molecule_key, ""),
                "generic_core": core_by_key.get(record.molecule_key, ""),
            }
            for record in eligible
        ),
    )
    _write_csv(
        output_dir / "splits.csv",
        ["molecule_key", "random_split", "scaffold_split", "core_split"],
        (
            {
                "molecule_key": key,
                "random_split": random_split[key],
                "scaffold_split": scaffold_split[key],
                "core_split": core_split[key],
            }
            for key in split_ids
        ),
    )
    _write_csv(output_dir / "identity_warnings.csv", ["row", "molecule_key", "error"], identity_warnings)
    _write_csv(output_dir / "identity_errors.csv", ["row", "molecule_key", "error"], identity_errors)
    _write_csv(
        output_dir / "duplicate_identities.csv",
        ["molecule_key", "representative_key", "identity"],
        duplicate_rows,
    )
    _write_csv(output_dir / "scaffold_errors.csv", ["molecule_key", "error"], scaffold_errors)
    _write_csv(
        output_dir / "hdf5_only_keys.csv", ["molecule_key"], ({"molecule_key": key} for key in hdf5_only)
    )
    _write_csv(
        output_dir / "csv_missing_hdf5.csv",
        ["molecule_key"],
        ({"molecule_key": key} for key in csv_missing_hdf5),
    )

    checksums: dict[str, object] = {}
    if checksum_path is not None:
        checksums["files"] = verify_sha512_files(checksum_path)
        expected = parse_sha512_file(checksum_path).get(Path(hdf5_path).name)
        if compressed_hdf5_path is not None and expected is not None:
            checksums["decompressed_gzip_stream"] = verify_gzip_decompressed_sha512(
                compressed_hdf5_path, expected
            )

    def split_counts(assignments: Mapping[str, str]) -> dict[str, int]:
        return dict(Counter(assignments.values()))

    report: dict[str, object] = {
        "inputs": {
            "identity_csv": str(csv_path),
            "hdf5": str(hdf5_path),
            "source_manifest": str(source_manifest_path),
        },
        "identity": {
            **dedup_report,
            "canonicalization_warnings": len(identity_warnings),
            "records_without_usable_identity": len(unparsed_identity_ids),
        },
        "alignment": {
            "hdf5_keys": len(hdf5_ids),
            "csv_unique_ids": len(csv_ids),
            "csv_ids_missing_hdf5": len(csv_missing_hdf5),
            "hdf5_ids_missing_csv": len(hdf5_only),
            "selected_ids_missing_manifest": len(selected_ids - found_manifest_ids),
            "hdf5_only_identity_matches": hdf5_only_report,
        },
        "manifest": manifest_report,
        "duplicate_target_differences": duplicate_target_report,
        "legacy_random_split_leakage": legacy_leakage_report,
        "eligible_molecules": len(split_ids),
        "scaffolds": {
            "errors": len(scaffold_errors),
            "exact_groups": len(set(scaffold_by_key.values())),
            "generic_core_groups": len(set(core_by_key.values())),
            "acyclic_molecules": sum(value.startswith("acyclic:") for value in scaffold_by_key.values()),
        },
        "splits": {
            "seed": seed,
            "fractions": list(fractions),
            "random": split_counts(random_split),
            "scaffold": split_counts(scaffold_split),
            "core": split_counts(core_split),
        },
        "checksums": checksums,
    }
    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
