#!/usr/bin/env python3
"""Repack PubChemQC extraction chunks into one packed HDF5 + manifest (TASK-217).

Input: train_st_<chunk>.h5 (schema pubchemqc-st-v1, one group per molecule
/m/<key>/{numbers,positions,S_eV}) and train_st_<chunk>_manifest.csv, as
written by scripts/extract_pubchemqc.py.

Output (schema pubchemqc-st-packed-v1, eV and Angstrom):
  HDF5  numbers u1[A]  positions f4[A,3]  atom_offsets i8[M+1]
        S_eV f4[M,10]  molecule_key S[M]
  CSV   molecule_key,pack_index,inchikey,natoms,S1_eV..S10_eV,split,status
  CSV   QM9GWBSE val/test identity exclusions (train.py exclude_keys_path)

merge_pubchemqc.py copied ~3.2M tiny groups one by one into a single HDF5
group and timed out twice (jobs 22684384, 22732921); molecule i here is the
contiguous slice atom_offsets[i]:atom_offsets[i+1] instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

SOURCE_SCHEMA = "pubchemqc-st-v1"
PACKED_SCHEMA = "pubchemqc-st-packed-v1"
N_STATES = 10
ENERGY_COLUMNS = [f"S{state}_eV" for state in range(1, N_STATES + 1)]
MANIFEST_FIELDS = ["molecule_key", "pack_index", "inchikey", "natoms", *ENERGY_COLUMNS, "split", "status"]
EXCLUSION_FIELDS = ["molecule_key", "inchikey", "qm9_ids", "high_level_splits"]


def read_chunk(chunk_h5: str, chunk_manifest: str, local_dir: str | None = None) -> dict:
    """Read one chunk into flat arrays, checking it against its manifest."""
    source = Path(chunk_h5)
    if local_dir:
        local = Path(local_dir) / source.name
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, local)
        source = local
    with open(chunk_manifest, newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status") == "ok"]
    numbers, positions = [], []
    counts = np.empty(len(rows), dtype=np.int64)
    energies = np.empty((len(rows), N_STATES), dtype=np.float64)
    with h5py.File(source, "r") as h5:
        root = h5["m"]
        if str(root.attrs.get("schema")) != SOURCE_SCHEMA:
            raise ValueError(f"{chunk_h5}: schema {root.attrs.get('schema')!r} != {SOURCE_SCHEMA}")
        if len(root) != len(rows):
            raise ValueError(f"{chunk_h5}: {len(root)} groups but {len(rows)} ok manifest rows")
        for i, row in enumerate(rows):
            group = root[row["molecule_key"]]
            z = np.asarray(group["numbers"][()]).reshape(-1)
            pos = np.asarray(group["positions"][()], dtype=np.float64)
            if pos.shape != (z.size, 3) or z.size != int(row["natoms"]):
                raise ValueError(f"{row['molecule_key']}: numbers {z.shape} / positions {pos.shape} / natoms {row['natoms']}")
            if z.min() < 1 or z.max() > 100:
                raise ValueError(f"{row['molecule_key']}: atomic number out of range")
            numbers.append(z.astype(np.uint8))
            positions.append(pos.astype(np.float32))
            counts[i] = z.size
            energies[i] = group["S_eV"][()]
    manifest_energies = np.asarray([[float(row[c]) for c in ENERGY_COLUMNS] for row in rows])
    drift = float(np.abs(manifest_energies - energies).max()) if rows else 0.0
    if drift > 1e-5:
        raise ValueError(f"{chunk_h5}: HDF5 vs manifest S_eV drift {drift:.2e}")
    if local_dir:
        source.unlink()
    return {
        "numbers": np.concatenate(numbers) if numbers else np.empty(0, np.uint8),
        "positions": np.concatenate(positions) if positions else np.empty((0, 3), np.float32),
        "counts": counts,
        "S_eV": energies.astype(np.float32),
        "rows": [[row["molecule_key"], row["inchikey"], row["split"]] for row in rows],
    }


def connectivity_block(inchikey: str) -> str:
    """First InChIKey block: constitution + H layer, ignoring stereo and protonation."""
    return inchikey.strip().split("-")[0] if inchikey and inchikey.strip() else ""


def qm9gwbse_holdout_blocks(
    manifest_path: str | Path,
    splits_path: str | Path,
    holdout: tuple[str, ...] = ("val", "test"),
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, int]]:
    """Map InChIKey connectivity blocks of QM9GWBSE holdout molecules to (qm9_id, split)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    with open(splits_path, newline="", encoding="utf-8") as stream:
        splits = {row["molecule_key"]: row["random_split"].strip().lower() for row in csv.DictReader(stream)}
    blocks: dict[str, list[tuple[str, str]]] = {}
    stats = {"holdout_molecules": 0, "holdout_unparsed": 0}
    with open(manifest_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split = splits.get(row["molecule_key"])
            if row.get("status") != "ok" or split not in holdout:
                continue
            stats["holdout_molecules"] += 1
            smiles = (row.get("canonical_smiles") or "").strip()
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            block = connectivity_block(Chem.MolToInchiKey(mol)) if mol is not None else ""
            if not block:
                stats["holdout_unparsed"] += 1
                continue
            blocks.setdefault(block, []).append((row["qm9_id"], split))
    stats["holdout_blocks"] = len(blocks)
    return blocks, stats


def exclusion_rows(
    keys: list[str], inchikeys: list[str], blocks: dict[str, list[tuple[str, str]]]
) -> list[dict[str, str]]:
    """PubChemQC rows whose identity block matches a QM9GWBSE holdout molecule."""
    excluded = []
    for key, inchikey in zip(keys, inchikeys):
        hits = blocks.get(connectivity_block(inchikey))
        if hits:
            excluded.append({
                "molecule_key": key,
                "inchikey": inchikey,
                "qm9_ids": ";".join(sorted({qm9_id for qm9_id, _ in hits})),
                "high_level_splits": ";".join(sorted({split for _, split in hits})),
            })
    return excluded


def write_packed(out_h5: str | Path, parts: list[dict], source_chunks: list[str]) -> dict[str, np.ndarray]:
    numbers = np.concatenate([part["numbers"] for part in parts])
    positions = np.concatenate([part["positions"] for part in parts])
    counts = np.concatenate([part["counts"] for part in parts])
    energies = np.concatenate([part["S_eV"] for part in parts])
    keys = [row[0] for part in parts for row in part["rows"]]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate molecule_key across chunks")
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    if offsets[-1] != numbers.size or numbers.size != positions.shape[0]:
        raise ValueError("atom array lengths disagree with per-molecule counts")
    tmp = Path(str(out_h5) + ".tmp")
    with h5py.File(tmp, "w") as h5:
        h5.attrs["schema"] = PACKED_SCHEMA
        h5.attrs["units"] = "eV"
        h5.attrs["positions_units"] = "angstrom"
        h5.attrs["n_molecules"] = counts.size
        h5.attrs["n_atoms"] = numbers.size
        h5.attrs["source_chunks"] = json.dumps(source_chunks)
        h5.create_dataset("numbers", data=numbers)
        h5.create_dataset("positions", data=positions)
        h5.create_dataset("atom_offsets", data=offsets)
        h5.create_dataset("S_eV", data=energies)
        h5.create_dataset("molecule_key", data=np.asarray(keys, dtype="S"))
    tmp.replace(out_h5)
    return {"counts": counts, "S_eV": energies, "numbers": numbers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--chunks", default="c0,c1,c2,c3,c4,c5,c6,c7")
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--local-dir", default=None, help="node-local dir to stage chunk copies in")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--qm9gwbse-manifest", default=None)
    parser.add_argument("--qm9gwbse-splits", default=None)
    parser.add_argument("--exclusions-out", default=None)
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    names = args.chunks.split(",")
    jobs = [(str(chunk_dir / f"train_st_{n}.h5"), str(chunk_dir / f"train_st_{n}_manifest.csv"), args.local_dir) for n in names]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        parts = list(pool.map(read_chunk, *zip(*jobs)))
    for name, part in zip(names, parts):
        print(f"chunk {name}: molecules={part['counts'].size} atoms={part['numbers'].size}", flush=True)

    packed = write_packed(args.out_h5, parts, names)
    rows = [row for part in parts for row in part["rows"]]
    keys = [row[0] for row in rows]
    inchikeys = [row[1] for row in rows]
    with open(args.manifest, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(MANIFEST_FIELDS)
        for index, ((key, inchikey, split), natoms, energies) in enumerate(zip(rows, packed["counts"], packed["S_eV"])):
            writer.writerow([key, index, inchikey, int(natoms), *[f"{v:.6f}" for v in energies], split, "ok"])

    counts, energies = packed["counts"], packed["S_eV"]
    elements, element_counts = np.unique(packed["numbers"], return_counts=True)
    blocks_seen = [connectivity_block(k) for k in inchikeys]
    report = {
        "molecules": int(counts.size),
        "atoms": int(counts.sum()),
        "natoms_mean": float(counts.mean()),
        "natoms_max": int(counts.max()),
        "elements": {int(z): int(c) for z, c in zip(elements, element_counts)},
        "S1_eV_mean": float(energies[:, 0].mean()),
        "S1_eV_range": [float(energies[:, 0].min()), float(energies[:, 0].max())],
        "S10_eV_mean": float(energies[:, 9].mean()),
        "missing_inchikey": int(sum(1 for b in blocks_seen if not b)),
        "duplicate_identity_rows": int(len(blocks_seen) - len(set(blocks_seen))),
    }
    if args.qm9gwbse_manifest and args.qm9gwbse_splits and args.exclusions_out:
        blocks, stats = qm9gwbse_holdout_blocks(args.qm9gwbse_manifest, args.qm9gwbse_splits)
        excluded = exclusion_rows(keys, inchikeys, blocks)
        with open(args.exclusions_out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=EXCLUSION_FIELDS)
            writer.writeheader()
            writer.writerows(excluded)
        matched_blocks = {connectivity_block(row["inchikey"]) for row in excluded}
        report["qm9gwbse_exclusions"] = {
            **stats,
            "excluded_pubchemqc_rows": len(excluded),
            "holdout_blocks_found_in_pubchemqc": len(matched_blocks),
        }
    print(json.dumps(report, indent=2), flush=True)
    Path(args.manifest).with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
