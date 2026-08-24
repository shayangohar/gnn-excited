#!/usr/bin/env python3
"""Convert OMol25 ASE-LMDB shards to project HDF5 + manifest.

Filter: HCNOF only (Z in {1,6,7,8,9}), <= --max-atoms atoms, requires
data['homo_lumo_gap']. Output schema v1:
  HDF5  /m/<id>/numbers(i32[N]) positions(f64[N,3]) gap(f64) homo(f64)
  CSV   id,natoms,formula,gap,homo,split   (split = hash bucket 90/10 train/val)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from ase.db import connect

ALLOWED_Z = {1, 6, 7, 8, 9}


def as_scalar(value):
    """Coerce a label to float; returns None for missing/ambiguous arrays."""
    if value is None:
        return None
    array = np.ravel(np.asarray(value, dtype=float))
    return float(array[0]) if array.size == 1 else None


def split_for(mol_id: str, val_fraction: float = 0.1) -> str:
    digest = hashlib.sha256(mol_id.encode()).hexdigest()
    return "val" if int(digest[:16], 16) % 10_000 < val_fraction * 10_000 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True)
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument(
        "--elements", choices=["HCNOF", "all"], default="HCNOF",
        help="HCNOF restricts composition; all keeps every element",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    shards = sorted(Path(args.lmdb_dir).rglob("*.aselmdb"))
    if not shards:
        raise SystemExit(f"no LMDB shards under {args.lmdb_dir}")
    print(f"shards: {len(shards)}", flush=True)

    kept = scanned = skipped_elem = skipped_size = skipped_label = 0
    gaps = []
    out_h5 = Path(args.out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines = ["id,natoms,formula,gap,homo,split"]
    with h5py.File(out_h5, "w") as h5:
        root = h5.create_group("m")
        root.attrs["schema"] = "omol25-gap-v1"
        root.attrs["units"] = "eV"
        stop = False
        for shard in shards:
            if stop:
                break
            db = connect(str(shard))
            for row in db.select():
                scanned += 1
                if scanned % 50_000 == 0:
                    print(f"scanned={scanned} kept={kept}", flush=True)
                data = getattr(row, "_data", None) or {}
                gap = as_scalar(data.get("homo_lumo_gap"))
                homo = as_scalar(data.get("homo_energy"))
                if gap is None:
                    skipped_label += 1
                    continue
                atoms = row.toatoms()
                numbers = atoms.get_atomic_numbers()
                if args.elements == "HCNOF" and not set(numbers.tolist()) <= ALLOWED_Z:
                    skipped_elem += 1
                    continue
                if len(atoms) > args.max_atoms:
                    skipped_size += 1
                    continue
                mol_id = (
                    f"omol_{hashlib.sha256(str(row.id).encode()).hexdigest()[:12]}"
                    f"_{shard.stem}"
                )
                g = float(gap)
                h = float(homo) if homo is not None else float("nan")
                grp = root.create_group(mol_id)
                grp.create_dataset("numbers", data=numbers.astype(np.int32))
                grp.create_dataset(
                    "positions", data=atoms.get_positions().astype(np.float64)
                )
                grp.create_dataset("gap", data=np.float64(g))
                grp.create_dataset("homo", data=np.float64(h))
                split = split_for(mol_id)
                grp.attrs["split"] = split
                formula = atoms.get_chemical_formula()
                manifest_lines.append(
                    f"{mol_id},{len(atoms)},{formula},{g:.10f},{h:.10f},{split}"
                )
                gaps.append(g)
                kept += 1
                if args.limit and kept >= args.limit:
                    stop = True
                    break

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text("\n".join(manifest_lines) + "\n")
    summary = {
        "scanned": scanned,
        "kept": kept,
        "skipped_element": skipped_elem,
        "skipped_size": skipped_size,
        "skipped_label": skipped_label,
        "gap_min_eV": min(gaps) if gaps else None,
        "gap_max_eV": max(gaps) if gaps else None,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    Path(str(out_h5) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
