#!/usr/bin/env python3
"""QM9GWBSE rebuilt on SMILES-derived GFN2-xTB geometries, for training on cheap geometries (TASK-310).

The wave E geometry test (KNOW-308) showed that models trained on B3LYP geometries lose most of their
accuracy on geometries built from SMILES. This writes a packed dataset with the same molecule keys,
labels and train/val/test split as --packed, but with each molecule's geometry rebuilt from its QM9
SMILES exactly as scripts/build_cheap_geometries.py does (ETKDGv3 seed 0 + MMFF94, then GFN2-xTB), so
the E3 test pools are reproduced for the test molecules. Molecules whose SMILES fails to embed, whose
composition disagrees with the dataset, or where xTB fails are dropped (counted in the report).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cheap_geometry import embed, xtb_optimize

XTB = None


def _build(item):
    row, smiles, numbers = item
    if smiles is None:
        return row, "no_smiles", None
    etkdg = embed(smiles, seed=0, forcefield="MMFF")
    if etkdg is None:
        return row, "embed_failed", None
    z, pos = etkdg
    if sorted(z.tolist()) != sorted(numbers.tolist()):
        return row, "composition_mismatch", None
    xtb_pos = xtb_optimize(z, pos, XTB)
    if xtb_pos is None:
        return row, "xtb_failed", None
    return row, "ok", (z, xtb_pos)


def per_molecule(name: str, value: np.ndarray, n_mol: int) -> bool:
    """True for a per-molecule array (first axis = molecules); target_columns is never one."""
    return name != "target_columns" and value.ndim > 0 and value.shape[0] == n_mol


def assemble(arrays: dict[str, np.ndarray], built: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Packed arrays for the kept rows (ascending), per-molecule fields subset, geometry replaced."""
    rows = np.array(sorted(built), dtype=np.int64)
    n_mol = arrays["molecule_key"].shape[0]
    counts = np.array([built[r][0].size for r in rows], dtype=np.int64)
    out = {"numbers": np.concatenate([built[r][0] for r in rows]).astype(np.uint8),
           "positions": np.concatenate([built[r][1] for r in rows]).astype(np.float32),
           "atom_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
           "source_row": rows}
    for name, value in arrays.items():
        if name in out or name in ("numbers", "positions", "atom_offsets"):
            continue
        out[name] = value[rows] if per_molecule(name, value, n_mol) else value
    return out


def main() -> None:
    global XTB
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--identities", default="data/raw/qm9gwbse/qm9_identities.csv")
    parser.add_argument("--xtb", required=True, help="path to the xtb binary")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="first N molecules only (smoke tests)")
    parser.add_argument("--out", default="data/processed/qm9gwbse_xtb/qm9gwbse_xtb_packed.npz")
    args = parser.parse_args()
    XTB = args.xtb
    arrays = dict(np.load(args.packed, allow_pickle=False))
    keys = [str(k) for k in arrays["molecule_key"]]
    offsets = arrays["atom_offsets"]
    with open(args.identities, newline="", encoding="utf-8") as stream:
        smiles = {row["qm9_id"]: row["smiles"] for row in csv.DictReader(stream)}
    n = len(keys) if args.limit is None else min(args.limit, len(keys))
    if args.limit is not None:
        arrays = {k: (v[:n] if per_molecule(k, v, len(keys)) else v) for k, v in arrays.items()}
        arrays["atom_offsets"] = offsets[:n + 1]
        arrays["numbers"] = arrays["numbers"][:offsets[n]]
        arrays["positions"] = arrays["positions"][:offsets[n]]
    items = [(r, smiles.get(keys[r]), arrays["numbers"][offsets[r]:offsets[r + 1]].astype(np.uint8)) for r in range(n)]

    built, status = {}, Counter()
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        for done, (row, state, geometry) in enumerate(pool.imap_unordered(_build, items, chunksize=16), start=1):
            status[state] += 1
            if geometry is not None:
                built[row] = geometry
            if done % 5000 == 0:
                print(f"{done}/{n} molecules, {dict(status)}, {time.perf_counter() - started:.0f}s", flush=True)

    out = assemble(arrays, built)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    split = out["split"].astype(str)
    report = {"source": args.packed, "molecules": n, "kept": int(len(built)), "status": dict(status),
              "kept_by_split": {s: int((split == s).sum()) for s in ("train", "val", "test")},
              "seconds": time.perf_counter() - started}
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
