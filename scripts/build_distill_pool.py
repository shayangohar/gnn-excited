#!/usr/bin/env python3
"""Unlabeled molecule pool for teacher pseudo-labelling (TASK-257, DOC-248 L1).

Selects PubChemQC molecules (B3LYP/6-31G* geometries) that look like QM9GWBSE chemistry:
elements within --elements, at most --max-heavy heavy atoms, and not named in --exclude (the
screened PubChemQC exclusions: QM9GWBSE val/test identities plus pathological geometries, so no
test identity can receive a teacher label). Optionally caps the pool at --max-molecules, sampled
with a fixed seed. Writes a packed .npz (numbers, positions, atom_offsets, molecule_key) that both
the UMA inference mode and scripts/predict_packed_visnet.py read.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def select_pool(numbers, offsets, keys, allowed, max_heavy, excluded):
    counts = np.diff(offsets)
    molecule_of_atom = np.repeat(np.arange(counts.size), counts)
    bad_atom = ~np.isin(numbers, list(allowed))
    has_bad = np.zeros(counts.size, dtype=bool)
    has_bad[np.unique(molecule_of_atom[bad_atom])] = True
    heavy = np.bincount(molecule_of_atom, weights=(numbers > 1), minlength=counts.size).astype(int)
    keep = (~has_bad) & (heavy <= max_heavy) & (heavy >= 1)
    if excluded:
        keep &= ~np.isin(keys, list(excluded))
    return keep, heavy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed-h5", default="data/processed/pubchemqc/pubchemqc_s10_packed.h5")
    parser.add_argument("--exclude", default="data/processed/pubchemqc/pretrain_exclusions_screened.csv")
    parser.add_argument("--elements", default="1,6,7,8,9")
    parser.add_argument("--max-heavy", type=int, default=12)
    parser.add_argument("--max-molecules", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="data/processed/distill/pool_pubchemqc.npz")
    args = parser.parse_args()

    with h5py.File(args.packed_h5, "r") as h5:
        numbers = h5["numbers"][()].astype(np.int64)
        positions = h5["positions"][()]
        offsets = h5["atom_offsets"][()]
        keys = np.array([k.decode() if isinstance(k, bytes) else str(k) for k in h5["molecule_key"][()]])
    excluded = set()
    if args.exclude:
        with open(args.exclude, newline="", encoding="utf-8-sig") as stream:
            excluded = {row["molecule_key"].strip() for row in csv.DictReader(stream)}
    allowed = {int(z) for z in args.elements.split(",")}
    keep, heavy = select_pool(numbers, offsets, keys, allowed, args.max_heavy, excluded)
    chosen = np.flatnonzero(keep)
    if args.max_molecules and chosen.size > args.max_molecules:
        chosen = np.sort(np.random.default_rng(args.seed).choice(chosen, size=args.max_molecules, replace=False))
    counts = np.diff(offsets)[chosen]
    new_offsets = np.concatenate([[0], np.cumsum(counts)])
    atom_index = np.concatenate([np.arange(offsets[m], offsets[m + 1]) for m in chosen])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, numbers=numbers[atom_index].astype(np.uint8), positions=positions[atom_index].astype(np.float32),
             atom_offsets=new_offsets.astype(np.int64), molecule_key=keys[chosen])
    report = {
        "source": args.packed_h5, "eligible": int(keep.sum()), "pool": int(chosen.size), "atoms": int(new_offsets[-1]),
        "max_heavy": args.max_heavy, "elements": sorted(allowed), "excluded_listed": len(excluded),
        "heavy_atom_histogram": {int(h): int(c) for h, c in zip(*np.unique(heavy[chosen], return_counts=True))},
    }
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
