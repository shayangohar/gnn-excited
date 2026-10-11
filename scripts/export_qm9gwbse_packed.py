#!/usr/bin/env python3
"""Export QM9GWBSE geometries, targets and the persisted split to one .npz (TASK-211).

The UMA environment (envs/uma-fairchem) has no h5py or torch_geometric, so the
UMA finetune reads this file instead of qm9gwbse.h5. Rows are the manifest's
status==ok rows; splits come from splits.csv exactly as train.py's
explicit_split resolves them (107,101 / 13,390 / 13,392).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

TARGET_COLUMNS = (
    *[f"S{k}_eV" for k in range(1, 6)],
    *[f"log1p_S{k}_f" for k in range(1, 6)],
    *[f"T{k}_eV" for k in range(1, 6)],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", default="data/processed/qm9gwbse/qm9gwbse.h5")
    parser.add_argument("--manifest", default="data/processed/qm9gwbse/manifest.csv")
    parser.add_argument("--splits", default="data/processed/qm9gwbse/splits.csv")
    parser.add_argument("--split-column", default="random_split")
    parser.add_argument("--out", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    args = parser.parse_args()

    with open(args.splits, newline="", encoding="utf-8") as stream:
        splits = {row["molecule_key"]: row[args.split_column].strip().lower() for row in csv.DictReader(stream)}
    with open(args.manifest, newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status") == "ok"]
    missing = [row["molecule_key"] for row in rows if row["molecule_key"] not in splits]
    if missing:
        raise ValueError(f"{len(missing)} manifest rows have no split; first={missing[0]}")

    numbers, positions = [], []
    counts = np.empty(len(rows), dtype=np.int64)
    with h5py.File(args.hdf5, "r") as h5:
        for i, row in enumerate(rows):
            ground = h5[row["molecule_key"]]["ground_state"]
            z = np.asarray(ground["labels"][()], dtype=np.int64).reshape(-1)
            pos = np.asarray(ground["coords"][()], dtype=np.float32)
            if pos.shape != (z.size, 3):
                raise ValueError(f"{row['molecule_key']}: labels {z.shape} vs coords {pos.shape}")
            numbers.append(z.astype(np.uint8))
            positions.append(pos)
            counts[i] = z.size
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    targets = np.asarray([[float(row[c]) for c in TARGET_COLUMNS] for row in rows], dtype=np.float32)
    split = np.asarray([splits[row["molecule_key"]] for row in rows])
    np.savez(
        args.out,
        numbers=np.concatenate(numbers),
        positions=np.concatenate(positions),
        atom_offsets=offsets,
        molecule_key=np.asarray([row["molecule_key"] for row in rows]),
        split=split,
        target_columns=np.asarray(TARGET_COLUMNS),
        targets=targets,
    )
    sizes = {name: int((split == name).sum()) for name in ("train", "val", "test")}
    print(f"wrote {args.out}: molecules={len(rows)} atoms={int(offsets[-1])} splits={sizes}", flush=True)


if __name__ == "__main__":
    main()
