#!/usr/bin/env python3
"""Exclusion lists for QM9GWBSE learning-curve and size-extrapolation runs (TASK-250, DOC-248 L5/L13).

train.py drops every molecule named in dataset.exclude_keys_path before applying the persisted
random_split, so a run on a subset only needs an exclusion CSV; test molecules are never excluded.

  lc_f<frac>_s<seed>.csv   keep a random <frac> of the training molecules (val/test untouched)
  size_le8.csv             drop train/val molecules with more than 8 heavy atoms (test: all sizes)
  size_matched_s<seed>.csv drop random train/val molecules so the kept counts equal size_le8's
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def heavy_atom_counts(numbers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    heavy = (numbers > 1).astype(np.int64)
    cumulative = np.concatenate([[0], np.cumsum(heavy)])
    return cumulative[offsets[1:]] - cumulative[offsets[:-1]]


def subsample_exclusions(indices: np.ndarray, keep: int, rng: np.random.Generator) -> np.ndarray:
    """Indices to exclude so that exactly `keep` of `indices` remain."""
    if keep > indices.size:
        raise ValueError(f"cannot keep {keep} of {indices.size}")
    kept = rng.choice(indices, size=keep, replace=False)
    return np.setdiff1d(indices, kept)


def write_exclusions(path: Path, keys: np.ndarray, reason: str) -> int:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "reason"])
        for key in keys:
            writer.writerow([key, reason])
    return int(len(keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--out-dir", default="data/processed/qm9gwbse/subsets")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--max-heavy", type=int, default=8)
    args = parser.parse_args()

    data = np.load(args.packed, allow_pickle=False)
    keys = np.array([str(k) for k in data["molecule_key"]])
    split = data["split"].astype(str)
    heavy = heavy_atom_counts(data["numbers"], data["atom_offsets"])
    train, val = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"molecules": int(keys.size), "train": int(train.size), "val": int(val.size), "files": {}}

    for frac in args.fractions:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            drop = subsample_exclusions(train, int(round(frac * train.size)), rng)
            name = f"lc_f{frac:g}_s{seed}.csv"
            report["files"][name] = {"excluded": write_exclusions(out / name, keys[drop], f"learning_curve_f{frac:g}"),
                                     "train_kept": int(train.size - drop.size)}

    big = np.concatenate([train[heavy[train] > args.max_heavy], val[heavy[val] > args.max_heavy]])
    kept_train = int((heavy[train] <= args.max_heavy).sum())
    kept_val = int((heavy[val] <= args.max_heavy).sum())
    name = f"size_le{args.max_heavy}.csv"
    report["files"][name] = {"excluded": write_exclusions(out / name, keys[big], f"heavy_atoms_gt{args.max_heavy}"),
                             "train_kept": kept_train, "val_kept": kept_val}
    for seed in args.seeds:
        rng = np.random.default_rng(1000 + seed)
        drop = np.concatenate([subsample_exclusions(train, kept_train, rng), subsample_exclusions(val, kept_val, rng)])
        name = f"size_matched_s{seed}.csv"
        report["files"][name] = {"excluded": write_exclusions(out / name, keys[drop], "size_matched_random"),
                                 "train_kept": kept_train, "val_kept": kept_val}
    report["test_heavy_atom_counts"] = {int(h): int(c) for h, c in zip(*np.unique(heavy[split == "test"], return_counts=True))}
    (out / "subsets_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
