#!/usr/bin/env python3
"""Turn teacher predictions on the molecule pool into a pseudo-labelled packed dataset (TASK-257, DOC-248 L1).

Teacher = production blend (KNOW-256): weighted mean of family means (UMA rich seeds, PubChemQC
ViSNet seeds), S1-S5 sorted ascending. sigma = std of S1-S5 over all members; molecules whose mean
sigma exceeds the --sigma-quantile are dropped (unreliable pseudo-labels). Output follows the
pubchemqc-st-packed-v1 schema, so the existing `pubchemqc` dataset type and configs train on it:
S_eV[:, :5] = teacher S1-S5, S_eV[:, 5:] = S5 (unused filler). A manifest and report go next to it.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

ENERGY = [f"S{i}_eV" for i in range(1, 6)]
PACKED_SCHEMA = "pubchemqc-st-packed-v1"


def read_member(path: str, n: int) -> np.ndarray:
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        cols = [f"prediction_{c}" for c in ENERGY]
        out = np.empty((n, 5), dtype=np.float64)
        keys = []
        for i, row in enumerate(reader):
            keys.append(row["molecule_key"])
            out[i] = [float(row[c]) for c in cols]
    if len(keys) != n:
        raise SystemExit(f"{path}: {len(keys)} rows, pool has {n}")
    return out, keys


def teacher_labels(families: list[list[np.ndarray]], weights: list[float]) -> tuple[np.ndarray, np.ndarray]:
    mean = sum(w * np.mean(f, axis=0) for w, f in zip(weights, families)) / sum(weights)
    sigma = np.std([m for f in families for m in f], axis=0)
    return np.sort(mean, axis=1), sigma


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--family", action="append", nargs="+", required=True, help="weight then member CSVs")
    parser.add_argument("--sigma-quantile", type=float, default=0.9)
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()

    pool = np.load(args.pool, allow_pickle=False)
    keys = np.array([str(k) for k in pool["molecule_key"]])
    n = keys.size
    weights, families = [], []
    for fam in args.family:
        weights.append(float(fam[0]))
        members = []
        for path in fam[1:]:
            values, member_keys = read_member(path, n)
            if member_keys != keys.tolist():
                raise SystemExit(f"{path}: molecule order differs from the pool")
            members.append(values)
        families.append(members)
    label, sigma = teacher_labels(families, weights)
    mol_sigma = sigma.mean(axis=1)
    finite = np.isfinite(label).all(axis=1) & np.isfinite(mol_sigma)
    cutoff = float(np.quantile(mol_sigma[finite], args.sigma_quantile))
    keep = finite & (mol_sigma <= cutoff)
    idx = np.flatnonzero(keep)

    offsets = pool["atom_offsets"]
    counts = np.diff(offsets)[idx]
    new_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    atom_index = np.concatenate([np.arange(offsets[m], offsets[m + 1]) for m in idx])
    s_ev = np.repeat(label[idx, 4:5], 10, axis=1).astype(np.float32)
    s_ev[:, :5] = label[idx]
    Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_h5, "w") as h5:
        h5.attrs["schema"] = PACKED_SCHEMA
        h5.attrs["source"] = "teacher pseudo-labels (scripts/build_distill_dataset.py)"
        h5["numbers"] = pool["numbers"][atom_index].astype(np.uint8)
        h5["positions"] = pool["positions"][atom_index].astype(np.float32)
        h5["atom_offsets"] = new_offsets
        h5["S_eV"] = s_ev
        h5["molecule_key"] = np.array([k.encode() for k in keys[idx]])
    with open(args.out_manifest, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "pack_index", "natoms", *[f"S{i}_eV" for i in range(1, 11)], "teacher_sigma", "status"])
        for j, m in enumerate(idx):
            writer.writerow([keys[m], j, int(counts[j]), *(f"{v:.6f}" for v in s_ev[j]), f"{mol_sigma[m]:.6f}", "ok"])
    report = {"pool": int(n), "kept": int(idx.size), "sigma_quantile": args.sigma_quantile, "sigma_cutoff_eV": cutoff,
              "kept_mean_sigma_eV": float(mol_sigma[idx].mean()), "dropped_mean_sigma_eV": float(mol_sigma[~keep & finite].mean()) if (~keep & finite).any() else None,
              "teacher_label_mean_eV": [float(v) for v in label[idx].mean(axis=0)],
              "families": [{"weight": w, "members": f[1:]} for w, f in zip(weights, args.family)]}
    Path(args.out_h5).with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
