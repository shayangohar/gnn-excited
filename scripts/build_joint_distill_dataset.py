#!/usr/bin/env python3
"""Joint pseudo-label + true-label training set for distillation v2 (TASK-262, DOC-248 L1).

Wave B (KNOW-260) trained on teacher pseudo-labels, then fine-tuned on true GW labels only, which
likely washed out the pseudo-label knowledge (student only -1.4%). This merges both into one
pubchemqc-schema packed set so a single stage sees them together:

  train  every pseudo-labelled pool molecule + each QM9GWBSE train molecule repeated --true-repeat
         times (true S1-S5), so true labels keep a fixed share of every epoch
  val    QM9GWBSE val molecules, true labels  (checkpoint selection on real GW error)
  test   QM9GWBSE test molecules, true labels (directly comparable S-agg)

Keys: pool keys unchanged; QM9 molecules become qm9:<key> (repeats qm9:<key>#r<i>). Writes the
packed HDF5, a manifest, and a split CSV (molecule_key, split) for dataset.split_path.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

PACKED_SCHEMA = "pubchemqc-st-packed-v1"
ENERGY = [f"S{i}_eV" for i in range(1, 6)]


def qm9_entries(split: np.ndarray, true_repeat: int) -> list[tuple[int, str, str]]:
    """(qm9 row, new key suffix, split) for every QM9 molecule; train rows repeated."""
    entries = []
    for row, name in enumerate(split):
        if name == "train":
            entries += [(row, f"#r{i}" if true_repeat > 1 else "", "train") for i in range(true_repeat)]
        elif name in ("val", "test"):
            entries.append((row, "", name))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pseudo-h5", default="data/processed/distill/teacher_pseudolabels.h5")
    parser.add_argument("--qm9-packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--true-repeat", type=int, default=2)
    parser.add_argument("--out-h5", default="data/processed/distill/joint_pseudo_true.h5")
    parser.add_argument("--out-manifest", default="data/processed/distill/joint_pseudo_true_manifest.csv")
    parser.add_argument("--out-split", default="data/processed/distill/joint_pseudo_true_split.csv")
    parser.add_argument("--max-pseudo", type=int, default=None, help="smoke tests: keep only the first N pool molecules")
    parser.add_argument("--max-qm9-per-split", type=int, default=None, help="smoke tests: cap QM9 molecules per split")
    args = parser.parse_args()

    with h5py.File(args.pseudo_h5, "r") as h5:
        p_numbers, p_positions = h5["numbers"][()], h5["positions"][()]
        p_offsets, p_s = h5["atom_offsets"][()], h5["S_eV"][()]
        p_keys = [k.decode() for k in h5["molecule_key"][()]]
    if args.max_pseudo:
        n = args.max_pseudo
        p_numbers, p_positions = p_numbers[: p_offsets[n]], p_positions[: p_offsets[n]]
        p_offsets, p_s, p_keys = p_offsets[: n + 1], p_s[:n], p_keys[:n]
    q = np.load(args.qm9_packed, allow_pickle=False)
    cols = [str(c) for c in q["target_columns"]]
    q_energy = q["targets"][:, [cols.index(c) for c in ENERGY]].astype(np.float32)
    q_keys = [str(k) for k in q["molecule_key"]]
    q_offsets, q_numbers, q_positions = q["atom_offsets"], q["numbers"], q["positions"]
    entries = qm9_entries(q["split"].astype(str), args.true_repeat)
    if args.max_qm9_per_split:
        seen = {}
        kept = []
        for e in entries:
            seen[e[2]] = seen.get(e[2], 0) + 1
            if seen[e[2]] <= args.max_qm9_per_split:
                kept.append(e)
        entries = kept

    numbers, positions, counts, s_rows, keys, splits = [p_numbers], [p_positions], [np.diff(p_offsets)], [p_s], list(p_keys), ["train"] * len(p_keys)
    qa = []
    for row, suffix, name in entries:
        a, b = int(q_offsets[row]), int(q_offsets[row + 1])
        qa.append(np.arange(a, b))
        keys.append(f"qm9:{q_keys[row]}{suffix}")
        splits.append(name)
    q_rows = np.array([e[0] for e in entries])
    atom_index = np.concatenate(qa)
    numbers.append(q_numbers[atom_index].astype(np.uint8))
    positions.append(q_positions[atom_index].astype(np.float32))
    counts.append(np.diff(q_offsets)[q_rows])
    q_s = np.repeat(q_energy[q_rows, 4:5], 10, axis=1)
    q_s[:, :5] = q_energy[q_rows]
    s_rows.append(q_s)

    all_counts = np.concatenate(counts)
    offsets = np.concatenate([[0], np.cumsum(all_counts)]).astype(np.int64)
    s_all = np.concatenate(s_rows).astype(np.float32)
    Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_h5, "w") as h5:
        h5.attrs["schema"] = PACKED_SCHEMA
        h5.attrs["source"] = "teacher pseudo-labels + QM9GWBSE true labels (scripts/build_joint_distill_dataset.py)"
        h5["numbers"] = np.concatenate(numbers).astype(np.uint8)
        h5["positions"] = np.concatenate(positions).astype(np.float32)
        h5["atom_offsets"] = offsets
        h5["S_eV"] = s_all
        h5["molecule_key"] = np.array([k.encode() for k in keys])
    with open(args.out_manifest, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "pack_index", "natoms", *[f"S{i}_eV" for i in range(1, 11)], "source", "status"])
        for j, key in enumerate(keys):
            writer.writerow([key, j, int(all_counts[j]), *(f"{v:.6f}" for v in s_all[j]), "qm9gwbse" if key.startswith("qm9:") else "teacher", "ok"])
    with open(args.out_split, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "split"])
        writer.writerows(zip(keys, splits))
    report = {"pseudo": len(p_keys), "qm9_train_rows": int(sum(1 for e in entries if e[2] == "train")),
              "qm9_val": int(sum(1 for e in entries if e[2] == "val")), "qm9_test": int(sum(1 for e in entries if e[2] == "test")),
              "true_repeat": args.true_repeat, "total": len(keys)}
    report["true_share_of_train"] = report["qm9_train_rows"] / (report["qm9_train_rows"] + report["pseudo"])
    Path(args.out_h5).with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
