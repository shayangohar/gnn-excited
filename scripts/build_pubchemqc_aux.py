#!/usr/bin/env python3
"""Heavy-element and matched CHONF PubChemQC subsets for multi-fidelity UMA training (TASK-287).

Reads the packed PubChemQC corpus (pack_pubchemqc.py: B3LYP/6-31+G* TD-DFT S1-S10, eV) and writes two
.npz files in the qm9gwbse_packed layout that scripts/train_uma_qm9gwbse.py reads as an `aux` dataset:
numbers, positions, atom_offsets, targets (S1-S5 eV), target_columns, molecule_key, split (train/val).

  heavy.npz  molecules with at least one of B, Si, P, S, Cl, Br and no element outside
             H B C N O F Si P S Cl Br
  chonf.npz  H C N O F only, sampled to the heavy set's heavy-atom-count histogram (the control:
             same amount and size of extra data, no new elements)

Both drop the screened exclusions (QM9GWBSE val/test identities and pathological geometries;
KNOW-226, KNOW-242), molecules over --max-atoms, non-finite or out-of-range S1-S5, and any molecule
whose connectivity graph matches an external head-to-head molecule (PC9, OE62, the Baum QM9 test set),
so OE62/PC9 results stay out of sample.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from build_external_testsets import connectivity_key

ALLOWED = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35)
HEAVY_MARKERS = (5, 14, 15, 16, 17, 35)
CHONF = (1, 6, 7, 8, 9)
N_STATES = 5
SYMBOL = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br"}


def element_counts(numbers: np.ndarray, offsets: np.ndarray, elements) -> dict[int, np.ndarray]:
    """Per-molecule count of each element; molecule i is numbers[offsets[i]:offsets[i+1]]."""
    return {z: np.add.reduceat((numbers == z).astype(np.int32), offsets[:-1]) for z in elements}


def matched_sample(heavy_atoms: np.ndarray, reference: np.ndarray, candidates: np.ndarray, rng) -> np.ndarray:
    """Candidates drawn without replacement so their heavy-atom-count histogram matches `reference`
    (indices into heavy_atoms); a size with too few candidates contributes all it has."""
    picks = []
    sizes, counts = np.unique(heavy_atoms[reference], return_counts=True)
    for size, count in zip(sizes, counts):
        pool = candidates[heavy_atoms[candidates] == size]
        picks.append(rng.choice(pool, size=min(int(count), pool.size), replace=False))
    return np.sort(np.concatenate(picks)) if picks else np.empty(0, dtype=np.int64)


def write_subset(path: Path, rows: np.ndarray, numbers, positions, offsets, energies, keys, val_fraction: float, rng) -> None:
    counts = (offsets[rows + 1] - offsets[rows]).astype(np.int64)
    split = np.full(rows.size, "train", dtype="<U5")
    split[rng.choice(rows.size, size=int(round(val_fraction * rows.size)), replace=False)] = "val"
    np.savez(path,
             numbers=np.concatenate([numbers[offsets[r]:offsets[r + 1]] for r in rows]).astype(np.uint8),
             positions=np.concatenate([positions[offsets[r]:offsets[r + 1]] for r in rows]).astype(np.float32),
             atom_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
             targets=energies[rows, :N_STATES].astype(np.float32),
             target_columns=np.array([f"S{i}_eV" for i in range(1, N_STATES + 1)]),
             molecule_key=np.array([keys[r] for r in rows]),
             split=split)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/pubchemqc/pubchemqc_s10_packed.h5")
    parser.add_argument("--exclusions", default="data/processed/pubchemqc/pretrain_exclusions_screened.csv")
    parser.add_argument("--external-pool", default="data/processed/external_testsets/pool.npz")
    parser.add_argument("--n-heavy", type=int, default=200_000)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-chonf", action="store_true", help="write heavy.npz only")
    parser.add_argument("--out-dir", default="data/processed/pubchemqc_aux")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.packed, "r") as h5:
        numbers, positions, offsets = h5["numbers"][()], h5["positions"][()], h5["atom_offsets"][()]
        energies = h5["S_eV"][()]
        keys = [k.decode() for k in h5["molecule_key"][()]]
    natoms = np.diff(offsets)
    counts = element_counts(numbers, offsets, ALLOWED)
    allowed = sum(counts.values())
    heavy_atoms = natoms - counts[1]
    with open(args.exclusions, newline="", encoding="utf-8") as stream:
        excluded = {row["molecule_key"] for row in csv.DictReader(stream)}
    s5 = energies[:, :N_STATES]
    ok = ((allowed == natoms) & (natoms <= args.max_atoms) & np.isfinite(s5).all(axis=1)
          & (s5[:, 0] > 0.5) & (s5[:, -1] < 20.0) & np.array([k not in excluded for k in keys]))
    has_marker = sum(counts[z] for z in HEAVY_MARKERS) > 0
    heavy_candidates = np.flatnonzero(ok & has_marker)
    chonf_candidates = np.flatnonzero(ok & (sum(counts[z] for z in CHONF) == natoms))
    print(f"candidates: heavy {heavy_candidates.size}, chonf {chonf_candidates.size}", flush=True)

    pool = np.load(args.external_pool, allow_pickle=False)
    p_off = pool["atom_offsets"]
    external = {connectivity_key(pool["numbers"][p_off[i]:p_off[i + 1]], pool["positions"][p_off[i]:p_off[i + 1]])
                for i in range(p_off.size - 1)} - {None}

    def drop_external(rows: np.ndarray) -> tuple[np.ndarray, int]:
        keep = np.array([connectivity_key(numbers[offsets[r]:offsets[r + 1]], positions[offsets[r]:offsets[r + 1]])
                         not in external for r in rows], dtype=bool)
        return rows[keep], int((~keep).sum())

    # A prefix of one seeded permutation, so runs with the same seed and different --n-heavy get
    # nested subsets (the exchange-rate curve, TASK-291). The wave D 200k set predates this.
    order = rng.permutation(heavy_candidates)
    heavy, heavy_dropped = drop_external(order[:min(int(args.n_heavy * 1.02) + 100, order.size)])
    heavy = np.sort(heavy[:args.n_heavy])
    subsets = [("heavy", heavy)]
    report = {"max_atoms": args.max_atoms, "n_heavy": args.n_heavy, "seed": args.seed,
              "candidates": {"heavy": int(heavy_candidates.size), "chonf": int(chonf_candidates.size)},
              "dropped_external_matches": {"heavy": heavy_dropped}, "subsets": {}}
    if not args.skip_chonf:
        chonf = matched_sample(heavy_atoms, heavy, chonf_candidates, rng)
        chonf, report["dropped_external_matches"]["chonf"] = drop_external(chonf)
        subsets.append(("chonf", chonf))
    for name, rows in subsets:
        write_subset(out / f"{name}.npz", rows, numbers, positions, offsets, energies, keys, args.val_fraction, rng)
        report["subsets"][name] = {
            "molecules": int(rows.size),
            "molecules_with_element": {SYMBOL[z]: int((counts[z][rows] > 0).sum()) for z in ALLOWED if z != 1},
            "heavy_atoms_median": float(np.median(heavy_atoms[rows])),
            "natoms_median": float(np.median(natoms[rows])),
            "S1_mean_eV": float(s5[rows, 0].mean()),
        }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
