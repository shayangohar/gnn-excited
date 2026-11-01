#!/usr/bin/env python3
"""Select QM9GWBSE molecules for the ORCA STEOM-DLPNO-CCSD pilot (TASK-262, KNOW-258).

Molecules come from the QM9GWBSE *train* split only, so the pilot labels never touch val/test.
Stage 1 is a small timing batch stratified over heavy-atom counts; stage 2 is a random batch
from the remaining train molecules. Writes <out>/xyz/<key>.xyz (QM9 B3LYP geometry, Angstrom)
and <out>/stage1.txt, <out>/stage2.txt (one key per line).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def stratified_pick(heavy: np.ndarray, candidates: np.ndarray, per_size: dict[int, int], rng) -> np.ndarray:
    picks = []
    for size, count in per_size.items():
        pool = candidates[heavy[candidates] == size]
        picks.append(rng.choice(pool, size=min(count, pool.size), replace=False))
    return np.concatenate(picks)


def write_xyz(path: Path, numbers: np.ndarray, positions: np.ndarray, comment: str) -> None:
    lines = [str(numbers.size), comment]
    lines += [f"{SYMBOLS[int(z)]} {x:.8f} {y:.8f} {w:.8f}" for z, (x, y, w) in zip(numbers, positions)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--out", default="data/processed/orca_pilot")
    parser.add_argument("--stage2", type=int, default=176)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    q = np.load(args.packed, allow_pickle=False)
    keys = np.array([str(k) for k in q["molecule_key"]])
    offsets, numbers, positions = q["atom_offsets"], q["numbers"], q["positions"]
    heavy = np.add.reduceat((numbers > 1).astype(int), offsets[:-1])
    train = np.flatnonzero(q["split"].astype(str) == "train")
    rng = np.random.default_rng(args.seed)
    stage1 = stratified_pick(heavy, train, {5: 2, 6: 2, 7: 4, 8: 6, 9: 10}, rng)
    rest = np.setdiff1d(train, stage1)
    stage2 = rng.choice(rest, size=args.stage2, replace=False)
    out = Path(args.out)
    (out / "xyz").mkdir(parents=True, exist_ok=True)
    for row in np.concatenate([stage1, stage2]):
        a, b = int(offsets[row]), int(offsets[row + 1])
        write_xyz(out / "xyz" / f"{keys[row]}.xyz", numbers[a:b], positions[a:b], f"qm9gwbse {keys[row]}")
    (out / "stage1.txt").write_text("\n".join(keys[stage1]) + "\n", encoding="utf-8")
    (out / "stage2.txt").write_text("\n".join(keys[stage2]) + "\n", encoding="utf-8")
    report = {"stage1": int(stage1.size), "stage2": int(stage2.size),
              "stage1_heavy": {int(h): int(c) for h, c in zip(*np.unique(heavy[stage1], return_counts=True))},
              "stage2_heavy": {int(h): int(c) for h, c in zip(*np.unique(heavy[stage2], return_counts=True))}}
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
