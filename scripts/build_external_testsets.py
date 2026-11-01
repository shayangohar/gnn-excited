#!/usr/bin/env python3
"""External GW-BSE test sets as one packed pool for head-to-head evaluation (TASK-271, DOC-248 L13).

Reads the Baum et al. test sets (data/external/ViSNetGWBSE/testsets/excitation_energies):
PC9, OE62L and OE62H hold 5 singlet energies per molecule (Hartree); QM9 holds S1 only (eV).
Adds --n-control molecules from our QM9GWBSE test split so pool inference can be checked
against stored test predictions. Writes to --out-dir:

  pool.npz               numbers, positions, atom_offsets, molecule_key ("<SET>:<name>") for
                         predict_packed_visnet.py, the UMA inference mode and eval_visnetgwbse.py
  labels.csv             molecule_key, set, natoms, heavy_atoms, elements, S1..S5_eV (blank if
                         absent), qm9_overlap_ids (QM9GWBSE ids with the same connectivity graph)
                         and the split of those ids in our split and in Baum et al.'s split
  visnetgwbse_split.csv  molecule_key (QM9 id), split: the seeded split their released code builds
                         (data/qm9gwbse.py). Their released QM9 test molecules fall in it at chance
                         rates, so it is probably not the split behind the paper (TASK-271).

Overlap uses an all-single-bond connectivity graph (RDKit DetermineConnectivity) with explicit
hydrogens, so different optimizations of the same molecule still match.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HARTREE_TO_EV = 27.211386245988
QM9_SIZE = 133885
ENERGY = [f"S{i}_eV" for i in range(1, 6)]


def visnetgwbse_split(n: int = QM9_SIZE, num_train: int = 120000, num_val: int = 10000, seed: int = 42) -> dict[int, str]:
    """QM9 id (1-based) -> split, as data/qm9gwbse.py builds it: random.seed(42) then one shuffle."""
    idx = list(range(1, n + 1))
    random.Random(seed).shuffle(idx)
    split = {i: "train" for i in idx[:num_train]}
    split.update({i: "val" for i in idx[num_train:num_train + num_val]})
    split.update({i: "test" for i in idx[num_train + num_val:]})
    return split


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Symbols and positions (Angstrom) from plain or extended XYZ (the comment line is ignored)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    rows = [line.split() for line in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows], dtype=np.float64)


def set_is_hartree(label_arrays: list[np.ndarray]) -> bool:
    """A set is in Hartree if its median label is below 2 (2 Ha = 54 eV). Decided per set, not per
    file: the QM9 set is in eV but holds a molecule with S1 = 1.57 eV (mol_271)."""
    return bool(np.median(np.concatenate(label_arrays)) < 2.0)


def connectivity_key(numbers, positions) -> str | None:
    """Canonical SMILES of the distance-based connectivity graph (single bonds, explicit H)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds

    RDLogger.DisableLog("rdApp.*")
    table = Chem.GetPeriodicTable()
    block = f"{len(numbers)}\n\n" + "\n".join(
        f"{table.GetElementSymbol(int(z))} {x:.6f} {y:.6f} {w:.6f}" for z, (x, y, w) in zip(numbers, positions))
    try:
        mol = Chem.MolFromXYZBlock(block)
        rdDetermineBonds.DetermineConnectivity(mol)
        mol.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testsets-dir", default="data/external/ViSNetGWBSE/testsets/excitation_energies")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--n-control", type=int, default=256)
    parser.add_argument("--out-dir", default="data/processed/external_testsets")
    args = parser.parse_args()
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    theirs = visnetgwbse_split()
    with open(out / "visnetgwbse_split.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "split"])
        writer.writerows(sorted(theirs.items()))

    q = np.load(args.packed, allow_pickle=False)
    q_keys = [str(k) for k in q["molecule_key"]]
    q_split = q["split"].astype(str)
    q_offsets, q_numbers, q_positions = q["atom_offsets"], q["numbers"], q["positions"]
    cols = [str(c) for c in q["target_columns"]]
    q_energy = q["targets"][:, [cols.index(c) for c in ENERGY]]
    ours = dict(zip(q_keys, q_split))
    qm9_by_key = defaultdict(list)
    for row, key in enumerate(q_keys):
        a, b = int(q_offsets[row]), int(q_offsets[row + 1])
        qm9_by_key[connectivity_key(q_numbers[a:b], q_positions[a:b])].append(key)
        if row % 20000 == 0:
            print(f"QM9 connectivity keys: {row}/{len(q_keys)}", flush=True)
    qm9_by_key.pop(None, None)

    records = []  # (molecule_key, set, numbers, positions, energies[5] with NaN)
    report = {"sets": {}}
    for set_dir in sorted(p for p in Path(args.testsets_dir).iterdir() if p.is_dir()):
        found, missing = [], 0
        for xyz in sorted((set_dir / "XYZ").glob("*.xyz")):
            label_path = set_dir / "E_exc_SS" / f"{xyz.stem}.dat"
            if label_path.exists():
                found.append((xyz, np.loadtxt(label_path, ndmin=1)))
            else:
                missing += 1
        hartree = set_is_hartree([raw for _, raw in found])
        for xyz, raw in found:
            energies = np.full(5, np.nan)
            values = (raw * HARTREE_TO_EV if hartree else raw)[:5]
            energies[:values.size] = values
            symbols, positions = read_xyz(xyz)
            numbers = np.array([table.GetAtomicNumber(s) for s in symbols], dtype=np.uint8)
            records.append((f"{set_dir.name}:{xyz.stem}", set_dir.name, numbers, positions, energies))
        report["sets"][set_dir.name] = {"molecules": len(found), "missing_labels": missing,
                                        "label_unit": "hartree" if hartree else "eV"}
    test_rows = np.flatnonzero(q_split == "test")
    for row in np.sort(np.random.default_rng(0).choice(test_rows, size=min(args.n_control, test_rows.size), replace=False)):
        a, b = int(q_offsets[row]), int(q_offsets[row + 1])
        records.append((f"QM9GWBSE:{q_keys[row]}", "QM9GWBSE_control", q_numbers[a:b].astype(np.uint8),
                        q_positions[a:b].astype(np.float64), q_energy[row].astype(np.float64)))

    counts = np.array([r[2].size for r in records])
    np.savez(out / "pool.npz",
             numbers=np.concatenate([r[2] for r in records]).astype(np.uint8),
             positions=np.concatenate([r[3] for r in records]).astype(np.float32),
             atom_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
             molecule_key=np.array([r[0] for r in records]))
    overlap_counts = Counter()
    theirs_split_of_qm9_set = Counter()
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "set", "natoms", "heavy_atoms", "elements", *ENERGY,
                         "qm9_overlap_ids", "ours_split_of_overlaps", "theirs_split_of_overlaps"])
        for key, name, numbers, positions, energies in records:
            ids = qm9_by_key.get(connectivity_key(numbers, positions), [])
            overlap_counts[name] += bool(ids)
            if name == "QM9":
                theirs_split_of_qm9_set.update(theirs[int(i)] for i in ids)
            elements = "-".join(sorted({table.GetElementSymbol(int(z)) for z in numbers}))
            writer.writerow([key, name, numbers.size, int((numbers > 1).sum()), elements,
                             *("" if np.isnan(e) else f"{e:.6f}" for e in energies),
                             "|".join(ids), "|".join(ours[i] for i in ids), "|".join(theirs[int(i)] for i in ids)])
    for name in report["sets"]:
        report["sets"][name]["with_qm9_overlap"] = overlap_counts[name]
    report["control_molecules"] = int(sum(r[1] == "QM9GWBSE_control" for r in records))
    report["theirs_split_of_qm9_testset_overlaps"] = dict(theirs_split_of_qm9_set)
    report["pool_molecules"] = len(records)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
