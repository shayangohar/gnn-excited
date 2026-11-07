#!/usr/bin/env python3
"""Train/test sets for comparing a UMA gap model with rCEG (Wang & Closser, ACS Omega 2026; KNOW-293).

rCEG predicts the B3LYP HOMO-LUMO gap from 2D descriptors. Its reproducibility package
(github.com/kclosser/rCEG, GPL-3; cloned to data/external/rCEG, not committed) ships its 424-molecule
held-out test split with rCEG's predictions (data/figure6_data.csv), its 2,137-point reference pool
(figures/figure1_data.csv: 1,228 PQR-QM9 overlap molecules with QM9 gaps + 909 Psi4 B3LYP
recomputations), the recomputations with charge and multiplicity (data/pqr_recomputed_reference.csv)
and QM9 gaps (data/qm9_gap_reference.csv). Molecules match on canonical SMILES without stereo.

Writes to --out-dir:
  ref.npz          rCEG's real-label pool minus the test molecules (its real-label budget)
  qm9.npz          ref.npz plus every QM9 gap whose molecule is not a test molecule
  pool_smiles.npz  the 424 test molecules on ETKDGv3 + UFF geometries (rCEG's recompute protocol; what a
                   SMILES-only user has)
  test_meta.csv    key, smiles, domain, ref_source, ref_gap, rceg_pred, charge, spin

Geometries: QM9 molecules use the QM9 B3LYP geometries; recomputed molecules use ETKDGv3 + UFF, the
protocol their labels were computed on. In the .npz files the 424 test molecules form the `test` split
with these label geometries; 5% of the training molecules form `val`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cheap_geometry import charge_and_multiplicity, embed, pack


def canonical(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rceg", default="data/external/rCEG")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--out-dir", default="data/processed/rceg_compare")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    root, out = Path(args.rceg), Path(args.out_dir)
    rng = np.random.default_rng(17)

    q = np.load(args.packed, allow_pickle=False)
    q_index = {str(k): i for i, k in enumerate(q["molecule_key"])}
    q_off = q["atom_offsets"]

    def qm9_geometry(qm9_id: str):
        i = q_index.get(qm9_id)
        if i is None:
            return None
        return q["numbers"][q_off[i]:q_off[i + 1]].astype(np.uint8), q["positions"][q_off[i]:q_off[i + 1]].astype(np.float64)

    qm9 = read_csv(root / "data/qm9_gap_reference.csv")
    qm9_by_smiles = {}
    for row in qm9:
        qm9_by_smiles.setdefault(canonical(row["smiles"]), row)
    recomputed = {canonical(r["smiles"]): r for r in read_csv(root / "data/pqr_recomputed_reference.csv")}

    def molecule(smiles: str, prefer_qm9: bool):
        """(numbers, positions, charge, spin, label source) for one SMILES."""
        c = canonical(smiles)
        if prefer_qm9 and c in qm9_by_smiles and (geom := qm9_geometry(qm9_by_smiles[c]["qm9_id"])) is not None:
            return (*geom, 0, 1)
        rec = recomputed.get(c)
        charge, spin = (int(rec["charge"]), int(rec["multiplicity"])) if rec else (charge_and_multiplicity(smiles) or (0, 1))
        geom = embed(smiles, seed=0, forcefield="UFF")
        return (*geom, charge, spin) if geom is not None else None

    test_rows = read_csv(root / "data/figure6_data.csv")
    test_canonical = {canonical(r["smiles"]) for r in test_rows}
    records, labels, splits, charges, spins = [], [], [], [], []
    meta, smiles_pool, pool_charge, pool_spin = [], [], [], []
    for i, row in enumerate(test_rows):
        key = f"rceg_test_{i:03d}"
        label_geom = molecule(row["smiles"], prefer_qm9=row["ref_source"].startswith("qm9_overlap"))
        cm = charge_and_multiplicity(row["smiles"]) or (0, 1)
        rec = recomputed.get(canonical(row["smiles"]))
        charge, spin = (int(rec["charge"]), int(rec["multiplicity"])) if rec else cm
        smiles_geom = embed(row["smiles"], seed=0, forcefield="UFF")
        if label_geom is None or smiles_geom is None:
            print(f"test molecule {i} skipped: no geometry for {row['smiles']}")
            continue
        records.append((key, label_geom[0], label_geom[1]))
        labels.append(float(row["ref_gap"]))
        splits.append("test")
        charges.append(label_geom[2])
        spins.append(label_geom[3])
        smiles_pool.append((key, *smiles_geom))
        pool_charge.append(charge)
        pool_spin.append(spin)
        meta.append([key, row["smiles"], row["domain"], row["ref_source"], row["ref_gap"], row["pred"], charge, spin])
    n_test = len(records)

    # figure1_data.csv lists the pool's SMILES, but its `gap` column is the raw semiempirical PQR gap.
    # Labels come from the QM9 gap table (overlap molecules) or the Psi4 recomputation (the others).
    pool_rows = read_csv(root / "figures/figure1_data.csv")
    ref_seen = set()
    for row in pool_rows:
        c = canonical(row["smiles"])
        if c is None or c in test_canonical or c in ref_seen:
            continue
        is_qm9 = "QM9" in row["series"] and c in qm9_by_smiles
        if not is_qm9 and c not in recomputed:
            continue
        ref_seen.add(c)
        geom = molecule(row["smiles"], prefer_qm9=is_qm9)
        if geom is None:
            continue
        records.append((f"rceg_ref_{len(records):05d}", geom[0], geom[1]))
        labels.append(float(qm9_by_smiles[c]["gap"] if is_qm9 else recomputed[c]["gap"]))
        splits.append("train")
        charges.append(geom[2])
        spins.append(geom[3])
    n_ref = len(records)

    def write(path: Path, rows, y, split, charge, spin):
        split = np.array(split, dtype="<U5")
        train = np.flatnonzero(split == "train")
        split[rng.choice(train, size=int(round(args.val_fraction * train.size)), replace=False)] = "val"
        counts = np.array([r[1].size for r in rows], dtype=np.int64)
        np.savez(path, numbers=np.concatenate([r[1] for r in rows]).astype(np.uint8),
                 positions=np.concatenate([r[2] for r in rows]).astype(np.float32),
                 atom_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                 targets=np.array(y, dtype=np.float32)[:, None], target_columns=np.array(["gap_eV"]),
                 molecule_key=np.array([r[0] for r in rows]), split=split,
                 charge=np.array(charge, dtype=np.int64), spin=np.array(spin, dtype=np.int64))
        return {name: int((split == name).sum()) for name in ("train", "val", "test")}

    out.mkdir(parents=True, exist_ok=True)
    report = {"test": n_test, "ref_train_pool": n_ref - n_test}
    report["ref_splits"] = write(out / "ref.npz", records, labels, splits, charges, spins)
    for row in qm9:
        c = canonical(row["smiles"])
        if c is None or c in test_canonical or c in ref_seen:
            continue
        geom = qm9_geometry(row["qm9_id"])
        if geom is None:
            continue
        ref_seen.add(c)
        records.append((f"qm9_{row['qm9_id']}", geom[0], geom[1]))
        labels.append(float(row["gap"]))
        splits.append("train")
        charges.append(0)
        spins.append(1)
    report["qm9_splits"] = write(out / "qm9.npz", records, labels, splits, charges, spins)
    pack(smiles_pool, out / "pool_smiles.npz", charge=pool_charge, spin=pool_spin)
    with open(out / "test_meta.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "smiles", "domain", "ref_source", "ref_gap", "rceg_pred", "charge", "spin"])
        writer.writerows(meta)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
