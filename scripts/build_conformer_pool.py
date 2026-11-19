#!/usr/bin/env python3
"""Several SMILES-built GFN2-xTB conformers per E3 test molecule (TASK-320, wave G conformer study).

The cheap-geometry pipeline (KNOW-308, KNOW-315) builds ONE conformer per molecule: ETKDGv3 seed 0 +
MMFF94, then GFN2-xTB. Its median heavy-atom RMSD to the dataset's B3LYP geometry is 0.64 A, mostly
because it lands in a different conformer, and the labels belong to the B3LYP one. For every molecule of
data/processed/cheap_geometry/geometries.csv this builds --n-conf conformers (ETKDG seeds 0..n-1, same
protocol, so conformer 0 reproduces pool_xtb.npz) and records each one's xTB energy and its heavy-atom
RMSD to the B3LYP geometry. scripts/conformer_report.py then asks which choice predicts best: conformer 0,
the mean over conformers, the lowest-energy conformer, a Boltzmann average, or the (oracle) conformer
closest to B3LYP. Writes --out (pool, keys "<key>:c<j>") and --csv (molecule_key, conf, xtb_energy_Eh,
rmsd_b3lyp_A).
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cheap_geometry import embed, heavy_atom_rmsd, pack, xtb_optimize

XTB = None


def _build(item):
    key, smiles, j, numbers, positions = item
    etkdg = embed(smiles, seed=j, forcefield="MMFF")
    if etkdg is None:
        return key, j, None
    z, pos = etkdg
    if sorted(z.tolist()) != sorted(numbers.tolist()):
        return key, j, None
    opt = xtb_optimize(z, pos, XTB, return_energy=True)
    if opt is None:
        return key, j, None
    xpos, energy = opt
    return key, j, (z, xpos, energy, heavy_atom_rmsd(numbers, positions, z, xpos))


def main() -> None:
    global XTB
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometries", default="data/processed/cheap_geometry/geometries.csv")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--xtb", required=True)
    parser.add_argument("--n-conf", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="data/processed/conformers/pool_xtb_conf.npz")
    parser.add_argument("--csv", default="data/processed/conformers/conformers.csv")
    args = parser.parse_args()
    XTB = args.xtb
    with open(args.geometries, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[: args.limit]
    q = np.load(args.packed, allow_pickle=False)
    index = {str(k): i for i, k in enumerate(q["molecule_key"])}
    off = q["atom_offsets"]
    items = []
    for row in rows:
        i = index[row["molecule_key"]]
        z, pos = q["numbers"][off[i]:off[i + 1]].astype(np.uint8), q["positions"][off[i]:off[i + 1]].astype(np.float64)
        items += [(row["molecule_key"], row["smiles"], j, z, pos) for j in range(args.n_conf)]
    with Pool(args.workers) as pool:
        results = pool.map(_build, items, chunksize=4)
    ok = [(k, j, r) for k, j, r in results if r is not None]
    pack([(f"{k}:c{j}", r[0], r[1]) for k, j, r in ok], args.out)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "conf", "xtb_energy_Eh", "rmsd_b3lyp_A"])
        writer.writerows([k, j, f"{r[2]:.8f}", "" if r[3] is None else f"{r[3]:.4f}"] for k, j, r in ok)
    per_mol = {}
    for k, j, r in ok:
        per_mol.setdefault(k, []).append(r[3])
    report = {"molecules": len(rows), "conformers_requested": len(items), "conformers_ok": len(ok),
              "molecules_with_conf0": sum(1 for k, j, _ in ok if j == 0),
              "median_conformers_per_molecule": float(np.median([len(v) for v in per_mol.values()])),
              "median_best_rmsd_A": float(np.median([min(x for x in v if x is not None) for v in per_mol.values()
                                                     if any(x is not None for x in v)]))}
    Path(str(Path(args.out).with_suffix("")) + ".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
