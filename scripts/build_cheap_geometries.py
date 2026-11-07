#!/usr/bin/env python3
"""SMILES-only geometries for a QM9GWBSE test subset (TASK-291 E3, TASK-280).

The production models were trained on B3LYP geometries. A lab user has a SMILES string, not a
B3LYP geometry. For --n random test molecules this writes pools (same molecule keys in each) to
--out-dir:

  pool_b3lyp.npz  the dataset geometries (reference)
  pool_etkdg.npz  RDKit ETKDGv3 + MMFF94 from the QM9 SMILES
  pool_xtb.npz    GFN2-xTB optimisation of the ETKDG geometry (--xtb), for molecules where xTB converges

and geometries.csv with heavy-atom RMSD to B3LYP for each source. scripts/relax_uma_pool.py adds
pool_uma.npz (UMA-relaxed ETKDG geometry). Molecules that fail any step are dropped from all pools,
so every pool holds the same molecules.
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
    key, smiles, numbers, positions = item
    etkdg = embed(smiles, seed=0, forcefield="MMFF")
    if etkdg is None:
        return key, None
    z, pos = etkdg
    if sorted(z.tolist()) != sorted(numbers.tolist()):
        return key, None  # SMILES and dataset geometry disagree on composition
    xtb_pos = xtb_optimize(z, pos, XTB) if XTB else None
    return key, {"etkdg": (z, pos), "xtb": (z, xtb_pos) if xtb_pos is not None else None,
                 "rmsd_etkdg": heavy_atom_rmsd(numbers, positions, z, pos),
                 "rmsd_xtb": heavy_atom_rmsd(numbers, positions, z, xtb_pos) if xtb_pos is not None else None}


def main() -> None:
    global XTB
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--identities", default="data/raw/qm9gwbse/qm9_identities.csv")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--xtb", default=None, help="path to the xtb binary; omit to skip GFN2-xTB")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", default="data/processed/cheap_geometry")
    args = parser.parse_args()
    XTB = args.xtb
    q = np.load(args.packed, allow_pickle=False)
    keys = [str(k) for k in q["molecule_key"]]
    offsets = q["atom_offsets"]
    with open(args.identities, newline="", encoding="utf-8") as stream:
        smiles = {row["qm9_id"]: row["smiles"] for row in csv.DictReader(stream)}
    test = np.flatnonzero(q["split"].astype(str) == "test")
    rows = np.sort(np.random.default_rng(0).choice(test, size=min(args.n, test.size), replace=False))
    items = [(keys[r], smiles[keys[r]], q["numbers"][offsets[r]:offsets[r + 1]].astype(np.uint8),
              q["positions"][offsets[r]:offsets[r + 1]].astype(np.float64)) for r in rows if keys[r] in smiles]
    with Pool(args.workers) as pool:
        built = dict(pool.map(_build, items, chunksize=8))

    reference = {key: (z, pos) for key, _, z, pos in items}
    ok = [key for key, _, _, _ in items if built.get(key) and (not XTB or built[key]["xtb"] is not None)]
    out = Path(args.out_dir)
    pack([(k, *reference[k]) for k in ok], out / "pool_b3lyp.npz")
    pack([(k, *built[k]["etkdg"]) for k in ok], out / "pool_etkdg.npz")
    if XTB:
        pack([(k, *built[k]["xtb"]) for k in ok], out / "pool_xtb.npz")
    with open(out / "geometries.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "smiles", "rmsd_etkdg", "rmsd_xtb"])
        for k in ok:
            writer.writerow([k, smiles[k], built[k]["rmsd_etkdg"], built[k]["rmsd_xtb"]])

    def median(name):
        values = [built[k][name] for k in ok if built[k][name] is not None]
        return float(np.median(values)) if values else None

    report = {"requested": int(rows.size), "with_smiles": len(items), "kept": len(ok),
              "embed_failed": sum(1 for k, *_ in items if not built.get(k)),
              "xtb_failed": sum(1 for k, *_ in items if built.get(k) and XTB and built[k]["xtb"] is None),
              "median_heavy_rmsd_A": {"etkdg": median("rmsd_etkdg"), "xtb": median("rmsd_xtb")}}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
