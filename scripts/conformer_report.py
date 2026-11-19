#!/usr/bin/env python3
"""Which SMILES-built conformer should the model see? (TASK-320, wave G conformer study)

Reads the conformers of scripts/build_conformer_pool.py (conformers.csv: xTB energy and heavy-atom RMSD
to the B3LYP geometry for "<key>:c<j>") and each model's predictions on that pool (--pred name=csv). Per
model and per molecule, it turns the conformer predictions (S1-S5 sorted) into one prediction by:
  c0             conformer 0 (the current pipeline, same as pool_xtb.npz)
  mean           the mean over all conformers
  min_energy     the conformer with the lowest GFN2-xTB energy
  boltzmann      the Boltzmann-weighted mean at 298.15 K over the xTB energies
  oracle_rmsd    the conformer closest to the B3LYP geometry (needs the answer; an upper bound)
and, with --b3lyp name=csv, the same model at the dataset's B3LYP geometry. Reports S-agg and S1 MAE
against the QM9GWBSE labels with paired-bootstrap CIs against c0, S-agg by c0 RMSD bin, and how often
the lowest-energy conformer is also the one closest to B3LYP.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from head_to_head_report import ENERGY, paired_bootstrap, read_csv_columns

PRED = [f"prediction_{c}" for c in ENERGY]
KT_EH = 0.00094418  # k_B * 298.15 K in hartree
BINS = [0.0, 0.1, 0.3, 0.6, 1.0, np.inf]


def select(confs: list[dict], how: str) -> np.ndarray:
    """One S1-S5 prediction from a molecule's conformers (each: j, energy, rmsd, pred)."""
    if how == "c0":
        return next(c["pred"] for c in confs if c["j"] == 0)
    if how == "mean":
        return np.mean([c["pred"] for c in confs], axis=0)
    if how == "min_energy":
        return min(confs, key=lambda c: c["energy"])["pred"]
    if how == "boltzmann":
        e = np.array([c["energy"] for c in confs])
        w = np.exp(-(e - e.min()) / KT_EH)
        return np.sum(w[:, None] * np.array([c["pred"] for c in confs]), axis=0) / w.sum()
    if how == "oracle_rmsd":
        ranked = [c for c in confs if c["rmsd"] is not None] or confs
        return min(ranked, key=lambda c: c["rmsd"] if c["rmsd"] is not None else np.inf)["pred"]
    raise ValueError(how)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/processed/conformers/conformers.csv")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--pred", action="append", required=True, help="name=csv of predictions on the conformer pool")
    parser.add_argument("--b3lyp", action="append", default=[], help="name=csv of the same model on pool_b3lyp")
    parser.add_argument("--report", default="runs/conformer_report.json")
    args = parser.parse_args()
    meta = defaultdict(dict)
    with open(args.csv, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            meta[row["molecule_key"]][int(row["conf"])] = (float(row["xtb_energy_Eh"]),
                                                           float(row["rmsd_b3lyp_A"]) if row["rmsd_b3lyp_A"] else None)
    q = np.load(args.packed, allow_pickle=False)
    cols = [str(c) for c in q["target_columns"]]
    idx = {str(k): i for i, k in enumerate(q["molecule_key"])}
    keys = sorted(k for k, confs in meta.items() if 0 in confs and k in idx)
    target = q["targets"][[idx[k] for k in keys]][:, [cols.index(c) for c in ENERGY]].astype(np.float64)
    b3lyp = dict(item.split("=", 1) for item in args.b3lyp)
    report = {"molecules": len(keys), "median_conformers": float(np.median([len(meta[k]) for k in keys])),
              "median_distinct_energies": float(np.median([len({round(e, 5) for e, _ in meta[k].values()}) for k in keys])),
              "min_energy_is_closest_to_b3lyp": float(np.mean([
                  min(meta[k], key=lambda j: meta[k][j][0]) == min(meta[k], key=lambda j: meta[k][j][1] if meta[k][j][1] is not None else np.inf)
                  for k in keys])),
              "median_rmsd_A": {"c0": float(np.nanmedian([meta[k][0][1] if meta[k][0][1] is not None else np.nan for k in keys])),
                                "min_energy": float(np.nanmedian([np.nan if (r := meta[k][min(meta[k], key=lambda j: meta[k][j][0])][1]) is None else r for k in keys])),
                                "closest": float(np.nanmedian([min((r for _, r in meta[k].values() if r is not None), default=np.nan) for k in keys]))},
              "models": {}}
    rmsd0 = np.array([meta[k][0][1] if meta[k][0][1] is not None else np.nan for k in keys])
    for item in args.pred:
        name, path = item.split("=", 1)
        pred = read_csv_columns(path, PRED)
        per = {}
        for how in ("c0", "mean", "min_energy", "boltzmann", "oracle_rmsd"):
            per[how] = np.array([np.sort(select([{"j": j, "energy": e, "rmsd": r, "pred": pred[f"{k}:c{j}"]}
                                                 for j, (e, r) in meta[k].items() if f"{k}:c{j}" in pred], how)) for k in keys])
        if name in b3lyp:
            ref = read_csv_columns(b3lyp[name], PRED)
            per["b3lyp_geometry"] = np.array([np.sort(ref[k]) for k in keys])
        err0 = np.abs(per["c0"] - target).mean(axis=1)
        rows = {}
        for how, p in per.items():
            e = np.abs(p - target).mean(axis=1)
            rows[how] = {"S_agg_eV": float(e.mean()), "S1_mae_eV": float(np.abs(p[:, 0] - target[:, 0]).mean()),
                         "S1_signed_eV": float((p[:, 0] - target[:, 0]).mean())}
            if how != "c0":
                rows[how]["minus_c0"] = float(e.mean() - err0.mean())
                rows[how]["minus_c0_ci95"] = paired_bootstrap(e, err0)
        rows["c0_by_rmsd_bin"] = {f"{lo:.1f}-{hi:.1f}": {"n": int(m.sum()), "S_agg_eV": float(err0[m].mean())}
                                  for lo, hi in zip(BINS[:-1], BINS[1:]) for m in [(rmsd0 >= lo) & (rmsd0 < hi)] if m.any()}
        report["models"][name] = rows
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
