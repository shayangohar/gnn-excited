#!/usr/bin/env python3
"""Compare two production specs, a base and a candidate, before a promotion (TASK-310).

For each spec, takes the guarded singlet blend (scripts/head_to_head_report.py; S1-S5 sorted per molecule)
and reports:
  QM9GWBSE test: S-agg and S1 MAE of each, with paired-bootstrap 95% CIs of candidate - base;
  external pool (labels.csv): S1 MAE per set (all molecules) and per OE62H element, with the same CIs.
Negative differences favour the candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from head_to_head_report import ENERGY, family_predictions, guarded_blend, load_production, paired_bootstrap, read_csv_columns


def diff_row(err_base: np.ndarray, err_cand: np.ndarray) -> dict:
    return {"n": int(err_base.size), "base": float(np.abs(err_base).mean()), "candidate": float(np.abs(err_cand).mean()),
            "candidate_minus_base": float(np.abs(err_cand).mean() - np.abs(err_base).mean()),
            "ci95": paired_bootstrap(err_cand, err_base)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="configs/production.yaml")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    cols = [f"prediction_{c}" for c in ENERGY]
    specs = {"base": load_production(args.base), "candidate": load_production(args.candidate)}
    weights = {name: [float(f["weight"]) for f in fams] for name, fams in specs.items()}
    report = {"base": args.base, "candidate": args.candidate,
              "weights": {name: {f["family"]: float(f["weight"]) for f in fams} for name, fams in specs.items()}}

    test = {name: [family_predictions(f, "test", cols) for f in fams] for name, fams in specs.items()}
    first = specs["base"][0]
    targets = read_csv_columns(first["test"].format(seed=first["seeds"][0]), [f"target_{c}" for c in ENERGY])
    keys = sorted(set(targets).intersection(*(set(d) for groups in test.values() for g in groups for d in g)))
    target = np.array([targets[k] for k in keys])
    blend = {name: guarded_blend(test[name], weights[name], keys)[0] for name in specs}
    report["qm9gwbse_test"] = {
        "S_agg": diff_row(np.abs(blend["base"] - target).mean(axis=1), np.abs(blend["candidate"] - target).mean(axis=1)),
        "S1": diff_row(blend["base"][:, 0] - target[:, 0], blend["candidate"][:, 0] - target[:, 0])}

    with open(args.labels, newline="", encoding="utf-8") as stream:
        labels = [r for r in csv.DictReader(stream) if r["set"] != "QM9GWBSE_control"]
    ext = {name: [family_predictions(f, "external", cols) for f in fams] for name, fams in specs.items()}
    subsets = {f"{s}/all": [r for r in labels if r["set"] == s] for s in sorted({r["set"] for r in labels})}
    oe62h = [r for r in labels if r["set"] == "OE62H"]
    for element in sorted({e for r in oe62h for e in r["elements"].split("-")} - set("HCNOF")):
        subsets[f"OE62H/{element}"] = [r for r in oe62h if element in r["elements"].split("-")]
    report["external_S1"] = {}
    for label, rows in subsets.items():
        ks = [r["molecule_key"] for r in rows]
        t = np.array([float(r["S1_eV"]) for r in rows])
        pred = {name: guarded_blend(ext[name], weights[name], ks)[0][:, 0] for name in specs}
        report["external_S1"][label] = diff_row(pred["base"] - t, pred["candidate"] - t)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label, row in [*(("QM9GWBSE test " + k, v) for k, v in report["qm9gwbse_test"].items()), *report["external_S1"].items()]:
        ci = row["ci95"]
        print(f"{label:24s} n={row['n']:6d} base {row['base']:.5f} cand {row['candidate']:.5f} "
              f"diff {row['candidate_minus_base']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]")


if __name__ == "__main__":
    main()
