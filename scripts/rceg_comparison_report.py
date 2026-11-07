#!/usr/bin/env python3
"""UMA gap models vs rCEG on rCEG's own 424-molecule held-out test split (KNOW-293).

Each --run is name=summary_json of a scripts/train_uma_qm9gwbse.py run on build_rceg_comparison.py data.
Its test predictions (label geometries: QM9 B3LYP or ETKDG+UFF) and its post_inference pool predictions
(ETKDG+UFF for every molecule, i.e. SMILES only) are both scored. rCEG's predictions come from
test_meta.csv. MAE is reported overall, per rCEG chemical domain and per reference source, with paired
bootstrap differences against rCEG (negative favours the UMA model).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from head_to_head_report import paired_bootstrap


def read_predictions(path: str | Path, column: str) -> dict[str, float]:
    with open(path, newline="", encoding="utf-8") as stream:
        return {row["molecule_key"]: float(row[column]) for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", default="data/processed/rceg_compare/test_meta.csv")
    parser.add_argument("--run", action="append", required=True, help="name=path/to/run.summary.json")
    parser.add_argument("--report", default="runs/rceg_comparison_report.json")
    args = parser.parse_args()
    with open(args.meta, newline="", encoding="utf-8") as stream:
        meta = {row["molecule_key"]: row for row in csv.DictReader(stream)}
    preds = {"rceg": {k: float(r["rceg_pred"]) for k, r in meta.items()}}
    for item in args.run:
        name, path = item.split("=", 1)
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        preds[f"{name}_label_geometry"] = read_predictions(summary["predictions_csv_path"], "prediction_gap_eV")
        if summary.get("post_inference"):
            preds[f"{name}_smiles_geometry"] = read_predictions(summary["post_inference"]["out_csv"], "prediction_gap_eV")
    keys = sorted(set(meta).intersection(*(set(p) for p in preds.values())))
    groups = {"all": keys}
    for field in ("domain", "ref_source"):
        for value in sorted({meta[k][field] for k in keys}):
            groups[f"{field}={value}"] = [k for k in keys if meta[k][field] == value]
    report = {"molecules": len(keys), "groups": {}}
    for group, ks in groups.items():
        target = np.array([float(meta[k]["ref_gap"]) for k in ks])
        errors = {name: np.array([p[k] for k in ks]) - target for name, p in preds.items()}
        row = {"n": len(ks)}
        for name, err in errors.items():
            row[name] = {"mae": float(np.abs(err).mean()), "signed": float(err.mean())}
            if name != "rceg":
                row[name]["minus_rceg"] = row[name]["mae"] - row["rceg"]["mae"]
                row[name]["minus_rceg_ci95"] = paired_bootstrap(err, errors["rceg"])
        report["groups"][group] = row
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    names = list(preds)
    print(f"{'group':48s} {'n':>4s} " + " ".join(f"{n[:26]:>26s}" for n in names))
    for group, row in report["groups"].items():
        print(f"{group:48s} {row['n']:4d} " + " ".join(f"{row[n]['mae']:26.3f}" for n in names))


if __name__ == "__main__":
    main()
