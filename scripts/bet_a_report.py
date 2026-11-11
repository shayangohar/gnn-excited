#!/usr/bin/env python3
"""Bet A R0/R1 vs the MLP control on frozen UMA features (TASK-277 / TASK-310).

Reads <run-dir>/<features>_<kind>_s<seed>.summary.json and the matching test predictions from
scripts/train_frozen_decoder.py. For each feature set and decoder: test metrics as mean and SD over
seeds, and the seed-ensemble (mean of the seeds' predictions, sorted) S-agg. Each decoder's seed
ensemble is compared with the mlp control's on the same features by a paired bootstrap over molecules.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from head_to_head_report import ENERGY, paired_bootstrap, read_csv_columns

METRICS = ("S_agg_sorted", "near_degenerate_state_mae", "near_degenerate_gap_mae", "gap_mae", "adjacent_inversion_rate",
           "log1p_osc_mae")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="runs/bet_a")
    parser.add_argument("--report", default="runs/bet_a_report.json")
    args = parser.parse_args()
    runs = defaultdict(list)
    for path in sorted(Path(args.run_dir).glob("*.summary.json")):
        features, kind, seed = path.name[: -len(".summary.json")].rsplit("_", 2)
        runs[(features, kind)].append((int(seed[1:]), json.loads(path.read_text(encoding="utf-8")), path))
    report: dict = {"groups": {}, "ensembles": {}}
    ensembles = {}
    for (features, kind), items in sorted(runs.items()):
        tests = [s["test"] for _, s, _ in items]
        row = {"seeds": [seed for seed, _, _ in items]}
        for m in METRICS:
            values = [t[m] for t in tests if t.get(m) is not None]
            if values:
                row[m] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        row["per_state_mean"] = np.mean([t["per_state"] for t in tests], axis=0).round(5).tolist()
        report["groups"][f"{features}/{kind}"] = row
        preds = [read_csv_columns(str(p).replace(".summary.json", ".test_predictions.csv"),
                                  [f"prediction_{c}" for c in ENERGY]) for _, _, p in items]
        target = read_csv_columns(str(items[0][2]).replace(".summary.json", ".test_predictions.csv"), [f"target_{c}" for c in ENERGY])
        keys = sorted(target)
        mean_pred = np.sort(np.mean([[p[k] for k in keys] for p in preds], axis=0), axis=1)
        ensembles[(features, kind)] = (keys, np.abs(mean_pred - np.array([target[k] for k in keys])).mean(axis=1))
    for (features, kind), (keys, err) in ensembles.items():
        row = {"S_agg_sorted": float(err.mean()), "seeds": len(runs[(features, kind)])}
        control = ensembles.get((features, "mlp"))
        if kind != "mlp" and control is not None and control[0] == keys:
            row["minus_mlp"] = float(err.mean() - control[1].mean())
            row["minus_mlp_ci95"] = paired_bootstrap(err, control[1])
        report["ensembles"][f"{features}/{kind}"] = row
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, row in report["groups"].items():
        ens = report["ensembles"][name]
        extra = f"  ens {ens['S_agg_sorted']:.5f}" + (f" vs mlp {ens['minus_mlp']:+.5f} {np.round(ens['minus_mlp_ci95'], 5).tolist()}"
                                                     if "minus_mlp" in ens else "")
        print(f"{name:28s} S-agg {row['S_agg_sorted']['mean']:.5f} ± {row['S_agg_sorted']['sd']:.5f}{extra}")


if __name__ == "__main__":
    main()
