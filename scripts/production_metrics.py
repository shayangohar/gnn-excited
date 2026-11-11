#!/usr/bin/env python3
"""QM9GWBSE test metrics of the production ensemble in configs/production.yaml (TASK-310).

Singlets: the guarded weighted blend of seed-averaged families, each molecule's S1-S5 sorted.
Triplets: the same for T1-T5 from the `triplets` families. Reports S-agg / T-agg, per-state MAE and,
for every family alone, the same numbers, so a promotion can be checked against its members.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_to_head_report import ENERGY, family_predictions, guarded_blend, load_production, read_csv_columns

TRIPLET = [f"T{i}_eV" for i in range(1, 6)]


def part_metrics(families: list[dict], columns: list[str]) -> dict:
    preds = [family_predictions(f, "test", [f"prediction_{c}" for c in columns]) for f in families]
    targets = read_csv_columns(families[0]["test"].format(seed=families[0]["seeds"][0]), [f"target_{c}" for c in columns])
    keys = sorted(set(targets).intersection(*(set(d) for group in preds for d in group)))
    target = np.array([targets[k] for k in keys])
    out = {"n": len(keys)}
    blend, guarded = guarded_blend(preds, [float(f["weight"]) for f in families], keys)
    for name, pred in [("blend", blend), *((f["family"], guarded_blend([g], [1.0], keys)[0]) for f, g in zip(families, preds))]:
        err = np.abs(pred - target)
        out[name] = {"agg_mae_eV": float(np.nanmean(err)), "per_state": np.nanmean(err, axis=0).round(5).tolist()}
    out["blend"]["guarded_molecules"] = guarded
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--production", default="configs/production.yaml")
    parser.add_argument("--report", default="runs/production_metrics.json")
    args = parser.parse_args()
    report = {"production": args.production,
              "singlets": part_metrics(load_production(args.production, "singlets"), ENERGY),
              "triplets": part_metrics(load_production(args.production, "triplets"), TRIPLET)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
