#!/usr/bin/env python3
"""Apply val-chosen blend weights to external-pool predictions (TASK-291).

Reads a blend spec (members with val/test CSVs plus a `pool` CSV from the external head-to-head pool;
groups; blends) and the scripts/blend_predictions.py report for it. For each blend in the report it
averages each group's pool CSVs, combines the groups with the report's val-chosen weights through
head_to_head_report.guarded_blend (a family that fails on a molecule is dropped for that molecule),
and writes <out-dir>/ext_blend_<name1>+<name2>....csv with sorted prediction_S1_eV..S5 columns, ready
for scripts/element_bias_report.py. Blends with a missing pool file are skipped with a note.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from head_to_head_report import ENERGY, guarded_blend, read_csv_columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--report", required=True, help="blend_predictions.py output for --spec")
    parser.add_argument("--out-dir", default="runs/head_to_head")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    pool = {name: member.get("pool") for name, member in spec["members"].items()}
    groups = spec.get("groups") or {}
    cols = [f"prediction_{c}" for c in ENERGY]
    for blend in report["blends"]:
        names, weights = blend["members"], blend["val_chosen_weights"]
        paths = [[pool.get(m) for m in groups.get(name, [name])] for name in names]
        missing = [p for family in paths for p in family if not p or not Path(p).exists()]
        if missing:
            print(f"skip {'+'.join(names)}: missing pool predictions {missing}")
            continue
        families = [[read_csv_columns(p, cols) for p in family] for family in paths]
        keys = sorted(set.intersection(*(set(d) for family in families for d in family)))
        pred, guarded = guarded_blend(families, weights, keys)
        out = Path(args.out_dir) / f"ext_blend_{'+'.join(names)}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["molecule_key", *cols])
            writer.writerows([k, *(f"{v:.6f}" for v in row)] for k, row in zip(keys, pred))
        print(f"{out}: {len(keys)} molecules, weights {weights}, guarded {guarded}")


if __name__ == "__main__":
    main()
