#!/usr/bin/env python3
"""Per-set and per-element S1 error of model variants on the external head-to-head pool (TASK-287).

Each --pred is name=path to a pool prediction CSV (molecule_key, prediction_S1_eV..prediction_S5_eV;
energies are sorted per molecule). For each external set (QM9, PC9, OE62L, OE62H) the report gives S1
MAE and mean signed error, and for OE62H the same per element outside CHONF. Every variant is compared
with --reference by a paired bootstrap (negative favours the variant). --summary name=path adds each
run's QM9GWBSE test energy_eV_mae, so a transfer gain can be checked against in-distribution cost.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from head_to_head_report import paired_bootstrap, read_csv_columns

ENERGY = [f"S{i}_eV" for i in range(1, 6)]
SETS = ("QM9", "PC9", "OE62L", "OE62H")


def named_paths(items: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items)


def group_rows(labels: list[dict], set_name: str) -> dict[str, list[dict]]:
    """'all' plus, for OE62H, one group per element outside CHONF."""
    rows = [r for r in labels if r["set"] == set_name]
    groups = {"all": rows}
    if set_name == "OE62H":
        for element in sorted({e for r in rows for e in r["elements"].split("-")} - set("HCNOF")):
            groups[element] = [r for r in rows if element in r["elements"].split("-")]
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--pred", action="append", required=True, help="name=path, repeatable")
    parser.add_argument("--reference", required=True, help="a --pred name")
    parser.add_argument("--summary", action="append", default=[], help="name=path to a training summary JSON")
    parser.add_argument("--report", default="runs/heavy_element_report.json")
    args = parser.parse_args()
    with open(args.labels, newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    cols = [f"prediction_{c}" for c in ENERGY]
    preds = {name: read_csv_columns(path, cols) for name, path in named_paths(args.pred).items()}
    if args.reference not in preds:
        raise SystemExit(f"--reference {args.reference!r} is not a --pred name")

    report: dict = {"reference": args.reference, "qm9gwbse_test_energy_eV_mae": {}, "external": {}}
    for name, path in named_paths(args.summary).items():
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        report["qm9gwbse_test_energy_eV_mae"][name] = (summary.get("test_metrics") or {}).get("energy_eV_mae")
    for set_name in SETS:
        for group, rows in group_rows(labels, set_name).items():
            keys = [r["molecule_key"] for r in rows if all(r["molecule_key"] in p for p in preds.values())]
            target = np.array([float(next(r for r in rows if r["molecule_key"] == k)["S1_eV"]) for k in keys])
            errors = {name: np.sort(np.stack([p[k] for k in keys]), axis=1)[:, 0] - target for name, p in preds.items()}
            row = {"n": len(keys)}
            for name, err in errors.items():
                row[name] = {"S1_mae_eV": float(np.abs(err).mean()), "S1_signed_eV": float(err.mean())}
                if name != args.reference:
                    row[name]["minus_reference_eV"] = row[name]["S1_mae_eV"] - float(np.abs(errors[args.reference]).mean())
                    row[name]["minus_reference_ci95"] = paired_bootstrap(err, errors[args.reference])
            report["external"][f"{set_name}/{group}"] = row
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    names = list(preds)
    print("QM9GWBSE test energy MAE:", report["qm9gwbse_test_energy_eV_mae"])
    print(f"{'subset':14s} {'n':>4s} " + " ".join(f"{n[:22]:>22s}" for n in names))
    for subset, row in report["external"].items():
        cells = [f"{row[n]['S1_mae_eV']:.3f} ({row[n]['S1_signed_eV']:+.2f})" for n in names]
        print(f"{subset:14s} {row['n']:4d} " + " ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
