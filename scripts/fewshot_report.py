#!/usr/bin/env python3
"""Few-shot OE62H GW labels: held-out error against the number of shots (TASK-310, bet B).

Reads <out-dir>/folds.json (scripts/build_oe62h_shots.py) and each run's external-pool predictions
<run-dir>/ext_f{f}_k{k}.csv. For every k, each OE62H molecule is scored by the model of the fold that
held it out, so all OE62H molecules are covered once and none was a shot for its own prediction.
k=0 is the same fine-tuning without shots (<run-dir>/ext_k0.csv), and "base" is the un-fine-tuned
model (--base). Reports S1 MAE, signed S1, sorted S1-S5 S-agg, per-element S1 MAE outside CHONF, the
paired-bootstrap difference against k=0, and, per k, the mean over folds of each run's S1 MAE on the
other external sets and of its QM9GWBSE test S-agg (forgetting check).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from head_to_head_report import ENERGY, paired_bootstrap, read_csv_columns

PRED = [f"prediction_{c}" for c in ENERGY]


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    err = np.sort(pred, axis=1) - target
    return {"S1_mae_eV": float(np.abs(err[:, 0]).mean()), "S1_signed_eV": float(err[:, 0].mean()),
            "S_agg_eV": float(np.abs(err).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--folds", default="data/processed/oe62h_fewshot/folds.json")
    parser.add_argument("--run-dir", default="runs/fewshot_oe62h")
    parser.add_argument("--base", default="runs/head_to_head/ext_uma_richind_he_heads_s17.csv")
    parser.add_argument("--report", default="runs/fewshot_oe62h_report.json")
    args = parser.parse_args()
    manifest = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    with open(args.labels, newline="", encoding="utf-8") as stream:
        labels = {r["molecule_key"]: r for r in csv.DictReader(stream)}
    run_dir = Path(args.run_dir)
    keys = sorted(k for fold in manifest["folds"] for k in fold["held_out"])
    target = np.array([[float(labels[k][c]) for c in ENERGY] for k in keys])
    elements = {k: set(labels[k]["elements"].split("-")) - set("HCNOF") for k in keys}

    arms: dict[str, np.ndarray] = {}
    base = read_csv_columns(args.base, PRED)
    arms["base"] = np.array([base[k] for k in keys])
    control = run_dir / "ext_k0.csv"
    if control.exists():
        k0 = read_csv_columns(control, PRED)
        arms["k0"] = np.array([k0[k] for k in keys])
    runs_by_k: dict[int, list[Path]] = {}
    for k in manifest["ks"]:
        assembled, paths = {}, []
        for f, fold in enumerate(manifest["folds"]):
            path = run_dir / f"ext_f{f}_k{k}.csv"
            if not path.exists():
                break
            paths.append(path)
            pred = read_csv_columns(path, PRED)
            assembled.update({key: pred[key] for key in fold["held_out"]})
        if len(paths) == len(manifest["folds"]):
            arms[f"k{k}"] = np.array([assembled[key] for key in keys])
            runs_by_k[k] = paths
        else:
            print(f"k={k}: {len(paths)} of {len(manifest['folds'])} folds present; skipped")

    reference = "k0" if "k0" in arms else "base"
    ref_err = np.abs(np.sort(arms[reference], axis=1)[:, 0] - target[:, 0])
    report = {"set": manifest["set"], "molecules": len(keys), "reference": reference, "arms": {}}
    for name, pred in arms.items():
        row = metrics(pred, target)
        err = np.abs(np.sort(pred, axis=1)[:, 0] - target[:, 0])
        if name != reference:
            row["S1_minus_reference_eV"] = float(err.mean() - ref_err.mean())
            row["S1_minus_reference_ci95"] = paired_bootstrap(err, ref_err)
        row["per_element_S1_mae_eV"] = {e: {"n": int(m.sum()), "mae": float(err[m].mean())}
                                       for e in sorted(set().union(*elements.values()))
                                       for m in [np.array([e in elements[k] for k in keys])] if m.any()}
        report["arms"][name] = row

    # Forgetting check: each run's full-set S1 MAE on the other external sets and its QM9GWBSE test S-agg.
    others = {s: [k for k, r in labels.items() if r["set"] == s] for s in ("QM9", "PC9", "OE62L")}
    paths_by_arm = {"k0": [control]} if control.exists() else {}
    paths_by_arm.update({f"k{k}": paths for k, paths in runs_by_k.items()})
    report["other_sets"] = {}
    for name, paths in paths_by_arm.items():
        row = {}
        for set_name, set_keys in others.items():
            t = np.array([float(labels[k]["S1_eV"]) for k in set_keys])
            maes = [float(np.abs(np.sort(np.stack([p[k] for k in set_keys]), axis=1)[:, 0] - t).mean())
                    for p in (read_csv_columns(path, PRED) for path in paths)]
            row[f"{set_name}_S1_mae_eV"] = float(np.mean(maes))
        sagg = []
        for path in paths:
            summary = path.with_name(path.name.replace("ext_", "").replace(".csv", ".summary.json"))
            if summary.exists():
                sagg.append((json.loads(summary.read_text(encoding="utf-8")).get("test_metrics") or {}).get("energy_eV_mae"))
        row["qm9gwbse_test_S_agg"] = float(np.mean(sagg)) if sagg else None
        report["other_sets"][name] = row
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, row in report["arms"].items():
        ci = row.get("S1_minus_reference_ci95")
        print(f"{name:5s} S1 {row['S1_mae_eV']:.4f} ({row['S1_signed_eV']:+.3f})  S-agg {row['S_agg_eV']:.4f}"
              + (f"  vs {reference} {row['S1_minus_reference_eV']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""))
    print(json.dumps(report["other_sets"], indent=1))


if __name__ == "__main__":
    main()
