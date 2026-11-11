#!/usr/bin/env python3
"""Does training on cheap geometries recover the accuracy lost on them? (TASK-310, follows KNOW-308)

Compares single models on the wave E geometry pools (data/processed/cheap_geometry/pool_<source>.npz:
the same 1,973 QM9GWBSE test molecules as B3LYP, ETKDG+MMFF, GFN2-xTB and UMA-relaxed geometries).
--model name=pattern gives each model's pool predictions, with {source} in the path. For every source
and model: S-agg and per-state MAE over sorted S1-S5 against the QM9GWBSE labels, the signed S1 error,
and the paired-bootstrap difference against --reference on the same source. The headline row compares
each model on each source with the reference model on B3LYP geometries (the accuracy a lab user would
get from a SMILES string vs the dataset geometry). --summary name=path adds full-test-split S-agg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_to_head_report import ENERGY, paired_bootstrap, read_csv_columns


def named(items: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--model", action="append", required=True, help="name=path pattern with {source}")
    parser.add_argument("--reference", required=True, help="a --model name")
    parser.add_argument("--sources", nargs="+", default=["b3lyp", "etkdg", "xtb", "uma"])
    parser.add_argument("--summary", action="append", default=[], help="name=training summary JSON")
    parser.add_argument("--report", default="runs/geometry_training_report.json")
    args = parser.parse_args()
    models = named(args.model)
    if args.reference not in models:
        raise SystemExit(f"--reference {args.reference!r} is not a --model name")
    q = np.load(args.packed, allow_pickle=False)
    cols = [str(c) for c in q["target_columns"]]
    labels = dict(zip([str(k) for k in q["molecule_key"]], q["targets"][:, [cols.index(c) for c in ENERGY]].astype(float)))
    pred_cols = [f"prediction_{c}" for c in ENERGY]

    preds = {}
    for name, pattern in models.items():
        for source in args.sources:
            path = Path(pattern.format(source=source))
            if path.exists():
                preds[(name, source)] = read_csv_columns(path, pred_cols)
            else:
                print(f"missing {path}; skipped")
    keys = sorted(set(labels).intersection(*(set(p) for p in preds.values())))
    target = np.array([labels[k] for k in keys])
    errors = {ms: np.sort(np.stack([p[k] for k in keys]), axis=1) - target for ms, p in preds.items()}
    report = {"molecules": len(keys), "reference": args.reference, "test_split_S_agg": {}, "pools": {}}
    for name, path in named(args.summary).items():
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        report["test_split_S_agg"][name] = {"S_agg": (summary.get("test_metrics") or {}).get("energy_eV_mae"),
                                            "test_molecules": (summary.get("split_sizes") or {}).get("test")}
    anchor = errors.get((args.reference, "b3lyp"))
    for (name, source), err in errors.items():
        per_mol = np.abs(err).mean(axis=1)
        row = {"S_agg": float(np.abs(err).mean()), "per_state": np.abs(err).mean(axis=0).round(5).tolist(),
               "S1_signed": float(err[:, 0].mean())}
        ref = errors.get((args.reference, source))
        if name != args.reference and ref is not None:
            row["minus_reference_same_source"] = float(per_mol.mean() - np.abs(ref).mean(axis=1).mean())
            row["minus_reference_same_source_ci95"] = paired_bootstrap(per_mol, np.abs(ref).mean(axis=1))
        if anchor is not None and (name, source) != (args.reference, "b3lyp"):
            row["minus_reference_on_b3lyp"] = float(per_mol.mean() - np.abs(anchor).mean(axis=1).mean())
            row["minus_reference_on_b3lyp_ci95"] = paired_bootstrap(per_mol, np.abs(anchor).mean(axis=1))
        report["pools"].setdefault(source, {})[name] = row
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'source':8s} " + " ".join(f"{n:>24s}" for n in models))
    for source, rows in report["pools"].items():
        print(f"{source:8s} " + " ".join(f"{rows[n]['S_agg']:>10.4f} ({rows[n]['S1_signed']:+.3f})      " if n in rows else f"{'-':>24s}"
                                         for n in models))
    print("test split:", report["test_split_S_agg"])


if __name__ == "__main__":
    main()
