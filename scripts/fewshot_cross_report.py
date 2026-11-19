#!/usr/bin/env python3
"""Few-shot cross-tests: is the OE62H few-shot gain element chemistry or a dataset-level offset? (TASK-320)

Wave F (KNOW-314) showed that a few OE62H GW-BSE labels cut OE62H S1 error from 0.241 to 0.099 eV. But
OE62L improved too. Two cross-tests built by scripts/build_oe62h_shots.py, all scored on S1 against the
wave F k0 control (the same 2-epoch fine-tune without shots) on the same molecules, with paired CIs:

  L->H   shots from OE62L only (5 folds over OE62L, tag oe62l_). OE62H is never seen, so each fold model
         scores all 574 OE62H molecules (reported per run and for the 5-run mean prediction), next to the
         wave F OE62H-shot result at the same k (cross-fitted). OE62L is scored cross-fitted.
  noS    k shots drawn only from S-free OE62H molecules (independent draws, tag nos_). Every S molecule is
         unseen. Their S1 error is compared with k0 and with the wave F k-shot cross-fitted predictions,
         whose shots included S molecules.
If L->H shots fix OE62H about as well as OE62H shots, the gain is mostly a dataset-level offset. If the S
molecules improve without S shots, the S fix is not S-specific.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from head_to_head_report import ENERGY, paired_bootstrap, read_csv_columns

PRED = [f"prediction_{c}" for c in ENERGY]


def s1(pred: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    return np.array([np.sort(pred[k])[0] for k in keys])


def compare(err: np.ndarray, ref: np.ndarray) -> dict:
    return {"S1_mae_eV": float(np.abs(err).mean()), "S1_signed_eV": float(err.mean()),
            "minus_k0_eV": float(np.abs(err).mean() - np.abs(ref).mean()), "minus_k0_ci95": paired_bootstrap(err, ref)}


def cross_fitted(manifest: dict, run_dir: Path, name: str, k: int) -> dict[str, np.ndarray] | None:
    """Each held-out molecule predicted by the run that held it out; None if a run is missing."""
    out = {}
    for u, fold in enumerate(manifest["folds"]):
        path = run_dir / f"ext_{name.format(u=u, k=k)}.csv"
        if not path.exists():
            return None
        pred = read_csv_columns(path, PRED)
        out.update({key: pred[key] for key in fold["held_out"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--k0", default="runs/fewshot_oe62h/ext_k0.csv")
    parser.add_argument("--oe62h-folds", default="data/processed/oe62h_fewshot/folds.json")
    parser.add_argument("--oe62h-run-dir", default="runs/fewshot_oe62h")
    parser.add_argument("--oe62l-folds", default="data/processed/oe62l_fewshot/folds.json")
    parser.add_argument("--nos-folds", default="data/processed/oe62h_nos_fewshot/folds.json")
    parser.add_argument("--run-dir", default="runs/fewshot_wave_g")
    parser.add_argument("--report", default="runs/fewshot_cross_report.json")
    args = parser.parse_args()
    with open(args.labels, newline="", encoding="utf-8") as stream:
        labels = {r["molecule_key"]: r for r in csv.DictReader(stream)}
    t1 = {k: float(r["S1_eV"]) for k, r in labels.items() if r["S1_eV"]}
    by_set = {s: sorted(k for k, r in labels.items() if r["set"] == s and k in t1) for s in ("OE62H", "OE62L", "PC9", "QM9")}
    heavy = {k: set(labels[k]["elements"].split("-")) - set("HCNOF") for k in by_set["OE62H"]}
    k0 = read_csv_columns(args.k0, PRED)
    run_dir, h_dir = Path(args.run_dir), Path(args.oe62h_run_dir)
    h_manifest = json.loads(Path(args.oe62h_folds).read_text(encoding="utf-8"))
    report: dict = {"k0": args.k0}

    def err(pred, keys):
        return s1(pred, keys) - np.array([t1[k] for k in keys])

    # L->H: OE62L shots.
    l_manifest = json.loads(Path(args.oe62l_folds).read_text(encoding="utf-8"))
    tag = l_manifest.get("tag", "oe62l_")
    report["L_to_H"] = {}
    for k in l_manifest["ks"]:
        paths = [run_dir / f"ext_{tag}f{u}_k{k}.csv" for u in range(len(l_manifest["folds"]))]
        if not all(p.exists() for p in paths):
            print(f"L->H k={k}: runs missing; skipped")
            continue
        preds = [read_csv_columns(p, PRED) for p in paths]
        h_keys = by_set["OE62H"]
        mean_pred = {key: np.mean([p[key] for p in preds], axis=0) for key in h_keys}
        ref_h = err(k0, h_keys)
        row = {"OE62H_per_run_S1_mae_eV": [float(np.abs(err(p, h_keys)).mean()) for p in preds],
               "OE62H_mean_prediction": compare(err(mean_pred, h_keys), ref_h)}
        row["OE62H_per_element_S1_mae_eV"] = {
            e: {"n": int(m.sum()), "k0": float(np.abs(ref_h[m]).mean()), "L_to_H": float(np.abs(err(mean_pred, h_keys)[m]).mean())}
            for e in sorted(set().union(*heavy.values())) for m in [np.array([e in heavy[x] for x in h_keys])] if m.sum() >= 5}
        h_shots = cross_fitted(h_manifest, h_dir, "f{u}_k{k}", k)
        if h_shots is not None:
            row["OE62H_with_OE62H_shots_same_k"] = compare(err(h_shots, h_keys), ref_h)
            row["L_to_H_minus_H_shots_ci95"] = paired_bootstrap(err(mean_pred, h_keys), err(h_shots, h_keys))
        l_cf = cross_fitted(l_manifest, run_dir, tag + "f{u}_k{k}", k)
        l_keys = sorted(key for fold in l_manifest["folds"] for key in fold["held_out"])
        row["OE62L_cross_fitted"] = compare(err(l_cf, l_keys), err(k0, l_keys))
        for s in ("PC9", "QM9"):
            row[f"{s}_mean_over_runs_S1_mae_eV"] = float(np.mean([np.abs(err(p, by_set[s])).mean() for p in preds]))
        report["L_to_H"][f"k{k}"] = row

    # noS: S-free OE62H shots; score the S molecules.
    n_manifest = json.loads(Path(args.nos_folds).read_text(encoding="utf-8"))
    tag, element = n_manifest.get("tag", "nos_"), n_manifest.get("exclude_element", "S")
    with_e = [k for k in by_set["OE62H"] if element in heavy[k]]
    report["noS"] = {"element": element, "molecules_with_element": len(with_e)}
    for k in n_manifest["ks"]:
        paths = [run_dir / f"ext_{tag}d{u}_k{k}.csv" for u in range(len(n_manifest["folds"]))]
        if not all(p.exists() for p in paths):
            print(f"noS k={k}: runs missing; skipped")
            continue
        preds = [read_csv_columns(p, PRED) for p in paths]
        mean_pred = {key: np.mean([p[key] for p in preds], axis=0) for key in with_e}
        ref = err(k0, with_e)
        row = {"per_draw_S1_mae_eV": [float(np.abs(err(p, with_e)).mean()) for p in preds],
               "mean_prediction": compare(err(mean_pred, with_e), ref)}
        h_shots = cross_fitted(h_manifest, h_dir, "f{u}_k{k}", k)
        if h_shots is not None:
            row["with_element_shots_same_k"] = compare(err(h_shots, with_e), ref)
            row["noS_minus_with_element_shots_ci95"] = paired_bootstrap(err(mean_pred, with_e), err(h_shots, with_e))
        free_held = [sorted(x for x in fold["held_out"] if element not in heavy[x]) for fold in n_manifest["folds"]]
        row["element_free_held_out_S1_mae_eV"] = [float(np.abs(err(p, ks)).mean()) for p, ks in zip(preds, free_held)]
        row["element_free_held_out_k0_S1_mae_eV"] = [float(np.abs(err(k0, ks)).mean()) for ks in free_held]
        report["noS"][f"k{k}"] = row

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1)[:6000])


if __name__ == "__main__":
    main()
