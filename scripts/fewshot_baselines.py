#!/usr/bin/env python3
"""Trivial recalibration baselines for the few-shot OE62H result (TASK-328; KNOW-323, DOC-327 D1).

Wave F (KNOW-314) fine-tuned he_heads_s17 with k OE62H GW-BSE labels and cut held-out OE62H S1 MAE from
0.241 eV (k0, the same 2-epoch fine-tune without shots) to 0.165 (k25) and 0.099 (k200). At k0 the signed
S1 error is +0.206 eV, so much of the error is bias. This asks how much of the gain a recalibration of the
k0 predictions recovers, fitted on the SAME k shots of each wave F fold and applied to that fold's held-out
molecules (cross-fitted, so every OE62H molecule is scored once and never by a model that saw it). Arms:

  k0                no correction
  shift_global      one constant added to all five singlet energies              (1 parameter)
  shift_per_state   one constant per state                                       (5)
  affine_per_state  y_n = a_n * pred_n + b_n                                     (10)
  element_offsets   per state, ridge of the residual on an intercept and one indicator per non-CHONF
                    element (alpha by leave-one-out CV on the shots)             (about 5 x 10)
  feature_ridge     per state, ridge of the residual on frozen pooled (mean, std) he_heads_s17 features
                    (scripts/extract_uma_features.py on the external pool; LOO-CV alpha)
  finetune          the wave F k-shot fine-tune (runs/fewshot_oe62h/ext_f{f}_k{k}.csv)

Every corrected prediction is re-sorted (S1 <= ... <= S5). Reports S1 MAE, signed S1, S-agg and per-element S1
MAE per arm and k, with paired-bootstrap CIs of each arm against the fine-tune and against k0. Also applies
the OE62H-fitted global and per-state shifts to OE62L (mean over folds) next to the fine-tune's OE62L error,
to test whether the fine-tune's OE62L gain is the same offset.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from head_to_head_report import ENERGY, paired_bootstrap, read_csv_columns

PRED = [f"prediction_{c}" for c in ENERGY]
ARMS = ("k0", "shift_global", "shift_per_state", "affine_per_state", "element_offsets", "feature_ridge", "finetune")


def pooled_features(prefix: str) -> dict[str, np.ndarray]:
    """molecule_key -> concatenated per-molecule mean and std of the per-atom features."""
    meta = np.load(f"{prefix}.meta.npz", allow_pickle=False)
    feats = np.load(f"{prefix}.features.npy", mmap_mode="r")
    off = meta["atom_offsets"]
    out = {}
    for i, key in enumerate(meta["molecule_key"]):
        block = np.asarray(feats[off[i]:off[i + 1]], dtype=np.float64)
        out[str(key)] = np.concatenate([block.mean(axis=0), block.std(axis=0)])
    return out


def ridge_cv(x: np.ndarray, y: np.ndarray, alphas: np.ndarray):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, alpha_per_target=True)).fit(x, y)


def fit_predict(arm: str, p_shot, y_shot, p_out, x_shot=None, x_out=None) -> np.ndarray:
    resid = y_shot - p_shot
    if arm == "k0":
        out = p_out
    elif arm == "shift_global":
        out = p_out + resid.mean()
    elif arm == "shift_per_state":
        out = p_out + resid.mean(axis=0)
    elif arm == "affine_per_state":
        out = np.column_stack([np.polyval(np.polyfit(p_shot[:, n], y_shot[:, n], 1), p_out[:, n]) for n in range(p_out.shape[1])])
    elif arm in ("element_offsets", "feature_ridge"):
        model = ridge_cv(x_shot, resid, np.logspace(-3, 5, 17))
        out = p_out + model.predict(x_out)
    else:
        raise ValueError(arm)
    return np.sort(out, axis=1)


def summarize(pred: np.ndarray, target: np.ndarray, elements: list[set[str]]) -> dict:
    err = pred - target
    row = {"S1_mae_eV": float(np.abs(err[:, 0]).mean()), "S1_signed_eV": float(err[:, 0].mean()),
           "S_agg_eV": float(np.abs(err).mean())}
    row["per_element_S1_mae_eV"] = {e: float(np.abs(err[m, 0]).mean())
                                   for e in sorted(set().union(*elements)) for m in [np.array([e in s for s in elements])] if m.sum() >= 5}
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--folds", default="data/processed/oe62h_fewshot/folds.json")
    parser.add_argument("--run-dir", default="runs/fewshot_oe62h")
    parser.add_argument("--features", default="data/processed/uma_features/pool_he_heads_s17",
                        help="prefix of scripts/extract_uma_features.py output on the external pool")
    parser.add_argument("--report", default="runs/fewshot_baselines_report.json")
    args = parser.parse_args()
    with open(args.labels, newline="", encoding="utf-8") as stream:
        labels = {r["molecule_key"]: r for r in csv.DictReader(stream)}
    manifest = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    k0 = read_csv_columns(run_dir / "ext_k0.csv", PRED)
    feats = pooled_features(args.features) if Path(f"{args.features}.meta.npz").exists() else None
    heavy = {k: set(r["elements"].split("-")) - set("HCNOF") for k, r in labels.items() if r["set"] == "OE62H"}
    element_cols = sorted(set().union(*heavy.values()))

    def target(keys):
        return np.array([[float(labels[k][c]) for c in ENERGY] for k in keys])

    def p0(keys):
        return np.sort(np.array([k0[k] for k in keys]), axis=1)

    def x_for(arm, keys):
        if arm == "element_offsets":
            return np.array([[float(e in heavy[k]) for e in element_cols] for k in keys])
        if arm == "feature_ridge":
            return np.array([feats[k] for k in keys]) if feats is not None else None
        return None

    keys_all = sorted(k for fold in manifest["folds"] for k in fold["held_out"])
    y_all = target(keys_all)
    report = {"molecules": len(keys_all), "features": args.features if feats is not None else None, "k": {}}
    l_keys = sorted(k for k, r in labels.items() if r["set"] == "OE62L")
    y_l, p0_l = target(l_keys), p0(l_keys)
    for k in manifest["ks"]:
        preds = {arm: {} for arm in ARMS}
        transfer = {"shift_global": [], "shift_per_state": [], "finetune": []}
        for f, fold in enumerate(manifest["folds"]):
            shots, held = fold["shots"][str(k)], fold["held_out"]
            ft = read_csv_columns(run_dir / f"ext_f{f}_k{k}.csv", PRED)
            for key in held:
                preds["finetune"][key] = np.sort(ft[key])
            for arm in ARMS[:-1]:
                if arm == "feature_ridge" and feats is None:
                    continue
                out = fit_predict(arm, p0(shots), target(shots), p0(held), x_for(arm, shots), x_for(arm, held))
                preds[arm].update(dict(zip(held, out)))
            for arm in ("shift_global", "shift_per_state"):
                out = fit_predict(arm, p0(shots), target(shots), p0_l)
                transfer[arm].append(float(np.abs(out[:, 0] - y_l[:, 0]).mean()))
            transfer["finetune"].append(float(np.abs(np.sort(np.array([ft[x] for x in l_keys]), axis=1)[:, 0] - y_l[:, 0]).mean()))
        rows = {}
        ft_err = np.abs(np.array([preds["finetune"][x] for x in keys_all])[:, 0] - y_all[:, 0])
        k0_err = np.abs(np.array([preds["k0"][x] for x in keys_all])[:, 0] - y_all[:, 0])
        for arm in ARMS:
            if not preds[arm]:
                continue
            pred = np.array([preds[arm][x] for x in keys_all])
            row = summarize(pred, y_all, [heavy[x] for x in keys_all])
            err = np.abs(pred[:, 0] - y_all[:, 0])
            if arm != "finetune":
                row["minus_finetune_ci95"] = paired_bootstrap(err, ft_err)
            if arm != "k0":
                row["minus_k0_ci95"] = paired_bootstrap(err, k0_err)
            rows[arm] = row
        rows["OE62L_S1_mae_mean_over_folds"] = {"k0": float(np.abs(p0_l[:, 0] - y_l[:, 0]).mean()),
                                               **{a: float(np.mean(v)) for a, v in transfer.items()}}
        report["k"][str(k)] = rows
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for k, rows in report["k"].items():
        print(f"k={k}")
        for arm in ARMS:
            if arm in rows:
                r = rows[arm]
                ci = r.get("minus_finetune_ci95")
                print(f"  {arm:17s} S1 {r['S1_mae_eV']:.4f} ({r['S1_signed_eV']:+.3f})  S-agg {r['S_agg_eV']:.4f}"
                      + (f"  vs finetune [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""))
        print("  OE62L S1:", {a: round(v, 4) for a, v in rows["OE62L_S1_mae_mean_over_folds"].items()})


if __name__ == "__main__":
    main()
