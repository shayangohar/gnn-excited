#!/usr/bin/env python3
"""Head-to-head S1 report: our production blend vs Baum et al. ViSNetGWBSE (TASK-271, DOC-248 L13).

Production (KNOW-261) is w_uma * mean(UMA richind x3) + w_visnet * mean(distilled ViSNet x3), with
each molecule's five energies sorted. Their checkpoints predict S1 only, so the comparison is S1.
Their best checkpoint is picked on their own (reproduced) val split, never on a test set.

  QM9GWBSE   our test split restricted to molecules outside their training set: "strict" = their
             test, "disjoint" = their val + test. Our side uses the stored test predictions.
  external   PC9, OE62L, OE62H and their QM9 set: all molecules, molecules outside both training
             sets, molecules with no QM9GWBSE overlap, and molecules with elements beyond CHONF;
             also our S1-S5 S-agg where five labels exist.

Differences carry paired-bootstrap 95% intervals (ours minus theirs; negative favours ours). The
QM9GWBSE control molecules check that pool inference reproduces the stored test predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ENERGY = [f"S{i}_eV" for i in range(1, 6)]
SEEDS = (17, 18, 19)


def read_csv_columns(path: str | Path, columns: list[str]) -> dict[str, np.ndarray]:
    """molecule_key -> float array of the named columns (blank -> NaN)."""
    with open(path, newline="", encoding="utf-8") as stream:
        return {row["molecule_key"]: np.array([float(row[c]) if row[c] != "" else np.nan for c in columns])
                for row in csv.DictReader(stream)}


def blend_sorted(groups: list[list[dict[str, np.ndarray]]], weights: list[float], keys: list[str]) -> np.ndarray:
    """Weighted blend of seed-averaged groups, each molecule's energies sorted ascending."""
    pred = sum(w * np.mean([np.stack([m[k] for k in keys]) for m in group], axis=0) for w, group in zip(weights, groups))
    return np.sort(pred, axis=1)


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of mean(|err_a|) - mean(|err_b|) over molecules resampled with replacement."""
    diff = np.abs(err_a) - np.abs(err_b)
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(n_boot)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def compare(target: np.ndarray, preds: dict[str, np.ndarray], ours: str, theirs: str) -> dict:
    row = {"n": int(target.size)}
    for name, pred in preds.items():
        err = pred - target
        row[name] = {"S1_mae_eV": float(np.abs(err).mean()), "S1_mean_signed_eV": float(err.mean())}
    row["ours_minus_theirs_eV"] = row[ours]["S1_mae_eV"] - row[theirs]["S1_mae_eV"]
    row["ours_minus_theirs_ci95"] = paired_bootstrap(preds[ours] - target, preds[theirs] - target)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--their-split", default="data/processed/external_testsets/visnetgwbse_split.csv")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--pred-dir", default="runs/head_to_head")
    parser.add_argument("--uma-test", default="runs/uma_qm9gwbse_richind_s{seed}.test_predictions.csv")
    parser.add_argument("--visnet-test", default="runs/ensemble_preds/distill_s{seed}.test_predictions.csv")
    parser.add_argument("--weights", type=float, nargs=2, default=[0.6, 0.4], help="UMA, ViSNet")
    parser.add_argument("--report", default="runs/head_to_head_report.json")
    args = parser.parse_args()
    pred_cols = [f"prediction_{c}" for c in ENERGY]
    pred_dir = Path(args.pred_dir)

    # Their checkpoints on every QM9GWBSE molecule; choose the best BSE-finetuned one on their val split.
    q = np.load(args.packed, allow_pickle=False)
    q_keys = [str(k) for k in q["molecule_key"]]
    cols = [str(c) for c in q["target_columns"]]
    q_s1 = dict(zip(q_keys, q["targets"][:, cols.index("S1_eV")].astype(float)))
    our_split = dict(zip(q_keys, q["split"].astype(str)))
    with open(args.their_split, newline="", encoding="utf-8") as stream:
        their_split = {row["molecule_key"]: row["split"] for row in csv.DictReader(stream)}
    theirs_qm9 = {p.name.split(".")[1]: read_csv_columns(p, ["prediction_S1_eV"])
                  for p in sorted((pred_dir / "visnetgwbse").glob("qm9gwbse_packed.*.csv"))}
    report = {"weights": args.weights, "their_checkpoints": {}}
    for name, pred in theirs_qm9.items():
        row = {}
        for split in ("train", "val", "test"):
            keys = [k for k in q_keys if their_split[k] == split]
            row[f"their_{split}_S1_mae_eV"] = float(np.nanmean([abs(pred[k][0] - q_s1[k]) for k in keys]))
        report["their_checkpoints"][name] = row
    finetuned = {n: r for n, r in report["their_checkpoints"].items() if not n.startswith("pre")}
    best = min(finetuned, key=lambda n: finetuned[n]["their_val_S1_mae_eV"])
    report["theirs_best_by_their_val"] = best

    # QM9GWBSE: our stored test predictions vs their best checkpoint, outside their training set.
    uma = [read_csv_columns(args.uma_test.format(seed=s), pred_cols) for s in SEEDS]
    vis = [read_csv_columns(args.visnet_test.format(seed=s), pred_cols) for s in SEEDS]
    report["qm9gwbse"] = {}
    for subset, allowed in (("strict_their_test", {"test"}), ("disjoint_their_val_test", {"val", "test"})):
        keys = sorted(k for k in uma[0] if our_split.get(k) == "test" and their_split[k] in allowed)
        target = np.array([q_s1[k] for k in keys])
        preds = {"ours_production": blend_sorted([uma, vis], args.weights, keys)[:, 0],
                 "ours_uma_only": blend_sorted([uma], [1.0], keys)[:, 0],
                 **{f"theirs_{n}": np.array([theirs_qm9[n][k][0] for k in keys]) for n in finetuned}}
        report["qm9gwbse"][subset] = compare(target, preds, "ours_production", f"theirs_{best}")

    # External pool: our six members and their checkpoints.
    ext_uma = [read_csv_columns(pred_dir / f"ext_uma_richind_s{s}.csv", pred_cols) for s in SEEDS]
    ext_vis = [read_csv_columns(pred_dir / f"ext_distill_s{s}.csv", pred_cols) for s in SEEDS]
    theirs_ext = {n: read_csv_columns(pred_dir / "visnetgwbse" / f"pool.{n}.csv", ["prediction_S1_eV"]) for n in finetuned}
    with open(args.labels, newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    control = [r["molecule_key"] for r in labels if r["set"] == "QM9GWBSE_control"]
    report["control_max_abs_diff_eV"] = {
        f"{family}_s{s}": float(max(np.abs(ext[i][k] - stored[i][k.split(":")[1]]).max() for k in control))
        for family, ext, stored in (("uma", ext_uma, uma), ("visnet", ext_vis, vis)) for i, s in enumerate(SEEDS)}
    report["external"] = {}
    for set_name in sorted({r["set"] for r in labels} - {"QM9GWBSE_control"}):
        rows = [r for r in labels if r["set"] == set_name]
        untrained = [r for r in rows if "train" not in r["ours_split_of_overlaps"].split("|")
                     and "train" not in r["theirs_split_of_overlaps"].split("|")]
        for subset, keep in (("all", rows), ("outside_both_train_sets", untrained),
                             ("no_qm9_overlap", [r for r in rows if not r["qm9_overlap_ids"]]),
                             ("non_CHONF", [r for r in rows if set(r["elements"].split("-")) - set("HCNOF")])):
            if not keep:
                continue
            keys = [r["molecule_key"] for r in keep]
            target = np.array([float(r["S1_eV"]) for r in keep])
            production = blend_sorted([ext_uma, ext_vis], args.weights, keys)
            preds = {"ours_production": production[:, 0],
                     "ours_uma_only": blend_sorted([ext_uma], [1.0], keys)[:, 0],
                     "ours_visnet_only": blend_sorted([ext_vis], [1.0], keys)[:, 0],
                     **{f"theirs_{n}": np.array([theirs_ext[n][k][0] for k in keys]) for n in finetuned}}
            row = compare(target, preds, "ours_production", f"theirs_{best}")
            full = np.array([[float(r[c]) if r[c] else np.nan for c in ENERGY] for r in keep])
            if not np.isnan(full).any():
                row["ours_production_S_agg_eV"] = float(np.abs(production - full).mean())
                row["ours_production_per_state_mae_eV"] = [float(v) for v in np.abs(production - full).mean(axis=0)]
            report["external"][f"{set_name}/{subset}"] = row

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    theirs_name = f"theirs_{best}"
    print(f"their best checkpoint by their val split: {best}")
    print(f"{'subset':34s} {'n':>6s} {'ours S1':>8s} {'theirs S1':>9s} {'diff':>8s} {'95% CI':>18s} {'ours S-agg':>10s}")
    for label, row in [*(("QM9GWBSE/" + k, v) for k, v in report["qm9gwbse"].items()), *report["external"].items()]:
        ci = row["ours_minus_theirs_ci95"]
        print(f"{label:34s} {row['n']:6d} {row['ours_production']['S1_mae_eV']:8.4f} {row[theirs_name]['S1_mae_eV']:9.4f} "
              f"{row['ours_minus_theirs_eV']:+8.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] {row.get('ours_production_S_agg_eV', float('nan')):10.4f}")
    print("control max |pool - stored| (eV):", report["control_max_abs_diff_eV"])


if __name__ == "__main__":
    main()
