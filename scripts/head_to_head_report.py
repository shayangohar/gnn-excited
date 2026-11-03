#!/usr/bin/env python3
"""Head-to-head S1 report: our production blend vs Baum et al. ViSNetGWBSE (TASK-271, DOC-248 L13).

Production (promoted 2026-11-03) is 0.45 * mean(UMA richind x3) + 0.20 * mean(UMA richind S+T x3)
+ 0.35 * mean(distilled ViSNet x3), with each molecule's five singlet energies sorted. Their
checkpoints predict S1 only, so the comparison is S1.

Their side is reported two ways: the paper's default checkpoint (--their-primary, chosen before
seeing any result) and, per subset, their best checkpoint on that subset ("oracle", optimistic for
them). Differences are ours minus theirs with paired-bootstrap 95% intervals (negative favours ours).

  external   PC9, OE62L, OE62H and their QM9 test set. "clean" drops molecules either model may have
             trained on: for their QM9 set (their held-out test split, per the paper) molecules in our
             training split; for the other sets any QM9GWBSE connectivity match. Also non-CHONF
             molecules, and our S1-S5 S-agg where five labels exist.
  qm9gwbse   our test split restricted to their val/test under the split rebuilt from their code
             (visnetgwbse_split.csv). Reported only if that split puts most of their released QM9 test
             molecules in its test part; otherwise the rebuilt split is not the paper's split.

The QM9GWBSE control molecules check that pool inference reproduces the stored test predictions.
On the external sets the production blend is guarded: a family whose energies for a molecule are
non-finite or outside VALID_EV is dropped for that molecule (count in *_guarded_molecules); the
single-family rows stay unguarded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ENERGY = [f"S{i}_eV" for i in range(1, 6)]
SEEDS = (17, 18, 19)
VALID_EV = (0.0, 30.0)  # outside this a member has failed (OE62L:SEJDAM, a 0.75 A H-H contact, gives 1e9 eV)


def read_csv_columns(path: str | Path, columns: list[str]) -> dict[str, np.ndarray]:
    """molecule_key -> float array of the named columns (blank -> NaN)."""
    with open(path, newline="", encoding="utf-8") as stream:
        return {row["molecule_key"]: np.array([float(row[c]) if row[c] != "" else np.nan for c in columns])
                for row in csv.DictReader(stream)}


def blend_sorted(groups: list[list[dict[str, np.ndarray]]], weights: list[float], keys: list[str]) -> np.ndarray:
    """Weighted blend of seed-averaged groups, each molecule's energies sorted ascending."""
    pred = sum(w * np.mean([np.stack([m[k] for k in keys]) for m in group], axis=0) for w, group in zip(weights, groups))
    return np.sort(pred, axis=1)


def guarded_blend(groups: list[list[dict[str, np.ndarray]]], weights: list[float], keys: list[str],
                  valid_ev: tuple[float, float] = VALID_EV) -> tuple[np.ndarray, int]:
    """blend_sorted, but a group whose seed-mean energy for a molecule is non-finite or outside
    valid_ev is dropped for that molecule and the other weights renormalised. Returns the blend
    and the number of molecules where a group was dropped (NaN rows if every group was)."""
    means = np.stack([np.mean([np.stack([m[k] for k in keys]) for m in group], axis=0) for group in groups])
    ok = np.all(np.isfinite(means) & (means >= valid_ev[0]) & (means <= valid_ev[1]), axis=2)  # (group, molecule)
    w = np.asarray(weights, dtype=float)[:, None] * ok
    total = w.sum(axis=0)
    pred = np.einsum("gm,gms->ms", w / np.where(total > 0, total, np.nan), np.where(ok[..., None], means, 0.0))
    return np.sort(pred, axis=1), int((~ok).any(axis=0).sum())


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of mean(|err_a|) - mean(|err_b|) over molecules resampled with replacement."""
    diff = np.abs(err_a) - np.abs(err_b)
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(n_boot)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def compare(target: np.ndarray, preds: dict[str, np.ndarray], ours: str, primary: str, theirs: list[str]) -> dict:
    row = {"n": int(target.size)}
    for name, pred in preds.items():
        err = pred - target
        row[name] = {"S1_mae_eV": float(np.abs(err).mean()), "S1_mean_signed_eV": float(err.mean())}
    oracle = min(theirs, key=lambda n: row[n]["S1_mae_eV"])
    for label, name in (("primary", primary), ("oracle", oracle)):
        row[f"theirs_{label}"] = name
        row[f"ours_minus_theirs_{label}_eV"] = row[ours]["S1_mae_eV"] - row[name]["S1_mae_eV"]
        row[f"ours_minus_theirs_{label}_ci95"] = paired_bootstrap(preds[ours] - target, preds[name] - target)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--their-split", default="data/processed/external_testsets/visnetgwbse_split.csv")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--pred-dir", default="runs/head_to_head")
    parser.add_argument("--uma-test", default="runs/uma_qm9gwbse_richind_s{seed}.test_predictions.csv")
    parser.add_argument("--uma-st-test", default="runs/uma_qm9gwbse_richind_st_s{seed}.test_predictions.csv")
    parser.add_argument("--visnet-test", default="runs/ensemble_preds/distill_s{seed}.test_predictions.csv")
    parser.add_argument("--their-primary", default="gapbse5M128502532", help="paper default: transfer model, 5M pretraining")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.45, 0.20, 0.35], help="UMA, UMA S+T, ViSNet")
    parser.add_argument("--report", default="runs/head_to_head_report.json")
    args = parser.parse_args()
    pred_cols = [f"prediction_{c}" for c in ENERGY]
    pred_dir = Path(args.pred_dir)
    primary = f"theirs_{args.their_primary}"

    q = np.load(args.packed, allow_pickle=False)
    q_keys = [str(k) for k in q["molecule_key"]]
    cols = [str(c) for c in q["target_columns"]]
    q_s1 = dict(zip(q_keys, q["targets"][:, cols.index("S1_eV")].astype(float)))
    our_split = dict(zip(q_keys, q["split"].astype(str)))
    with open(args.their_split, newline="", encoding="utf-8") as stream:
        their_split = {row["molecule_key"]: row["split"] for row in csv.DictReader(stream)}
    with open(args.labels, newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    theirs_qm9 = {p.name.split(".")[1]: read_csv_columns(p, ["prediction_S1_eV"])
                  for p in sorted((pred_dir / "visnetgwbse").glob("qm9gwbse_packed.*.csv"))}
    finetuned = [n for n in theirs_qm9 if not n.startswith("pre")]
    their_names = [f"theirs_{n}" for n in finetuned]

    # Rebuilt-split diagnostics: their MAE by rebuilt split, and where their released QM9 test set falls.
    report = {"weights": args.weights, "their_primary": args.their_primary, "their_checkpoints": {}}
    for name, pred in theirs_qm9.items():
        report["their_checkpoints"][name] = {
            f"rebuilt_{split}_S1_mae_eV": float(np.nanmean([abs(pred[k][0] - q_s1[k]) for k in q_keys if their_split[k] == split]))
            for split in ("train", "val", "test")}
    released = [s for r in labels if r["set"] == "QM9" for s in r["theirs_split_of_overlaps"].split("|") if s]
    frac = sum(s == "test" for s in released) / max(len(released), 1)
    report["rebuilt_split_check"] = {"their_qm9_test_ids": len(released), "fraction_in_rebuilt_test": frac,
                                     "valid": frac > 0.5}

    uma = [read_csv_columns(args.uma_test.format(seed=s), pred_cols) for s in SEEDS]
    uma_st = [read_csv_columns(args.uma_st_test.format(seed=s), pred_cols) for s in SEEDS]
    vis = [read_csv_columns(args.visnet_test.format(seed=s), pred_cols) for s in SEEDS]
    report["qm9gwbse"] = {}
    if report["rebuilt_split_check"]["valid"]:
        for subset, allowed in (("strict_their_test", {"test"}), ("disjoint_their_val_test", {"val", "test"})):
            keys = sorted(k for k in uma[0] if our_split.get(k) == "test" and their_split[k] in allowed)
            target = np.array([q_s1[k] for k in keys])
            preds = {"ours_production": blend_sorted([uma, uma_st, vis], args.weights, keys)[:, 0],
                     "ours_uma_only": blend_sorted([uma], [1.0], keys)[:, 0],
                     **{f"theirs_{n}": np.array([theirs_qm9[n][k][0] for k in keys]) for n in finetuned}}
            report["qm9gwbse"][subset] = compare(target, preds, "ours_production", primary, their_names)

    ext_uma = [read_csv_columns(pred_dir / f"ext_uma_richind_s{s}.csv", pred_cols) for s in SEEDS]
    ext_uma_st = [read_csv_columns(pred_dir / f"ext_uma_richind_st_s{s}.csv", pred_cols) for s in SEEDS]
    ext_vis = [read_csv_columns(pred_dir / f"ext_distill_s{s}.csv", pred_cols) for s in SEEDS]
    theirs_ext = {n: read_csv_columns(pred_dir / "visnetgwbse" / f"pool.{n}.csv", ["prediction_S1_eV"]) for n in finetuned}
    control = [r["molecule_key"] for r in labels if r["set"] == "QM9GWBSE_control"]
    report["control_max_abs_diff_eV"] = {
        f"{family}_s{s}": float(max(np.abs(ext[i][k] - stored[i][k.split(":")[1]]).max() for k in control))
        for family, ext, stored in (("uma", ext_uma, uma), ("uma_st", ext_uma_st, uma_st), ("visnet", ext_vis, vis))
        for i, s in enumerate(SEEDS)}
    report["external"] = {}
    for set_name in sorted({r["set"] for r in labels} - {"QM9GWBSE_control"}):
        rows = [r for r in labels if r["set"] == set_name]
        if set_name == "QM9":
            clean = [r for r in rows if "train" not in r["ours_split_of_overlaps"].split("|")]
        else:
            clean = [r for r in rows if not r["qm9_overlap_ids"]]
        for subset, keep in (("all", rows), ("clean", clean),
                             ("non_CHONF", [r for r in rows if set(r["elements"].split("-")) - set("HCNOF")])):
            if not keep:
                continue
            keys = [r["molecule_key"] for r in keep]
            target = np.array([float(r["S1_eV"]) for r in keep])
            production, guarded = guarded_blend([ext_uma, ext_uma_st, ext_vis], args.weights, keys)
            preds = {"ours_production": production[:, 0],
                     "ours_uma_only": blend_sorted([ext_uma], [1.0], keys)[:, 0],
                     "ours_uma_st_only": blend_sorted([ext_uma_st], [1.0], keys)[:, 0],
                     "ours_visnet_only": blend_sorted([ext_vis], [1.0], keys)[:, 0],
                     **{f"theirs_{n}": np.array([theirs_ext[n][k][0] for k in keys]) for n in finetuned}}
            row = compare(target, preds, "ours_production", primary, their_names)
            row["ours_production_guarded_molecules"] = guarded
            full = np.array([[float(r[c]) if r[c] else np.nan for c in ENERGY] for r in keep])
            if not np.isnan(full).any():
                row["ours_production_S_agg_eV"] = float(np.abs(production - full).mean())
                row["ours_production_per_state_mae_eV"] = [float(v) for v in np.abs(production - full).mean(axis=0)]
            report["external"][f"{set_name}/{subset}"] = row

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("rebuilt split check:", report["rebuilt_split_check"])
    print(f"{'subset':28s} {'n':>5s} {'ours':>7s} {'default':>7s} {'diff [95% CI]':>26s} {'best':>7s} {'diff [95% CI]':>26s} {'S-agg':>7s}")
    for label, row in [*(("QM9GWBSE/" + k, v) for k, v in report["qm9gwbse"].items()), *report["external"].items()]:
        cells = []
        for kind in ("primary", "oracle"):
            ci = row[f"ours_minus_theirs_{kind}_ci95"]
            cells.append(f"{row[row[f'theirs_{kind}']]['S1_mae_eV']:7.4f} "
                         f"{row[f'ours_minus_theirs_{kind}_eV']:+8.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"{label:28s} {row['n']:5d} {row['ours_production']['S1_mae_eV']:7.4f} {cells[0]} {cells[1]} "
              f"{row.get('ours_production_S_agg_eV', float('nan')):7.4f}")
    print("control max |pool - stored| (eV):", report["control_max_abs_diff_eV"])


if __name__ == "__main__":
    main()
