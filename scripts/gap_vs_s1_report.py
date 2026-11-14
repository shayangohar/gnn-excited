#!/usr/bin/env python3
"""How our qsGW-BSE targets relate to rCEG's B3LYP HOMO-LUMO gap (KNOW-293, KNOW-309).

Part 1, QM9GWBSE labels only. Each molecule's qsGW-BSE labels (S1-S5, f1-f5, T1-T5) are joined with its
QM9 B3LYP HOMO-LUMO gap (rCEG's data/qm9_gap_reference.csv, by qm9_id) and its qsGW and DFT frontier
levels (qp_features.csv from scripts/label_noise_diagnostic.py). Reported:
  distributions and pairwise differences (B3LYP gap - S1, qsGW gap - S1 = electron-hole binding, S1 - T1);
  Pearson and Spearman correlations;
  how well the labels recover the B3LYP gap: maps fit on the train split, scored on test (identity,
  constant offset, a line in S1, S1+T1, all 15 labels linear, all 15 labels gradient-boosted);
  the S1 -> gap line by S1 brightness (dark S1, f1 < 0.01, is mostly not a HOMO->LUMO pi-pi* state).

Part 2, rCEG's 424-molecule test split. Our he_heads x3 seed-mean S1-S5/f predictions are turned into a
gap estimate three ways: (a) the S1 -> gap line fit on QM9GWBSE labels (no rCEG data used); (b) a line in
predicted S1 fit on rCEG's 1,693 real-label training molecules; (c) ridge on all ten predicted outputs,
fit on the same molecules. Each is scored at label geometries and at SMILES (ETKDG+UFF) geometries
against rCEG and the UMA gap models retrained on rCEG's target, overall and per group, with paired
bootstrap CIs (negative favours the first model).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from head_to_head_report import paired_bootstrap

SEEDS = (17, 18, 19)


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def stats(x: np.ndarray) -> dict:
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"mean": float(x.mean()), "sd": float(x.std()), "p5": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "min": float(x.min()), "max": float(x.max())}


def correlations(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(spearmanr(a, b)[0])}


def fit_linear(x: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Least squares (optionally ridge on the non-intercept weights) for y = [x, 1] @ w."""
    x1 = np.column_stack([x, np.ones(len(x))])
    reg = ridge * np.eye(x1.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(x1.T @ x1 + reg, x1.T @ y)


def apply_linear(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.ones(len(x))]) @ w


def scores(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target
    ss = ((target - target.mean()) ** 2).sum()
    return {"mae": float(np.abs(err).mean()), "signed": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - (err ** 2).sum() / ss) if ss > 0 else None}


def part1(args) -> tuple[dict, np.ndarray]:
    q = np.load(args.packed, allow_pickle=False)
    cols = [str(c) for c in q["target_columns"]]
    keys = [str(k) for k in q["molecule_key"]]
    split = q["split"].astype(str)
    targets = q["targets"].astype(np.float64)
    S = targets[:, [cols.index(f"S{i}_eV") for i in range(1, 6)]]
    T = targets[:, [cols.index(f"T{i}_eV") for i in range(1, 6)]]
    F = np.expm1(targets[:, [cols.index(f"log1p_S{i}_f") for i in range(1, 6)]])
    gap = {r["qm9_id"]: float(r["gap"]) for r in read_csv(args.qm9_gap)}
    qp = {r["molecule_key"]: r for r in read_csv(args.qp)}
    rows = [i for i, k in enumerate(keys) if k in gap and k in qp and np.isfinite(targets[i]).all()]
    print(f"part 1: {len(rows)} of {len(keys)} QM9GWBSE molecules have a QM9 B3LYP gap and QP levels", flush=True)
    rows = np.array(rows)
    b3 = np.array([gap[keys[i]] for i in rows])
    qpg = np.array([float(qp[keys[i]]["qp_gap"]) for i in rows])
    dftg = np.array([float(qp[keys[i]]["dft_gap"]) for i in rows])
    ip = np.array([float(qp[keys[i]]["ip"]) for i in rows])
    S, T, F, sp = S[rows], T[rows], F[rows], split[rows]
    s1, t1, f1 = S[:, 0], T[:, 0], F[:, 0]
    out = {"molecules": int(rows.size), "splits": {s: int((sp == s).sum()) for s in ("train", "val", "test")}}
    out["distributions_eV"] = {"b3lyp_gap": stats(b3), "S1": stats(s1), "T1": stats(t1), "qsgw_gap": stats(qpg),
                               "qm9gwbse_dft_gap": stats(dftg), "qsgw_ip": stats(ip)}
    out["differences_eV"] = {"b3lyp_gap_minus_S1": stats(b3 - s1), "qsgw_gap_minus_S1 (binding)": stats(qpg - s1),
                             "S1_minus_T1": stats(s1 - t1), "b3lyp_gap_minus_T1": stats(b3 - t1),
                             "qsgw_gap_minus_b3lyp_gap": stats(qpg - b3)}
    out["fractions"] = {"S1_below_b3lyp_gap": float((s1 < b3).mean()), "S1_below_qsgw_gap": float((s1 < qpg).mean()),
                        "T1_below_S1": float((t1 < s1).mean()), "S1_dark_f_lt_0.01": float((f1 < 0.01).mean())}
    out["correlations"] = {"b3lyp_gap~S1": correlations(b3, s1), "b3lyp_gap~T1": correlations(b3, t1),
                           "qsgw_gap~S1": correlations(qpg, s1), "b3lyp_gap~qsgw_gap": correlations(b3, qpg),
                           "b3lyp_gap~qm9gwbse_dft_gap": correlations(b3, dftg), "qm9gwbse_dft_gap~S1": correlations(dftg, s1)}

    tr, te = sp == "train", sp == "test"
    labels15 = np.column_stack([S, T, np.log1p(F)])
    maps = {}
    maps["identity: gap = S1"] = scores(s1[te], b3[te])
    c = float((b3[tr] - s1[tr]).mean())
    maps["offset: gap = S1 + c"] = {"c": c, **scores(s1[te] + c, b3[te])}
    w_s1 = fit_linear(s1[tr, None], b3[tr])
    maps["line: gap = a*S1 + b"] = {"a": float(w_s1[0]), "b": float(w_s1[1]), **scores(apply_linear(w_s1, s1[te, None]), b3[te])}
    w = fit_linear(np.column_stack([s1, t1])[tr], b3[tr])
    maps["linear: S1, T1"] = scores(apply_linear(w, np.column_stack([s1, t1])[te]), b3[te])
    w = fit_linear(labels15[tr], b3[tr], ridge=1e-3)
    maps["linear: all 15 labels"] = scores(apply_linear(w, labels15[te]), b3[te])
    w = fit_linear(qpg[tr, None], b3[tr])
    maps["line: gap = a*qsGW_gap + b"] = scores(apply_linear(w, qpg[te, None]), b3[te])
    w = fit_linear(dftg[tr, None], b3[tr])
    maps["line: gap = a*QM9GWBSE_DFT_gap + b"] = scores(apply_linear(w, dftg[te, None]), b3[te])
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0).fit(labels15[tr], b3[tr])
        maps["boosted trees: all 15 labels"] = scores(hgb.predict(labels15[te]), b3[te])
    except ImportError:
        maps["boosted trees: all 15 labels"] = "sklearn unavailable"
    out["recover_b3lyp_gap_from_labels_test"] = maps

    by_f = {}
    line_pred = apply_linear(w_s1, s1[:, None])
    for name, sel in (("dark S1 (f1 < 0.01)", f1 < 0.01), ("weak S1 (0.01 <= f1 < 0.1)", (f1 >= 0.01) & (f1 < 0.1)),
                      ("bright S1 (f1 >= 0.1)", f1 >= 0.1)):
        m = te & sel
        by_f[name] = {"n_test": int(m.sum()), "b3lyp_gap_minus_S1_mean": float((b3[m] - s1[m]).mean()),
                      "b3lyp_gap_minus_S1_sd": float((b3[m] - s1[m]).std()), "line_map_mae": float(np.abs(line_pred[m] - b3[m]).mean())}
    out["by_S1_brightness_test"] = by_f
    return out, w_s1


def seed_mean(pattern: str) -> dict[str, np.ndarray]:
    per = []
    for seed in SEEDS:
        rows = read_csv(pattern.format(seed=seed))
        per.append({r["molecule_key"]: r for r in rows})
    keys = set.intersection(*(set(p) for p in per))
    cols = [f"prediction_S{i}_eV" for i in range(1, 6)] + [f"prediction_log1p_S{i}_f" for i in range(1, 6)]
    return {k: np.mean([[float(p[k][c]) for c in cols] for p in per], axis=0) for k in keys}


def part2(args, w_qm9) -> dict:
    meta = {r["molecule_key"]: r for r in read_csv(args.meta)}
    ref = np.load(args.ref, allow_pickle=False)
    ref_keys = [str(k) for k in ref["molecule_key"]]
    ref_split = dict(zip(ref_keys, ref["split"].astype(str)))
    ref_gap = dict(zip(ref_keys, ref["targets"][:, 0].astype(np.float64)))
    label = seed_mean(args.pred_label)
    smiles = seed_mean(args.pred_smiles)
    fit_keys = [k for k in ref_keys if ref_split[k] in ("train", "val") and k in label]
    x_fit = np.array([label[k] for k in fit_keys])
    y_fit = np.array([ref_gap[k] for k in fit_keys])
    w_line = fit_linear(x_fit[:, :1], y_fit)
    w_ridge = fit_linear(x_fit, y_fit, ridge=1.0)
    out = {"fit_molecules": len(fit_keys), "map_b_line": {"a": float(w_line[0]), "b": float(w_line[1])},
           "map_a_line_from_qm9gwbse": {"a": float(w_qm9[0]), "b": float(w_qm9[1])}}

    def maps(x: np.ndarray) -> dict[str, np.ndarray]:
        return {"S1 as is": x[:, 0], "a: QM9GWBSE line": apply_linear(w_qm9, x[:, :1]),
                "b: rCEG-fit line": apply_linear(w_line, x[:, :1]), "c: rCEG-fit ridge, 10 outputs": apply_linear(w_ridge, x)}

    test = [k for k in meta if k in label and k in smiles]
    preds = {"rceg": np.array([float(meta[k]["rceg_pred"]) for k in test])}
    for geom, source in (("label_geometry", label), ("smiles_geometry", smiles)):
        x = np.array([source[k] for k in test])
        for name, p in maps(x).items():
            preds[f"ours_S1 {name} [{geom}]"] = p
    for arm, path in (("ref", args.gap_ref_summary), ("qm9", args.gap_qm9_summary)):
        if not Path(path).exists():
            continue
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        for geom, csv_path in (("label_geometry", summary["predictions_csv_path"]),
                               ("smiles_geometry", (summary.get("post_inference") or {}).get("out_csv"))):
            if csv_path and Path(csv_path).exists():
                p = {r["molecule_key"]: float(r["prediction_gap_eV"]) for r in read_csv(csv_path)}
                if all(k in p for k in test):
                    preds[f"uma_gap_model_{arm} [{geom}]"] = np.array([p[k] for k in test])
    target = np.array([float(meta[k]["ref_gap"]) for k in test])
    groups = {"all": np.ones(len(test), bool),
              "neutral closed-shell": np.array([meta[k]["charge"] == "0" and meta[k]["spin"] == "1" for k in test])}
    for field in ("domain", "ref_source"):
        for value in sorted({meta[k][field] for k in test}):
            groups[f"{field}={value}"] = np.array([meta[k][field] == value for k in test])
    out["test_molecules"] = len(test)
    out["groups"] = {}
    for g, m in groups.items():
        row = {"n": int(m.sum())}
        for name, p in preds.items():
            row[name] = scores(p[m], target[m])
            if name != "rceg":
                row[name]["minus_rceg_ci95"] = paired_bootstrap(p[m] - target[m], preds["rceg"][m] - target[m])
        out["groups"][g] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--qm9-gap", default="data/external/rCEG/data/qm9_gap_reference.csv")
    parser.add_argument("--qp", default="data/processed/qm9gwbse/qp_features.csv")
    parser.add_argument("--meta", default="data/processed/rceg_compare/test_meta.csv")
    parser.add_argument("--ref", default="data/processed/rceg_compare/ref.npz")
    parser.add_argument("--pred-label", default="runs/gap_vs_s1/ref_he_heads_s{seed}.csv")
    parser.add_argument("--pred-smiles", default="runs/gap_vs_s1/smiles_he_heads_s{seed}.csv")
    parser.add_argument("--gap-ref-summary", default="runs/uma_rceg_gap_ref_s17.summary.json")
    parser.add_argument("--gap-qm9-summary", default="runs/uma_rceg_gap_qm9_s17.summary.json")
    parser.add_argument("--report", default="runs/gap_vs_s1_report.json")
    args = parser.parse_args()
    report = {"model": "uma_qm9gwbse_richind_he_heads s17-19, seed mean"}
    report["part1_qm9gwbse_labels"], w_qm9 = part1(args)
    report["part2_rceg_test"] = part2(args, w_qm9)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    p1 = report["part1_qm9gwbse_labels"]
    print(json.dumps({k: p1[k] for k in ("molecules", "fractions", "correlations", "recover_b3lyp_gap_from_labels_test")}, indent=1))
    for g in ("all", "neutral closed-shell"):
        row = report["part2_rceg_test"]["groups"][g]
        print(g, {k: round(v["mae"], 3) for k, v in row.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
