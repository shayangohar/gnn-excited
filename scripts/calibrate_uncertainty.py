#!/usr/bin/env python3
"""Calibrated uncertainty for the production blend (TASK-250, DOC-248 L6, TASK-188).

Members are grouped into families (e.g. UMA seeds, PubChemQC-ViSNet seeds); the prediction is the
weighted mean of family means, and the per-molecule, per-state spread sigma is the std over all
members. Split conformal on VAL with score |y - yhat| / sigma gives the interval half-width
q * sigma at each coverage level; TEST reports empirical coverage and mean width, overall, per
state, and per heavy-atom group (Mondrian check). Also reports Spearman(sigma, |error|).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ENERGY = [f"S{i}_eV" for i in range(1, 6)]


def read(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = sorted(csv.DictReader(open(path, newline="", encoding="utf-8")), key=lambda r: r["molecule_key"])
    target = np.array([[float(r[f"target_{c}"]) for c in ENERGY] for r in rows])
    pred = np.array([[float(r[f"prediction_{c}"]) for c in ENERGY] for r in rows])
    return [r["molecule_key"] for r in rows], target, pred


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    n = scores.size
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def blend(families: list[list[np.ndarray]], weights: list[float]) -> tuple[np.ndarray, np.ndarray]:
    mean = sum(w * np.mean(f, axis=0) for w, f in zip(weights, families)) / sum(weights)
    sigma = np.std([m for f in families for m in f], axis=0)
    return np.sort(mean, axis=1), np.maximum(sigma, 1e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", action="append", nargs="+", required=True,
                        help="weight followed by member prefixes (<prefix>.val_predictions.csv / .test_predictions.csv)")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    weights = [float(f[0]) for f in args.family]
    data = {}
    for split in ("val", "test"):
        families, keys, target = [], None, None
        for fam in args.family:
            members = []
            for prefix in fam[1:]:
                k, t, p = read(f"{prefix}.{split}_predictions.csv")
                if keys is None:
                    keys, target = k, t
                elif k != keys:
                    raise SystemExit(f"{prefix} {split} keys differ")
                members.append(p)
            families.append(members)
        pred, sigma = blend(families, weights)
        data[split] = {"keys": keys, "target": target, "pred": pred, "sigma": sigma, "err": np.abs(pred - target)}

    packed = np.load(args.packed, allow_pickle=False)
    index = {str(k): i for i, k in enumerate(packed["molecule_key"])}
    heavy_all = np.add.reduceat((packed["numbers"] > 1).astype(int), packed["atom_offsets"][:-1])
    test_heavy = np.array([heavy_all[index[k]] for k in data["test"]["keys"]])

    val, test = data["val"], data["test"]
    report = {"families": [{"weight": w, "members": f[1:]} for w, f in zip(weights, args.family)],
              "test_S_agg": float(test["err"].mean()),
              "spearman_sigma_error_per_state": [spearman(test["sigma"][:, k], test["err"][:, k]) for k in range(5)],
              "spearman_sigma_error_molecule": spearman(test["sigma"].mean(1), test["err"].mean(1)),
              "conformal": {}}
    for coverage in args.coverages:
        entry = {"per_state": []}
        cover_all, width_all = [], []
        for k in range(5):
            q = conformal_quantile(val["err"][:, k] / val["sigma"][:, k], coverage)
            q_abs = conformal_quantile(val["err"][:, k], coverage)
            inside = test["err"][:, k] <= q * test["sigma"][:, k]
            entry["per_state"].append({
                "state": ENERGY[k], "q": q, "coverage": float(inside.mean()),
                "mean_halfwidth_eV": float((q * test["sigma"][:, k]).mean()),
                "constant_halfwidth_eV": q_abs,
                "coverage_by_heavy_atoms": {"le8": float(inside[test_heavy <= 8].mean()), "9": float(inside[test_heavy == 9].mean())},
            })
            cover_all.append(inside)
            width_all.append(q * test["sigma"][:, k])
        entry["coverage"] = float(np.mean(cover_all))
        entry["mean_halfwidth_eV"] = float(np.mean(width_all))
        report["conformal"][f"{coverage:g}"] = entry
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("test_S_agg", "spearman_sigma_error_molecule", "spearman_sigma_error_per_state")}, indent=2))
    for cov, entry in report["conformal"].items():
        print(f"target {cov}: coverage {entry['coverage']:.3f}, mean half-width {entry['mean_halfwidth_eV']:.4f} eV")


if __name__ == "__main__":
    main()
