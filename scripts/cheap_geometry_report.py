#!/usr/bin/env python3
"""How much accuracy does production lose on SMILES-derived geometries? (TASK-291 E3, TASK-280)

For each geometry source (b3lyp, etkdg, xtb, uma; see build_cheap_geometries.py and relax_uma_pool.py)
reads the nine production members' predictions <pred-dir>/<source>.<member>.csv and blends them as
production does (KNOW-288: 0.45 UMA richind + 0.20 UMA richind S+T + 0.35 distilled ViSNet, guarded,
sorted). Reports S-agg and per-state MAE against the QM9GWBSE labels, the change against the b3lyp
source (paired bootstrap), error by heavy-atom RMSD bin, and whether the members' spread flags the
molecules whose geometry hurts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cheap_geometry import heavy_atom_rmsd
from head_to_head_report import ENERGY, guarded_blend, paired_bootstrap, read_csv_columns

FAMILIES = {"uma": "uma_richind_s{s}", "uma_st": "uma_richind_st_s{s}", "visnet": "distill_s{s}"}
WEIGHTS = [0.45, 0.20, 0.35]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--geom-dir", default="data/processed/cheap_geometry")
    parser.add_argument("--pred-dir", default="runs/cheap_geometry")
    parser.add_argument("--sources", nargs="+", default=["b3lyp", "etkdg", "xtb", "uma"])
    parser.add_argument("--report", default="runs/cheap_geometry_report.json")
    args = parser.parse_args()
    q = np.load(args.packed, allow_pickle=False)
    cols = [str(c) for c in q["target_columns"]]
    labels = dict(zip([str(k) for k in q["molecule_key"]], q["targets"][:, [cols.index(c) for c in ENERGY]].astype(float)))
    pools: dict[str, dict] = {}

    def geometry(source: str, key: str):
        if source not in pools:
            p = np.load(Path(args.geom_dir) / f"pool_{source}.npz", allow_pickle=False)
            index = {str(k): i for i, k in enumerate(p["molecule_key"])}
            pools[source] = {"p": p, "index": index}
        p, i = pools[source]["p"], pools[source]["index"][key]
        a, b = int(p["atom_offsets"][i]), int(p["atom_offsets"][i + 1])
        return p["numbers"][a:b], p["positions"][a:b].astype(np.float64)

    def rmsd_to_b3lyp(source: str, key: str) -> float:
        value = heavy_atom_rmsd(*geometry("b3lyp", key), *geometry(source, key))
        return np.nan if value is None else value

    pred_cols = [f"prediction_{c}" for c in ENERGY]

    blends, members = {}, {}
    for source in args.sources:
        try:
            groups = [[read_csv_columns(Path(args.pred_dir) / f"{source}.{pattern.format(s=s)}.csv", pred_cols) for s in (17, 18, 19)]
                      for pattern in FAMILIES.values()]
        except FileNotFoundError as exc:
            print(f"skip {source}: {exc}")
            continue
        keys = sorted(set.intersection(*(set(d) for g in groups for d in g)) & set(labels))
        blends[source] = dict(zip(keys, guarded_blend(groups, WEIGHTS, keys)[0]))
        members[source] = {k: np.array([d[k][0] for g in groups for d in g]) for k in keys}
    keys = sorted(set.intersection(*(set(b) for b in blends.values())))
    target = np.array([labels[k] for k in keys])
    report = {"molecules": len(keys), "sources": {}}
    ref_err = None
    for source, blend in blends.items():
        pred = np.array([blend[k] for k in keys])
        err = np.abs(pred - target)
        per_mol = err.mean(axis=1)
        row = {"S_agg": float(err.mean()), "per_state": err.mean(axis=0).round(5).tolist(),
               "S1_signed": float((pred[:, 0] - target[:, 0]).mean())}
        if source == "b3lyp":
            ref_err = per_mol
        elif ref_err is not None:
            row["minus_b3lyp"] = float(per_mol.mean() - ref_err.mean())
            row["minus_b3lyp_ci95"] = paired_bootstrap(per_mol, ref_err)
        spread = np.array([members[source][k].std() for k in keys])
        row["spread_vs_abs_S1_error_spearman"] = spearman(spread, err[:, 0])
        top = spread >= np.quantile(spread, 0.9)
        row["S1_mae_top_decile_spread"] = float(err[top, 0].mean())
        row["S1_mae_other"] = float(err[~top, 0].mean())
        if source != "b3lyp" and (Path(args.geom_dir) / f"pool_{source}.npz").exists():
            values = np.array([rmsd_to_b3lyp(source, k) for k in keys], dtype=float)
            row["median_heavy_rmsd_A"] = float(np.nanmedian(values))
            row["S_agg_by_rmsd"] = {f"({lo}, {hi}]": {"n": int(m.sum()), "S_agg": float(err[m].mean()) if m.any() else None}
                                    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 99))
                                    for m in [(values > lo) & (values <= hi)]}
        report["sources"][source] = row
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
