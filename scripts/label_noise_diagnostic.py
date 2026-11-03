#!/usr/bin/env python3
"""Bet C: is the shared S1-S5 error floor label/basis noise? (TASK-276, DOC-274 §3)

Joins per-molecule qsGW quasiparticle and DFT orbital energies (QM9GWBSE eqp.zip / edft.zip, Zenodo
17902233: one ascending list of orbital energies in eV per molecule, all-electron and closed shell, so
the HOMO is index n_electrons/2 - 1) with the production blend's val+test errors (UMA richind 0.45 +
UMA richind S+T 0.20 + distilled ViSNet 0.35, singlets sorted).

Hypothesis: states close to or above the qsGW ionization threshold (E_n - IP near or above 0, i.e.
Rydberg-like), and molecules with an unbound qsGW LUMO, are diffuse. The finite TZ3P basis then makes
their labels noisy, and the error the model families share concentrates there.

Writes --features-csv (per molecule QP/DFT features, reusable as bet A supervision) and --report:
  alignment checks (S1 below the QP gap, correlation);
  production MAE by E_n - IP bin, by LUMO sign, per state below/above an IP margin;
  a clean-subset S-agg (all five states at least m eV below IP);
  UMA-vs-ViSNet error correlation inside and outside the near-IP region;
  cross-validated R^2 of a gradient-boosted model for |error| from QP features against a model
  that sees only state index, energy, neighbour gap and size (the increment is what QP physics adds);
  label S_n < T_n counts over the whole dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np

S = [f"S{i}_eV" for i in range(1, 6)]
T = [f"T{i}_eV" for i in range(1, 6)]
FAMILIES = {"uma": "runs/uma_qm9gwbse_richind_s{s}.{split}_predictions.csv",
            "uma_st": "runs/uma_qm9gwbse_richind_st_s{s}.{split}_predictions.csv",
            "visnet": "runs/ensemble_preds/distill_s{s}.{split}_predictions.csv"}
WEIGHTS = {"uma": 0.45, "uma_st": 0.20, "visnet": 0.35}
IP_BINS = [-np.inf, -4.0, -3.0, -2.0, -1.0, 0.0, np.inf]
MOL = re.compile(r"mol_(\d+)\.dat$")


def read_levels(zip_path: str | Path) -> dict[str, np.ndarray]:
    """QM9 id (string) -> ascending orbital energies (eV)."""
    out = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match = MOL.search(name)
            if match:
                out[match.group(1)] = np.sort(np.array(archive.read(name).decode().split(), dtype=np.float64))
    return out


def frontier(levels: np.ndarray, n_occ: int) -> dict[str, float]:
    """HOMO-1, HOMO, LUMO..LUMO+4 and the number of bound (negative) virtual levels."""
    virt = levels[n_occ:]
    pad = np.full(5, np.nan)
    pad[:min(5, virt.size)] = virt[:5]
    return {"homo1": float(levels[n_occ - 2]) if n_occ >= 2 else np.nan, "homo": float(levels[n_occ - 1]),
            **{("lumo" if k == 0 else f"lumo{k}"): float(pad[k]) for k in range(5)},
            "n_bound_virtual": int((virt < 0).sum())}


def read_family(pattern: str, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    preds, keys, target = [], None, None
    for s in (17, 18, 19):
        with open(pattern.format(s=s, split=split), newline="", encoding="utf-8") as stream:
            rows = sorted(csv.DictReader(stream), key=lambda r: r["molecule_key"])
        k = [r["molecule_key"] for r in rows]
        if keys is not None and k != keys:
            raise SystemExit(f"{pattern} seed {s}: molecule keys differ")
        keys = k
        target = np.array([[float(r[f"target_{c}"]) for c in S] for r in rows])
        preds.append(np.array([[float(r[f"prediction_{c}"]) for c in S] for r in rows]))
    return keys, target, np.mean(preds, axis=0)


def binned(values: np.ndarray, errors: np.ndarray, edges) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values > lo) & (values <= hi)
        out.append({"bin": f"({lo}, {hi}]", "n": int(mask.sum()),
                    "mae": float(np.abs(errors[mask]).mean()) if mask.any() else None})
    return out


def cv_r2(features: np.ndarray, target: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:
    """Out-of-fold R^2 of a gradient-boosted regressor, folds grouped by molecule."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    oof = np.empty_like(target)
    for train, test in GroupKFold(n_splits=5).split(features, target, groups):
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
        oof[test] = model.fit(features[train], target[train]).predict(features[test])
    return float(1.0 - ((target - oof) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eqp", default="data/raw/qm9gwbse/eqp.zip")
    parser.add_argument("--edft", default="data/raw/qm9gwbse/edft.zip")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--features-csv", default="data/processed/qm9gwbse/qp_features.csv")
    parser.add_argument("--report", default="runs/label_noise_diagnostic.json")
    args = parser.parse_args()

    q = np.load(args.packed, allow_pickle=False)
    keys = [str(k) for k in q["molecule_key"]]
    offsets = q["atom_offsets"]
    n_el = np.add.reduceat(q["numbers"].astype(np.int64), offsets[:-1])
    natoms = np.diff(offsets)
    heavy = np.add.reduceat((q["numbers"] > 1).astype(np.int64), offsets[:-1])
    cols = [str(c) for c in q["target_columns"]]
    labels_s = q["targets"][:, [cols.index(c) for c in S]]
    labels_t = q["targets"][:, [cols.index(c) for c in T]] if all(c in cols for c in T) else None

    qp, dft = read_levels(args.eqp), read_levels(args.edft)
    feats: dict[str, dict] = {}
    missing = 0
    for row, key in enumerate(keys):
        if key not in qp or key not in dft or n_el[row] % 2:
            missing += 1
            continue
        n_occ = int(n_el[row] // 2)
        g, d = frontier(qp[key], n_occ), frontier(dft[key], n_occ)
        feats[key] = {"natoms": int(natoms[row]), "heavy": int(heavy[row]), "n_occ": n_occ,
                      **{f"qp_{k}": v for k, v in g.items()}, **{f"dft_{k}": v for k, v in d.items()},
                      "ip": -g["homo"], "ea": -g["lumo"], "qp_gap": g["lumo"] - g["homo"],
                      "dft_gap": d["lumo"] - d["homo"], "gw_gap_shift": (g["lumo"] - g["homo"]) - (d["lumo"] - d["homo"]),
                      "lumo_unbound": int(g["lumo"] > 0)}
    Path(args.features_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["molecule_key", *next(iter(feats.values())).keys()]
    with open(args.features_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for key, f in feats.items():
            writer.writerow([key, *(f[c] for c in fields[1:])])

    have = np.array([k in feats for k in keys])
    s1 = labels_s[have, 0]
    gap = np.array([feats[k]["qp_gap"] for k in keys if k in feats])
    report: dict = {"molecules": len(keys), "with_qp_features": int(have.sum()), "missing_or_open_shell": missing,
                    "alignment": {"S1_below_qp_gap_fraction": float((s1 < gap).mean()),
                                  "corr_S1_qp_gap": float(np.corrcoef(s1, gap)[0, 1]),
                                  "lumo_unbound_fraction": float(np.mean([f["lumo_unbound"] for f in feats.values()]))}}
    if labels_t is not None:
        viol = labels_s < labels_t - 1e-6
        report["label_S_lt_T"] = {"per_state_rate": viol.mean(axis=0).round(6).tolist(),
                                  "molecules_any": int(viol.any(axis=1).sum()), "worst_eV": float((labels_t - labels_s).max())}

    # Production and family errors on val+test, one row per molecule-state.
    fam_err, prod_err, rows_keys, targets = {}, None, [], []
    for split in ("val", "test"):
        loaded = {name: read_family(pattern, split) for name, pattern in FAMILIES.items()}
        k0 = loaded["uma"][0]
        if any(v[0] != k0 for v in loaded.values()):
            raise SystemExit(f"{split}: families disagree on molecule keys")
        tgt = loaded["uma"][1]
        prod = np.sort(sum(WEIGHTS[n] * loaded[n][2] for n in FAMILIES), axis=1)
        prod_err = prod - tgt if prod_err is None else np.vstack([prod_err, prod - tgt])
        for name in ("uma", "visnet"):
            e = np.sort(loaded[name][2], axis=1) - tgt
            fam_err[name] = e if name not in fam_err else np.vstack([fam_err[name], e])
        rows_keys += k0
        targets.append(tgt)
    targets = np.vstack(targets)
    keep = np.array([k in feats for k in rows_keys])
    rows_keys = [k for k, m in zip(rows_keys, keep) if m]
    prod_err, targets = prod_err[keep], targets[keep]
    fam_err = {n: e[keep] for n, e in fam_err.items()}
    ip = np.array([feats[k]["ip"] for k in rows_keys])
    margin = targets - ip[:, None]  # E_n - IP per state
    unbound = np.array([feats[k]["lumo_unbound"] for k in rows_keys], dtype=bool)

    report["val_test_molecules"] = len(rows_keys)
    report["production_S_agg"] = float(np.abs(prod_err).mean())
    report["production_per_state"] = np.abs(prod_err).mean(axis=0).round(5).tolist()
    report["E_minus_IP_quantiles_per_state"] = {S[n]: np.quantile(margin[:, n], [0.05, 0.5, 0.95]).round(3).tolist() for n in range(5)}
    report["mae_by_E_minus_IP"] = binned(margin.ravel(), prod_err.ravel(), IP_BINS)
    report["mae_by_E_minus_IP_per_state"] = {S[n]: binned(margin[:, n], prod_err[:, n], IP_BINS) for n in range(5)}
    report["mae_by_lumo_sign"] = {"bound": np.abs(prod_err[~unbound]).mean(axis=0).round(5).tolist(),
                                  "unbound": np.abs(prod_err[unbound]).mean(axis=0).round(5).tolist(),
                                  "n_bound": int((~unbound).sum()), "n_unbound": int(unbound.sum())}
    report["clean_subset"] = {}
    for m in (1.0, 2.0, 3.0):
        clean = (margin < -m).all(axis=1)
        near = margin >= -m
        report["clean_subset"][f"all_states_{m:g}eV_below_IP"] = {
            "molecules": int(clean.sum()), "S_agg": float(np.abs(prod_err[clean]).mean()),
            "per_state": np.abs(prod_err[clean]).mean(axis=0).round(5).tolist(),
            "share_of_states_near_IP": float(near.mean()),
            "share_of_abs_error_near_IP": float(np.abs(prod_err)[near].sum() / np.abs(prod_err).sum())}
    near = margin >= -2.0
    report["family_error_correlation"] = {
        "all": float(np.corrcoef(fam_err["uma"].ravel(), fam_err["visnet"].ravel())[0, 1]),
        "within_2eV_of_IP": float(np.corrcoef(fam_err["uma"][near], fam_err["visnet"][near])[0, 1]),
        "more_than_2eV_below_IP": float(np.corrcoef(fam_err["uma"][~near], fam_err["visnet"][~near])[0, 1])}

    # How much of |error| do QP features explain beyond state index, energy, neighbour gap and size?
    n_mol = len(rows_keys)
    state = np.tile(np.arange(5), n_mol)
    energy = targets.ravel()
    padded = np.hstack([np.full((n_mol, 1), np.inf), targets, np.full((n_mol, 1), np.inf)])
    neighbour = np.minimum(np.abs(padded[:, 1:-1] - padded[:, :-2]), np.abs(padded[:, 2:] - padded[:, 1:-1])).ravel()
    size = np.repeat([[feats[k]["natoms"], feats[k]["heavy"]] for k in rows_keys], 5, axis=0)
    base = np.column_stack([state, energy, neighbour, size])
    qp_cols = ["ip", "ea", "qp_gap", "dft_gap", "gw_gap_shift", "lumo_unbound", "qp_n_bound_virtual",
               "qp_lumo1", "qp_lumo2", "qp_homo1", "dft_lumo", "dft_homo"]
    qp_feats = np.repeat([[feats[k][c] for c in qp_cols] for k in rows_keys], 5, axis=0)
    full = np.column_stack([base, qp_feats, margin.ravel(), (targets - np.array([feats[k]["qp_gap"] for k in rows_keys])[:, None]).ravel()])
    groups = np.repeat(np.arange(n_mol), 5)
    try:
        report["explained_abs_error_r2"] = {}
        for name, err in (("production", prod_err), ("shared_mean_of_families", (fam_err["uma"] + fam_err["visnet"]) / 2)):
            y = np.abs(err).ravel()
            r2_base, r2_full = cv_r2(base, y, groups), cv_r2(full, y, groups)
            report["explained_abs_error_r2"][name] = {"state_energy_size_only": r2_base, "plus_qp_features": r2_full,
                                                      "increment": r2_full - r2_base}
    except ImportError as exc:
        report["explained_abs_error_r2"] = f"skipped: {exc}"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
