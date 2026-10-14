#!/usr/bin/env python3
"""Val-weighted blends of QM9GWBSE prediction files (UMA + ViSNet ensembles).

Reads a YAML spec of members (val/test prediction CSVs with molecule_key,
target_<col>, prediction_<col>), averages named groups (seed ensembles), and
for each listed blend searches simplex weights on a grid using VAL S-agg only,
then reports TEST metrics at the chosen weights. Missing files are skipped
with a note, so a blend job can run with afterany dependencies.

S-agg is the mean absolute error over the S1-S5 energy columns (eV), the same
quantity as batch_mae's energy_eV_mae. "Sorted" metrics sort each molecule's
predicted energies ascending; for ascending targets this cannot raise the MAE.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

ENERGY = [f"S{i}_eV" for i in range(1, 6)]
OSCILLATOR = [f"log1p_S{i}_f" for i in range(1, 6)]


def read_predictions(path: str | Path) -> dict[str, dict[str, np.ndarray] | list[str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: row["molecule_key"])
    out: dict = {"keys": [row["molecule_key"] for row in rows]}
    for kind in ("target", "prediction"):
        out[f"{kind}_energy"] = np.array([[float(row[f"{kind}_{c}"]) for c in ENERGY] for row in rows])
        if rows and all(f"{kind}_{c}" in rows[0] for c in OSCILLATOR):
            out[f"{kind}_osc"] = np.array([[float(row[f"{kind}_{c}"]) for c in OSCILLATOR] for row in rows])
    return out


def metrics(pred_energy, target_energy, pred_osc=None, target_osc=None) -> dict[str, float]:
    sorted_pred = np.sort(pred_energy, axis=1)
    result = {
        "S_agg": float(np.abs(pred_energy - target_energy).mean()),
        "S_agg_sorted": float(np.abs(sorted_pred - target_energy).mean()),
        "per_state": [float(v) for v in np.abs(pred_energy - target_energy).mean(axis=0)],
        "adjacent_inversion_rate": float((np.diff(pred_energy, axis=1) < 0).mean()),
    }
    if pred_osc is not None and target_osc is not None:
        result["log1p_osc_mae"] = float(np.abs(pred_osc - target_osc).mean())
    return result


def simplex_grid(k: int, step: float):
    n = int(round(1.0 / step))
    for combo in itertools.product(range(n + 1), repeat=k - 1):
        if sum(combo) <= n:
            yield np.array([*combo, n - sum(combo)], dtype=np.float64) / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    step = float(spec.get("grid_step", 0.05))

    notes: list[str] = []
    members: dict[str, dict] = {}
    reference_keys: dict[str, list[str]] = {}
    reference_targets: dict[str, np.ndarray] = {}
    for name, paths in (spec.get("members") or {}).items():
        missing = [p for p in (paths["val"], paths["test"]) if not Path(p).exists()]
        if missing:
            notes.append(f"member {name} skipped; missing {missing}")
            continue
        member = {split: read_predictions(paths[split]) for split in ("val", "test")}
        for split in ("val", "test"):
            keys = member[split]["keys"]
            if split not in reference_keys:
                reference_keys[split], reference_targets[split] = keys, member[split]["target_energy"]
            elif keys != reference_keys[split]:
                raise SystemExit(f"member {name} {split} molecule keys differ from the first member")
            elif not np.allclose(member[split]["target_energy"], reference_targets[split], atol=1e-5):
                raise SystemExit(f"member {name} {split} targets differ from the first member")
        members[name] = member

    entries: dict[str, dict] = {}
    for name, member in members.items():
        entries[name] = {split: {"pred_energy": member[split]["prediction_energy"],
                                 "pred_osc": member[split].get("prediction_osc")} for split in ("val", "test")}
        entries[name]["members"] = [name]
    for name, group in (spec.get("groups") or {}).items():
        present = [m for m in group if m in members]
        if not present:
            notes.append(f"group {name} skipped; no members available")
            continue
        if len(present) < len(group):
            notes.append(f"group {name} uses {present} of {group}")
        entry = {"members": present}
        for split in ("val", "test"):
            energies = [members[m][split]["prediction_energy"] for m in present]
            oscs = [members[m][split].get("prediction_osc") for m in present]
            entry[split] = {"pred_energy": np.mean(energies, axis=0),
                            "pred_osc": np.mean(oscs, axis=0) if all(o is not None for o in oscs) else None}
        entries[name] = entry

    def score(entry_preds, split):
        target = members_any[split]
        return metrics(entry_preds["pred_energy"], target["target_energy"], entry_preds["pred_osc"], target.get("target_osc"))

    if not members:
        raise SystemExit("no members available")
    first = next(iter(members.values()))
    members_any = {split: first[split] for split in ("val", "test")}

    report: dict = {"spec": args.spec, "grid_step": step, "notes": notes,
                    "n_val": len(reference_keys.get("val", [])), "n_test": len(reference_keys.get("test", [])),
                    "single": {}, "blends": []}
    for name, entry in entries.items():
        report["single"][name] = {"members": entry["members"], "val": score(entry["val"], "val"), "test": score(entry["test"], "test")}

    for blend in spec.get("blends") or []:
        names = list(blend)
        if any(n not in entries for n in names):
            notes.append(f"blend {names} skipped; missing {[n for n in names if n not in entries]}")
            continue
        best = None
        for weights in simplex_grid(len(names), step):
            pred = sum(w * entries[n]["val"]["pred_energy"] for w, n in zip(weights, names))
            value = float(np.abs(pred - members_any["val"]["target_energy"]).mean())
            if best is None or value < best[0]:
                best = (value, weights)
        weights = best[1]
        equal = np.full(len(names), 1.0 / len(names))
        result = {"members": names, "val_chosen_weights": [round(float(w), 4) for w in weights]}
        for label, w in (("val_weighted", weights), ("equal_weight", equal)):
            for split in ("val", "test"):
                energy = sum(wi * entries[n][split]["pred_energy"] for wi, n in zip(w, names))
                oscs = [entries[n][split]["pred_osc"] for n in names]
                osc = sum(wi * o for wi, o in zip(w, oscs)) if all(o is not None for o in oscs) else None
                target = members_any[split]
                result[f"{label}_{split}"] = metrics(energy, target["target_energy"], osc, target.get("target_osc"))
        report["blends"].append(result)

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'entry':40s} {'val S-agg':>10s} {'test S-agg':>10s} {'test sorted':>11s} {'test inv%':>9s} {'test osc':>9s}")
    for name, row in report["single"].items():
        t = row["test"]
        print(f"{name:40s} {row['val']['S_agg']:10.6f} {t['S_agg']:10.6f} {t['S_agg_sorted']:11.6f} {100*t['adjacent_inversion_rate']:9.3f} {t.get('log1p_osc_mae', float('nan')):9.5f}")
    for row in report["blends"]:
        t = row["val_weighted_test"]
        label = "+".join(row["members"]) + f" w={row['val_chosen_weights']}"
        print(f"{label[:40]:40s} {row['val_weighted_val']['S_agg']:10.6f} {t['S_agg']:10.6f} {t['S_agg_sorted']:11.6f} {100*t['adjacent_inversion_rate']:9.3f} {t.get('log1p_osc_mae', float('nan')):9.5f}  (equal-weight test {row['equal_weight_test']['S_agg']:.6f})")
    for note in notes:
        print("NOTE:", note)


if __name__ == "__main__":
    main()
