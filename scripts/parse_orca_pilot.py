#!/usr/bin/env python3
"""Parse ORCA STEOM-DLPNO-CCSD pilot outputs and compare with QM9GWBSE qsGW-BSE labels (TASK-262).

For each <key>.out in --out-dir: the last STEOM results block's IROOT energies (eV) and the run
time. Reports per-state MAE / mean signed difference / Pearson r against the qsGW-BSE S1-S5
labels in the QM9GWBSE manifest, plus timing (seconds vs heavy-atom count) to project the cost
of larger label campaigns.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

IROOT = re.compile(r"IROOT\s*=\s*(\d+):\s+(-?\d+\.\d+)\s+au\s+(-?\d+\.\d+)\s+eV")


def parse_steom(text: str, nroots: int = 5) -> list[float] | None:
    """Energies (eV) of the last block of IROOT lines that follows a STEOM results header."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if "STEOM" in line.upper() and "RESULT" in line.upper()]
    if not starts:
        return None
    energies: dict[int, float] = {}
    for line in lines[starts[-1]:]:
        match = IROOT.search(line)
        if match:
            energies.setdefault(int(match.group(1)), float(match.group(3)))
            if len(energies) == nroots:
                break
    if len(energies) < nroots:
        return None
    return [energies[i] for i in range(1, nroots + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/processed/orca_pilot/out")
    parser.add_argument("--manifest", default="data/processed/qm9gwbse/manifest.csv")
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--report", default="runs/orca_pilot_report.json")
    args = parser.parse_args()

    labels = {r["molecule_key"]: [float(r[f"S{i}_eV"]) for i in range(1, 6)]
              for r in csv.DictReader(open(args.manifest, newline="", encoding="utf-8")) if r.get("status") == "ok"}
    q = np.load(args.packed, allow_pickle=False)
    heavy_all = np.add.reduceat((q["numbers"] > 1).astype(int), q["atom_offsets"][:-1])
    heavy = {str(k): int(h) for k, h in zip(q["molecule_key"], heavy_all)}
    rows, failed = [], []
    for out in sorted(Path(args.out_dir).glob("*.out")):
        key = out.stem
        timing_path = out.with_suffix(".timing.json")
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        energies = parse_steom(out.read_text(errors="replace"))
        if energies is None or key not in labels:
            failed.append({"key": key, "exit": timing.get("exit"), "tail": out.read_text(errors="replace")[-400:]})
            continue
        rows.append({"key": key, "heavy": heavy.get(key), "seconds": timing.get("seconds"), "steom": energies, "gw": labels[key]})
    report = {"parsed": len(rows), "failed": len(failed), "failures": failed[:5]}
    if rows:
        steom = np.array([r["steom"] for r in rows])
        gw = np.array([r["gw"] for r in rows])
        diff = steom - gw
        report["per_state"] = [{"state": f"S{k + 1}", "mae_eV": float(np.abs(diff[:, k]).mean()),
                                "mean_signed_eV": float(diff[:, k].mean()),
                                "pearson_r": float(np.corrcoef(steom[:, k], gw[:, k])[0, 1]) if len(rows) > 2 else None}
                               for k in range(5)]
        report["mae_all_eV"] = float(np.abs(diff).mean())
        report["mae_after_mean_shift_eV"] = float(np.abs(diff - diff.mean(axis=0)).mean())
        secs = np.array([r["seconds"] or np.nan for r in rows], float)
        report["seconds"] = {"median": float(np.nanmedian(secs)), "max": float(np.nanmax(secs))}
        report["seconds_by_heavy"] = {int(h): float(np.nanmedian(secs[[r["heavy"] == h for r in rows]]))
                                      for h in sorted({r["heavy"] for r in rows})}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with open(Path(args.report).with_suffix(".csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "heavy_atoms", "seconds", *[f"steom_S{i}_eV" for i in range(1, 6)], *[f"gw_S{i}_eV" for i in range(1, 6)]])
        for r in rows:
            writer.writerow([r["key"], r["heavy"], r["seconds"], *r["steom"], *r["gw"]])
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
