#!/usr/bin/env python3
"""Parse ORCA STEOM-DLPNO-CCSD pilot outputs and compare with QM9GWBSE qsGW-BSE labels (TASK-262).

For each <key>.out in --out-dir: the last STEOM results block's roots (eV, percentage active
character) and the run time. ORCA removes duplicate roots ("PAIRWISE REDUNDANCY FOUND ... AND
REMOVED FOR IROOT=a and IROOT=b") and warns below 98% active character; a molecule is "reliable"
when its lowest five unique roots all reach that threshold. Per-state MAE / mean signed
difference / Pearson r against the qsGW-BSE S1-S5 labels are reported for all parsed molecules
and for the reliable ones, plus failure causes and timing (seconds vs heavy-atom count) to project
the cost of larger label campaigns.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

IROOT = re.compile(r"IROOT\s*=\s*(\d+):\s+(-?\d+\.\d+)\s+au\s+(-?\d+\.\d+)\s+eV")
ACTIVE = re.compile(r"Percentage Active Character\s+(-?\d+\.\d+)")
REDUNDANT = re.compile(r"PAIRWISE REDUNDANCY FOUND.*FOR IROOT=\s*(\d+) and IROOT=\s*(\d+)")
MIN_ACTIVE = 98.0  # ORCA warns "may have not converged with respect to active space" below this
FAILURES = {"EA-EOM": "EA=EOM-DLPNO-CCSD equations failed", "IP-EOM": "IP-EOM-DLPNO-CCSD equations failed",
            "EOM": "Solution of the EOM-DLPNO-CCSD equations failed"}


def parse_steom_roots(text: str) -> list[dict] | None:
    """Roots of the last STEOM results block in IROOT order: eV, active character (None if not
    printed) and whether ORCA removed the root as a duplicate of an earlier one."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if "STEOM" in line.upper() and "RESULT" in line.upper()]
    if not starts:
        return None
    roots: dict[int, dict] = {}
    current = None
    redundant = set()
    for line in lines[starts[-1]:]:
        match = IROOT.search(line)
        if match and int(match.group(1)) not in roots:
            current = int(match.group(1))
            roots[current] = {"eV": float(match.group(3)), "active": None}
        elif (match := ACTIVE.search(line)) and current is not None and roots[current]["active"] is None:
            roots[current]["active"] = float(match.group(1))
        elif match := REDUNDANT.search(line):
            redundant.add(int(match.group(2)))
    return [{**roots[i], "redundant": i in redundant} for i in sorted(roots)] or None


def parse_steom(text: str, nroots: int = 5) -> list[float] | None:
    """Lowest nroots energies (eV, ascending) of the unique roots of the last STEOM results block."""
    roots = parse_steom_roots(text)
    energies = sorted(r["eV"] for r in roots or [] if not r["redundant"])
    return energies[:nroots] if len(energies) >= nroots else None


def is_reliable(roots: list[dict], nroots: int = 5) -> bool:
    """The lowest nroots unique roots all have at least MIN_ACTIVE active character."""
    unique = sorted((r for r in roots if not r["redundant"]), key=lambda r: r["eV"])[:nroots]
    return len(unique) == nroots and all(r["active"] is not None and r["active"] >= MIN_ACTIVE for r in unique)


def failure_cause(text: str) -> str:
    for cause, marker in FAILURES.items():
        if marker in text:
            return cause
    return "no STEOM results" if "TOTAL RUN TIME" in text else "incomplete (killed or timed out)"


def state_stats(steom: np.ndarray, gw: np.ndarray) -> list[dict]:
    diff = steom - gw
    return [{"state": f"S{k + 1}", "mae_eV": float(np.abs(diff[:, k]).mean()),
             "mean_signed_eV": float(diff[:, k].mean()),
             "pearson_r": float(np.corrcoef(steom[:, k], gw[:, k])[0, 1]) if len(gw) > 2 else None}
            for k in range(gw.shape[1])]


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
        text = out.read_text(errors="replace")
        roots = parse_steom_roots(text)
        energies = parse_steom(text)
        if energies is None or key not in labels:
            cause = failure_cause(text) if roots is None else f"{sum(not r['redundant'] for r in roots)} unique roots"
            failed.append({"key": key, "exit": timing.get("exit"), "cause": cause, "tail": text[-400:]})
            continue
        unique = sorted((r for r in roots if not r["redundant"]), key=lambda r: r["eV"])[:5]
        rows.append({"key": key, "heavy": heavy.get(key), "seconds": timing.get("seconds"), "steom": energies,
                     "gw": labels[key], "redundant": sum(r["redundant"] for r in roots),
                     "min_active": min((r["active"] for r in unique if r["active"] is not None), default=None),
                     "reliable": is_reliable(roots)})
    report = {"parsed": len(rows), "failed": len(failed), "min_active_character": MIN_ACTIVE,
              "failure_causes": dict(Counter(f["cause"] for f in failed)),
              "parsed_with_duplicate_roots": sum(r["redundant"] > 0 for r in rows),
              "reliable": sum(r["reliable"] for r in rows), "failures": failed[:5]}
    if rows:
        steom = np.array([r["steom"] for r in rows])
        gw = np.array([r["gw"] for r in rows])
        diff = steom - gw
        report["per_state"] = state_stats(steom, gw)
        report["mae_all_eV"] = float(np.abs(diff).mean())
        report["mae_after_mean_shift_eV"] = float(np.abs(diff - diff.mean(axis=0)).mean())
        ok = np.array([r["reliable"] for r in rows])
        if ok.sum() > 2:
            report["reliable_per_state"] = state_stats(steom[ok], gw[ok])
            report["reliable_mae_all_eV"] = float(np.abs(diff[ok]).mean())
            report["reliable_mae_after_mean_shift_eV"] = float(np.abs(diff[ok] - diff[ok].mean(axis=0)).mean())
        secs = np.array([r["seconds"] or np.nan for r in rows], float)
        report["seconds"] = {"median": float(np.nanmedian(secs)), "max": float(np.nanmax(secs))}
        report["seconds_by_heavy"] = {int(h): float(np.nanmedian(secs[[r["heavy"] == h for r in rows]]))
                                      for h in sorted({r["heavy"] for r in rows})}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with open(Path(args.report).with_suffix(".csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "heavy_atoms", "seconds", "duplicate_roots", "min_active", "reliable",
                         *[f"steom_S{i}_eV" for i in range(1, 6)], *[f"gw_S{i}_eV" for i in range(1, 6)]])
        for r in rows:
            writer.writerow([r["key"], r["heavy"], r["seconds"], r["redundant"], r["min_active"], int(r["reliable"]),
                             *r["steom"], *r["gw"]])
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
