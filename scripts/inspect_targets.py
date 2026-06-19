from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _read_ok_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row.get("status") == "ok"]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p05": _quantile(values, 0.05),
        "median": _quantile(values, 0.5),
        "mean": sum(values) / len(values),
        "p95": _quantile(values, 0.95),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect S1 target distributions in a QCDGE manifest.")
    parser.add_argument("manifest", help="Manifest CSV from build-qcdge-manifest")
    args = parser.parse_args()

    rows = _read_ok_rows(Path(args.manifest))
    if not rows:
        raise SystemExit(f"No ok rows found in {args.manifest}")

    s1_ev = [float(row["S1_eV"]) for row in rows]
    s1_f = [float(row["S1_f"]) for row in rows]
    atom_counts = [float(row["atom_count"]) for row in rows]
    nonzero_f = sum(value > 0 for value in s1_f)

    print(f"rows: {len(rows)}")
    print(f"oscillator_strength_nonzero: {nonzero_f} ({nonzero_f / len(rows):.1%})")
    for name, values in [("S1_eV", s1_ev), ("S1_f", s1_f), ("atom_count", atom_counts)]:
        stats = _summary(values)
        print(
            f"{name}: min={stats['min']:.6g} p05={stats['p05']:.6g} "
            f"median={stats['median']:.6g} mean={stats['mean']:.6g} "
            f"p95={stats['p95']:.6g} max={stats['max']:.6g}"
        )


if __name__ == "__main__":
    main()
