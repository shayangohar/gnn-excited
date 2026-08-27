#!/usr/bin/env python3
"""Build a beyond-HCNOF probe manifest from the OMol25 converter manifest."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ALLOWED = {"H", "C", "N", "O", "F"}
SYMBOL = re.compile(r"[A-Z][a-z]?")


def has_beyond_hcnof(formula: str) -> bool:
    symbols = {match.group(0) for match in SYMBOL.finditer(formula)}
    return bool(symbols - ALLOWED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="converter manifest CSV")
    parser.add_argument("--output", required=True, help="probe manifest CSV")
    parser.add_argument("--max", type=int, default=0, help="cap probe size")
    args = parser.parse_args()

    kept = 0
    element_counts: dict[str, int] = {}
    with open(args.input, newline="", encoding="utf-8") as src, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.writer(dst)
        writer.writerow(["molecule_key", "status", "natoms", "gap_eV"])
        for row in reader:
            formula = row["formula"]
            if not has_beyond_hcnof(formula):
                continue
            for symbol in {m.group(0) for m in SYMBOL.finditer(formula)}:
                element_counts[symbol] = element_counts.get(symbol, 0) + 1
            writer.writerow([row["id"], "ok", row["natoms"], row["gap"]])
            kept += 1
            if args.max and kept >= args.max:
                break
    print(f"probe molecules: {kept}")
    print("beyond-HCNOF element presence (mol counts):",
          dict(sorted(element_counts.items(), key=lambda kv: -kv[1])[:15]))


if __name__ == "__main__":
    main()
