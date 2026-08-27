#!/usr/bin/env python3
"""Convert an OMol25 converter manifest into a train.py-compatible manifest."""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="converter manifest CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    written = 0
    with open(args.input, newline="", encoding="utf-8") as src, open(args.output, "w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.writer(dst)
        writer.writerow(["molecule_key", "status", "natoms", "gap_eV", "homo_eV"])
        for row in reader:
            writer.writerow([row["id"], "ok", row["natoms"], row["gap"], row["homo"]])
            written += 1
    print(f"wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
