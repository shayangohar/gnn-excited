from __future__ import annotations

import argparse
from pathlib import Path

from gnn_excited.data.qcdge import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a QCDGE S1 manifest from A_9.hdf5.")
    parser.add_argument("--hdf5", default="data/A_9.hdf5", help="Path to QCDGE HDF5 subset")
    parser.add_argument("--out", default="data/processed/a9_manifest_100.csv", help="Output CSV path")
    parser.add_argument("--max-count", type=int, default=100, help="Maximum molecules to inspect")
    args = parser.parse_args()

    counts = build_manifest(Path(args.hdf5), Path(args.out), max_count=args.max_count)
    print(f"wrote {args.out}: {counts['ok']} ok, {counts['error']} errors")
