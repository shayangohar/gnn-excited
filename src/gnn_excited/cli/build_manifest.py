from __future__ import annotations

import argparse
from pathlib import Path

from gnn_excited.data.qcdge import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a QCDGE excited-state manifest from an HDF5 dataset.')
    parser.add_argument('--hdf5', default='data/A_9.hdf5', help='Path to QCDGE HDF5 subset')
    parser.add_argument('--out', default='data/processed/a9_manifest_100.csv', help='Output CSV path')
    parser.add_argument('--max-count', type=int, default=None, help='Maximum molecules to inspect; omit for all')
    parser.add_argument('--singlets', type=int, default=1, help='Number of lowest singlet states to include')
    parser.add_argument('--triplets', type=int, default=0, help='Number of lowest triplet states to include')
    parser.add_argument('--progress-every', type=int, default=1000, help='Progress print interval; 0 disables progress')
    args = parser.parse_args()

    counts = build_manifest(
        Path(args.hdf5),
        Path(args.out),
        max_count=args.max_count,
        progress_every=args.progress_every,
        singlet_count=args.singlets,
        triplet_count=args.triplets,
    )
    print('wrote {}: {} ok, {} errors'.format(args.out, counts['ok'], counts['error']))
