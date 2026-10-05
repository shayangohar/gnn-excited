#!/usr/bin/env python3
"""Merge PubChemQC chunk HDF5s + manifests into the final training set.

Verifies per-chunk schemas, copies groups (names are globally unique:
pqc_<cid>), concatenates manifests (single header), and reports split
balance + label stats. Overwrites the (corrupt, partial) train_st.* outputs
from the timed-out single-pass run.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

EXPECTED_SCHEMA = "pubchemqc-st-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chunks", default="c0,c1,c2,c3,c4,c5,c6,c7")
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    names = args.chunks.split(",")
    total = 0
    splits: dict[str, int] = {}
    with h5py.File(args.out_h5, "w") as out:
        root = out.create_group("m")
        root.attrs["schema"] = EXPECTED_SCHEMA
        root.attrs["units"] = "eV"
        with open(args.manifest, "w", newline="", encoding="utf-8") as csvfile:
            writer = None
            for name in names:
                src = h5py.File(chunk_dir / f"train_st_{name}.h5", "r")
                try:
                    assert src["m"].attrs["schema"] == EXPECTED_SCHEMA, name
                    for key in src["m"]:
                        src.copy(src["m"][key], root, name=key)
                        total += 1
                finally:
                    src.close()
                with open(chunk_dir / f"train_st_{name}_manifest.csv", newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
                        splits[row["split"]] = splits.get(row["split"], 0) + 1
    print(f"merged molecules: {total}", flush=True)
    print(f"splits: {splits}", flush=True)
    with h5py.File(args.out_h5, "r") as h5:
        keys = list(h5["m"])[:20000]
        S = np.array([h5["m"][k]["S_eV"][()] for k in keys])
        print(f"sample={len(keys)} S range {S.min():.3f}..{S.max():.3f} "
              f"S1 mean {S[:, 0].mean():.3f} S10 mean {S[:, 9].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
