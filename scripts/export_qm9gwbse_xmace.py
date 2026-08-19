#!/usr/bin/env python3
"""Export a deterministic split-stratified QM9GWBSE subset as X-MACE extxyz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from gnn_excited.data.electronic_descriptors import _selected_rows


ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def export(hdf5_path: Path, manifest_path: Path, output_dir: Path, limit: int, seed: int) -> dict[str, object]:
    rows = _selected_rows(manifest_path, limit, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: output_dir / f"{split}.xyz" for split in ("train", "val", "test")}
    streams = {split: path.open("w", encoding="utf-8") for split, path in paths.items()}
    counts = {split: 0 for split in paths}
    try:
        with h5py.File(hdf5_path, "r") as source:
            for row in rows:
                split = row.get("random_split", "")
                if split not in streams:
                    continue
                ground = source[row["molecule_key"]]["ground_state"]
                labels = ground["labels"][()]
                coords = ground["coords"][()]
                energies = " ".join(f"{float(row[f'S{state}_eV']):.10f}" for state in range(1, 6))
                stream = streams[split]
                stream.write(f"{len(labels)}\n")
                stream.write(
                    'Properties=species:S:1:pos:R:3 '
                    f'REF_energy="{energies}" molecule_key={row["molecule_key"]} '
                    'config_type=Default pbc="F F F"\n'
                )
                for label, (x, y, z) in zip(labels, coords):
                    stream.write(f"{ELEMENTS[int(label)]} {x:.10f} {y:.10f} {z:.10f}\n")
                counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    summary = {
        "schema_version": "qm9gwbse-xmace-extxyz-v1",
        "source_hdf5": str(hdf5_path),
        "source_manifest": str(manifest_path),
        "selection": {"type": "split-stratified deterministic hash", "seed": seed, "limit": limit},
        "counts": counts,
        "paths": {split: str(path) for split, path in paths.items()},
        "energy_key": "REF_energy",
        "energy_units": "eV",
        "states": ["S1", "S2", "S3", "S4", "S5"],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(export(args.hdf5, args.manifest, args.output_dir, args.limit, args.seed), indent=2))


if __name__ == "__main__":
    main()
