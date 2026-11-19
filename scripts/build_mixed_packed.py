#!/usr/bin/env python3
"""QM9GWBSE with every molecule at two geometries: B3LYP and SMILES-built GFN2-xTB (TASK-320, wave G).

The xTB-trained model (KNOW-315) is much better on cheap geometries but loses 0.038 eV on B3LYP ones.
This packs both geometry sets into one training file so one model sees both: the B3LYP file
(data/processed/qm9gwbse/qm9gwbse_packed.npz) unchanged, followed by the xTB file
(scripts/build_xtb_packed.py; 132,069 molecules) with molecule keys suffixed ":xtb". Labels, splits and
every other per-molecule field are copied, so each xTB copy stays in its molecule's split. Adds
geometry_source ("b3lyp" / "xtb") and writes <out>.report.json with the counts per split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_xtb_packed import per_molecule

GEOMETRY = ("numbers", "positions", "atom_offsets")


def merge(b3: dict[str, np.ndarray], xtb: dict[str, np.ndarray], suffix: str = ":xtb") -> dict[str, np.ndarray]:
    """B3LYP molecules, then the xTB molecules (keys + suffix); per-molecule fields concatenated."""
    n_b, n_x = b3["molecule_key"].shape[0], xtb["molecule_key"].shape[0]
    out = {"numbers": np.concatenate([b3["numbers"], xtb["numbers"]]).astype(np.uint8),
           "positions": np.concatenate([b3["positions"], xtb["positions"]]).astype(np.float32),
           "atom_offsets": np.concatenate([b3["atom_offsets"], xtb["atom_offsets"][1:] + b3["atom_offsets"][-1]]).astype(np.int64)}
    for name, value in b3.items():
        if name in GEOMETRY:
            continue
        if per_molecule(name, value, n_b):
            other = xtb[name]
            if name == "molecule_key":
                other = np.char.add(other.astype(str), suffix)
                value = value.astype(str)
            out[name] = np.concatenate([value, other])
        else:
            out[name] = value
    out["geometry_source"] = np.array(["b3lyp"] * n_b + ["xtb"] * n_x)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b3lyp", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--xtb", default="data/processed/qm9gwbse_xtb/qm9gwbse_xtb_packed.npz")
    parser.add_argument("--out", default="data/processed/qm9gwbse_mixed/qm9gwbse_mixed_packed.npz")
    args = parser.parse_args()
    b3 = dict(np.load(args.b3lyp, allow_pickle=False))
    xtb = dict(np.load(args.xtb, allow_pickle=False))
    out = merge(b3, xtb)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    split = out["split"].astype(str)
    report = {name: {s: int(((split == s) & (out["geometry_source"] == name)).sum()) for s in ("train", "val", "test")}
              for name in ("b3lyp", "xtb")}
    report["molecules"] = int(out["molecule_key"].shape[0])
    report["atoms"] = int(out["numbers"].shape[0])
    Path(str(Path(args.out).with_suffix("")) + ".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
