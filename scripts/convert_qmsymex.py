#!/usr/bin/env python3
"""Convert QM-symex xyz files to project HDF5 + manifest.

Source: Liang et al. Sci Data 7, 400 (2020), Figshare doi 10.6084/m9.figshare.12815276.
Per file: N atoms (Element x y z q), HOMO line, then 10 transition lines
  n|Ssym Es_eV wl fs 0.000|o o c|...|Tsym Et_eV wl ft 2.000|o o c|...
Output schema v1 (units eV, oscillator strengths dimensionless):
  HDF5  /m/<id>/numbers(i32[N]) positions(f64[N,3]) S_eV(f64[10]) T_eV(f64[10]) S_f(f64[10])
  CSV   id,natoms,S1_eV..S10_eV,T1_eV..T10_eV,S1_f..S10_f,split
Split = hash bucket (default 95/5 train/val); pretraining corpus, test on QM9GWBSE.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np

ELEMENTS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
}
N_STATES = 10


def split_for(mol_id: str, val_fraction: float = 0.05) -> str:
    digest = hashlib.sha256(mol_id.encode()).hexdigest()
    return "val" if int(digest[:16], 16) % 10_000 < val_fraction * 10_000 else "train"


def parse_state_field(field: str):
    """Parse a 5-token state header (sym E wl f spin); None for orbital triples."""
    tokens = field.strip().split()
    if len(tokens) != 5:
        return None
    try:
        return tokens[0], float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
    except ValueError:
        return None


def parse_qmsymex_file(path: Path):
    """Return (numbers, positions, S_eV[10], T_eV[10], S_f[10]) or None if unparseable."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return None
    try:
        natoms = int(lines[0].strip())
    except ValueError:
        return None
    if len(lines) < 2 + natoms + 1 + N_STATES:
        return None
    numbers, positions = [], []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        if len(parts) < 5 or parts[0] not in ELEMENTS:
            return None
        numbers.append(ELEMENTS[parts[0]])
        try:
            positions.append([float(v) for v in parts[1:4]])
        except ValueError:
            return None
    body = lines[2 + natoms:]
    homo_idx = next((i for i, l in enumerate(body) if l.strip().startswith("HOMO")), None)
    if homo_idx is None:
        return None
    transitions = body[homo_idx + 1:homo_idx + 1 + N_STATES]
    if len(transitions) < N_STATES:
        return None
    S_eV, T_eV, S_f = [], [], []
    for tline in transitions:
        fields = tline.split("|")
        singlet = triplet = None
        for field in fields[1:]:
            parsed = parse_state_field(field)
            if parsed is None:
                continue
            _, energy, _, osc, spin = parsed
            if abs(spin) < 0.5 and singlet is None:
                singlet = (energy, osc)
            elif abs(spin - 2.0) < 0.5 and triplet is None:
                triplet = (energy,)
        if singlet is None or triplet is None:
            return None
        S_eV.append(singlet[0])
        S_f.append(singlet[1])
        T_eV.append(triplet[0])
    S_eV = np.asarray(S_eV, dtype=np.float64)
    T_eV = np.asarray(T_eV, dtype=np.float64)
    if not (np.isfinite(S_eV).all() and np.isfinite(T_eV).all()):
        return None
    if bool((np.diff(S_eV) < 0).any()) or bool((np.diff(T_eV) < 0).any()):
        return None
    return (
        np.asarray(numbers, dtype=np.int32),
        np.asarray(positions, dtype=np.float64),
        S_eV, T_eV, np.asarray(S_f, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz-dir", required=True)
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    files = sorted(Path(args.xyz_dir).glob("QM_symex_*.xyz"))
    if not files:
        raise SystemExit(f"no QM_symex_*.xyz under {args.xyz_dir}")
    if args.limit:
        files = files[:args.limit]
    print(f"files: {len(files)}", flush=True)

    header = (["id", "natoms"]
              + [f"S{s}_eV" for s in range(1, N_STATES + 1)]
              + [f"T{s}_eV" for s in range(1, N_STATES + 1)]
              + [f"S{s}_f" for s in range(1, N_STATES + 1)]
              + ["split"])
    kept = scanned = skipped_parse = skipped_size = 0
    out_h5 = Path(args.out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h5, open(args.manifest, "w", encoding="utf-8") as csv:
        root = h5.create_group("m")
        root.attrs["schema"] = "qmsymex-st-v1"
        root.attrs["units"] = "eV"
        csv.write(",".join(header) + "\n")
        for path in files:
            scanned += 1
            if scanned % 20_000 == 0:
                print(f"scanned={scanned} kept={kept}", flush=True)
            parsed = parse_qmsymex_file(path)
            if parsed is None:
                skipped_parse += 1
                continue
            numbers, positions, S_eV, T_eV, S_f = parsed
            if len(numbers) > args.max_atoms:
                skipped_size += 1
                continue
            mol_id = path.stem
            grp = root.create_group(mol_id)
            grp.create_dataset("numbers", data=numbers)
            grp.create_dataset("positions", data=positions)
            grp.create_dataset("S_eV", data=S_eV)
            grp.create_dataset("T_eV", data=T_eV)
            grp.create_dataset("S_f", data=S_f)
            split = split_for(mol_id, args.val_fraction)
            grp.attrs["split"] = split
            row = ([mol_id, str(len(numbers))]
                   + [f"{v:.6f}" for v in S_eV]
                   + [f"{v:.6f}" for v in T_eV]
                   + [f"{v:.6f}" for v in S_f]
                   + [split])
            csv.write(",".join(row) + "\n")
            kept += 1
    print(f"done: scanned={scanned} kept={kept} skipped_parse={skipped_parse} skipped_size={skipped_size}", flush=True)


if __name__ == "__main__":
    main()
