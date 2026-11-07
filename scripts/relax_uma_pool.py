#!/usr/bin/env python3
"""Relax every molecule of a pool .npz with the UMA-S interatomic potential (TASK-291 E3, TASK-280).

Runs in envs/uma-fairchem. The pretrained UMA-S checkpoint is used as an omol-task force field
(wB97M-V-level forces), with per-molecule charge/spin from the pool when present (default 0 / 1).
ASE FIRE runs to --fmax or --steps. Writes the relaxed pool plus <out>.relax.csv with the step count
and final max force; unconverged molecules keep their last geometry and are flagged there.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/uma/checkpoints/uma-s-1p2p1.pt")
    parser.add_argument("--fmax", type=float, default=0.05, help="eV/Angstrom")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    from ase import Atoms
    from ase.optimize import FIRE
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    # "batch" keeps MoLE unmerged: every molecule has a different composition.
    calc = FAIRChemCalculator(load_predict_unit(args.checkpoint, inference_settings="batch", device="cuda"), task_name="omol")
    pool = dict(np.load(args.pool, allow_pickle=False))
    offsets, keys = pool["atom_offsets"], [str(k) for k in pool["molecule_key"]]
    charge = pool.get("charge", np.zeros(len(keys), dtype=np.int64))
    spin = pool.get("spin", np.ones(len(keys), dtype=np.int64))
    positions = pool["positions"].astype(np.float64).copy()
    log = []
    started = time.perf_counter()
    for i, key in enumerate(keys):
        a, b = int(offsets[i]), int(offsets[i + 1])
        atoms = Atoms(numbers=pool["numbers"][a:b].astype(int), positions=positions[a:b])
        atoms.info.update(charge=int(charge[i]), spin=int(spin[i]))
        atoms.calc = calc
        opt = FIRE(atoms, logfile=None)
        converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
        positions[a:b] = atoms.get_positions()
        log.append((key, opt.nsteps, float(np.abs(atoms.get_forces()).max()), int(converged)))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(keys)} relaxed, {time.perf_counter() - started:.0f}s", flush=True)
    out = {**pool, "positions": positions.astype(np.float32)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    with open(Path(args.out).with_suffix(".relax.csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "steps", "max_force_eV_A", "converged"])
        writer.writerows(log)
    print(f"RELAXED {len(keys)} molecules, {sum(r[3] for r in log)} converged, {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
