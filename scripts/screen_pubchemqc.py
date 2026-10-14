#!/usr/bin/env python3
"""Screen packed PubChemQC geometries for pretraining pathologies (Stele KNOW-242).

The 10-epoch pretrain (job 22800889) had epoch-mean train losses up to 4.5e10,
and ViSNet is known to emit million-scale outputs on disconnected species
(KNOW-114). This flags, per molecule:

  graph_disconnected     >1 connected component in the model's radius graph
                         (atoms closer than --cutoff, 5 A for our ViSNet)
  covalent_multifragment >1 covalently bonded fragment (d < r_i + r_j + 0.4 A)
  short_contact          any interatomic distance below --min-distance
  single_atom            fewer than two atoms
  s1_outlier             S1 below --s1-min or above --s1-max eV

and writes an exclusion CSV (molecule_key column, as train.py
exclude_keys_path expects) that also carries every row of --base-exclusions,
so one file replaces the QM9GWBSE holdout exclusions in the pretrain config.

With --checkpoint, it also runs that ViSNet checkpoint on CPU over flagged
and random unflagged molecules and reports how many predict |S| > --pathological-ev,
to check that the flags catch the molecules that blow up the model.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse.csgraph import connected_components

# Cordero et al. 2008 covalent radii (A) for Z = 1..86; index 0 unused.
COVALENT_RADII = np.array([
    1.50,
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
])
BOND_TOLERANCE = 0.4
FLAGS = ("graph_disconnected", "covalent_multifragment", "short_contact", "single_atom", "s1_outlier")


def molecule_geometry_stats(numbers: np.ndarray, positions: np.ndarray, cutoff: float) -> tuple[int, int, float]:
    """Return (radius-graph components, covalent fragments, minimum distance) for one molecule."""
    n = int(numbers.size)
    if n < 2:
        return 1, 1, float("inf")
    pos = np.asarray(positions, dtype=np.float64)
    dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    radii = COVALENT_RADII[np.clip(numbers.astype(np.int64), 0, COVALENT_RADII.size - 1)]
    graph_parts = connected_components(dist < cutoff, directed=False)[0]
    bonded = dist < (radii[:, None] + radii[None, :] + BOND_TOLERANCE)
    covalent_parts = connected_components(bonded, directed=False)[0]
    return int(graph_parts), int(covalent_parts), float(dist.min())


def _screen_range(args: tuple[str, int, int, float]) -> np.ndarray:
    h5_path, start, stop, cutoff = args
    with h5py.File(h5_path, "r") as h5:
        offsets = h5["atom_offsets"][start : stop + 1]
        numbers = h5["numbers"][offsets[0] : offsets[-1]]
        positions = h5["positions"][offsets[0] : offsets[-1]]
    offsets = offsets - offsets[0]
    out = np.empty((stop - start, 4), dtype=np.float64)
    for i in range(stop - start):
        a, b = offsets[i], offsets[i + 1]
        graph_parts, covalent_parts, min_dist = molecule_geometry_stats(numbers[a:b], positions[a:b], cutoff)
        out[i] = (b - a, graph_parts, covalent_parts, min_dist)
    return out


def flag_rows(stats: np.ndarray, s1: np.ndarray, min_distance: float, s1_min: float, s1_max: float) -> dict[str, np.ndarray]:
    natoms, graph_parts, covalent_parts, min_dist = stats.T
    return {
        "graph_disconnected": graph_parts > 1,
        "covalent_multifragment": covalent_parts > 1,
        "short_contact": min_dist < min_distance,
        "single_atom": natoms < 2,
        "s1_outlier": (s1 < s1_min) | (s1 > s1_max),
    }


def read_base_exclusions(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, newline="", encoding="utf-8-sig") as stream:
        return [row["molecule_key"].strip() for row in csv.DictReader(stream) if row.get("molecule_key", "").strip()]


def probe_checkpoint(h5_path, checkpoint, indices_by_group, pathological_ev, batch_size=256):
    """Run a ViSNet checkpoint on CPU over a few molecule groups; summarize output magnitudes."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from gnn_excited.models.visnet import build_visnet

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_visnet(**state["model_kwargs"])
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    report = {}
    with h5py.File(h5_path, "r") as h5:
        offsets = h5["atom_offsets"][()]
        for group, indices in indices_by_group.items():
            data = []
            for m in indices:
                a, b = int(offsets[m]), int(offsets[m + 1])
                data.append(Data(z=torch.as_tensor(h5["numbers"][a:b], dtype=torch.long),
                                 pos=torch.as_tensor(h5["positions"][a:b], dtype=torch.float32)))
            peaks = []
            with torch.no_grad():
                for batch in DataLoader(data, batch_size=batch_size):
                    out = model(batch.z, batch.pos, batch.batch)
                    out = out[0] if isinstance(out, tuple) else out
                    peaks.append(out.view(batch.num_graphs, -1).abs().amax(dim=1))
            peak = torch.cat(peaks).numpy() if peaks else np.empty(0)
            report[group] = {
                "n": int(peak.size),
                "pathological": int((peak > pathological_ev).sum()),
                "max_abs_prediction": float(peak.max()) if peak.size else None,
                "median_abs_prediction": float(np.median(peak)) if peak.size else None,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed-h5", required=True)
    parser.add_argument("--base-exclusions", default=None, help="exclusion CSV to carry over (molecule_key column)")
    parser.add_argument("--exclusions-out", required=True)
    parser.add_argument("--flagged-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--min-distance", type=float, default=0.5)
    parser.add_argument("--s1-min", type=float, default=0.5)
    parser.add_argument("--s1-max", type=float, default=10.0)
    parser.add_argument("--exclude-flags", default=",".join(FLAGS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=20000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--probe-per-group", type=int, default=5000)
    parser.add_argument("--pathological-ev", type=float, default=50.0)
    args = parser.parse_args()

    with h5py.File(args.packed_h5, "r") as h5:
        n_molecules = int(h5["atom_offsets"].shape[0] - 1)
        keys = np.array([k.decode() if isinstance(k, bytes) else str(k) for k in h5["molecule_key"][()]])
        s1 = h5["S_eV"][:, 0].astype(np.float64)
    bounds = [(args.packed_h5, s, min(s + args.chunk, n_molecules), args.cutoff) for s in range(0, n_molecules, args.chunk)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        stats = np.concatenate(list(pool.map(_screen_range, bounds)))
    flags = flag_rows(stats, s1, args.min_distance, args.s1_min, args.s1_max)
    exclude_flags = [f for f in args.exclude_flags.split(",") if f]
    unknown = sorted(set(exclude_flags) - set(FLAGS))
    if unknown:
        raise SystemExit(f"unknown --exclude-flags {unknown}; choose from {FLAGS}")
    excluded = np.zeros(n_molecules, dtype=bool)
    for name in exclude_flags:
        excluded |= flags[name]
    any_flag = np.zeros(n_molecules, dtype=bool)
    for mask in flags.values():
        any_flag |= mask

    flagged_idx = np.flatnonzero(any_flag)
    with open(args.flagged_out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "pack_index", "flags", "natoms", "graph_components", "covalent_fragments", "min_distance", "S1_eV", "excluded"])
        for m in flagged_idx:
            names = "|".join(name for name in FLAGS if flags[name][m])
            writer.writerow([keys[m], int(m), names, int(stats[m, 0]), int(stats[m, 1]), int(stats[m, 2]),
                             f"{stats[m, 3]:.4f}", f"{s1[m]:.4f}", int(excluded[m])])

    base = read_base_exclusions(args.base_exclusions)
    screened = [str(k) for k in keys[excluded]]
    with open(args.exclusions_out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", "reason"])
        seen = set()
        for key in base:
            if key not in seen:
                writer.writerow([key, "qm9gwbse_holdout"])
                seen.add(key)
        for m in np.flatnonzero(excluded):
            if keys[m] not in seen:
                writer.writerow([keys[m], "|".join(name for name in exclude_flags if flags[name][m])])
                seen.add(keys[m])

    report = {
        "packed_h5": args.packed_h5,
        "molecules": n_molecules,
        "cutoff_angstrom": args.cutoff,
        "flag_counts": {name: int(mask.sum()) for name, mask in flags.items()},
        "any_flag": int(any_flag.sum()),
        "exclude_flags": exclude_flags,
        "screen_excluded": int(excluded.sum()),
        "base_exclusions": len(base),
        "total_exclusion_rows": len(seen),
        "natoms_max": int(stats[:, 0].max()),
    }
    if args.checkpoint:
        rng = np.random.default_rng(17)
        groups = {}
        for name in FLAGS:
            idx = np.flatnonzero(flags[name])
            if idx.size:
                groups[name] = np.sort(rng.choice(idx, size=min(idx.size, args.probe_per_group), replace=False))
        clean = np.flatnonzero(~any_flag)
        groups["unflagged_random"] = np.sort(rng.choice(clean, size=min(clean.size, args.probe_per_group), replace=False))
        probe = {"checkpoint": args.checkpoint, "pathological_ev": args.pathological_ev}
        try:
            probe["groups"] = probe_checkpoint(args.packed_h5, args.checkpoint, groups, args.pathological_ev)
        except Exception as exc:  # diagnostic only; the exclusion CSV above is the deliverable
            probe["error"] = f"{type(exc).__name__}: {exc}"
        report["checkpoint_probe"] = probe
    Path(args.report_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
