#!/usr/bin/env python3
"""Few-shot GW-label sets on OE62H for the bet B follow-up (TASK-310, follows KNOW-307).

Past ~25k heavy-element TD-DFT labels, the GW head's OE62H error stops falling and keeps a +0.2 eV
bias. This measures what a few GW-level heavy-element labels do instead. The OE62H molecules (GW-BSE
S1-S5 labels in data/processed/external_testsets/labels.csv, geometries in pool.npz) are split into
--folds folds. For fold f, the other folds form the shot pool, and shot sets of size k (--ks) are
nested prefixes of one seeded permutation of that pool. Writes per (f, k):

  <out-dir>/fold{f}_k{k}.npz     packed shots (numbers, positions, atom_offsets, molecule_key,
                                  targets S1-S5 in eV, target_columns, split="train")
  <config-dir>/f{f}_k{k}.yaml    the --template fine-tuning config with these shots
plus <config-dir>/k0.yaml (the same fine-tuning without shots: the control), <config-dir>/configs.txt
(one config per line; the array job's index), and <out-dir>/folds.json (each fold's held-out keys).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

ENERGY = [f"S{i}_eV" for i in range(1, 6)]


def fold_assignment(keys: list[str], folds: int, seed: int) -> list[list[str]]:
    """Disjoint folds covering every key, from one seeded permutation."""
    order = np.random.default_rng(seed).permutation(len(keys))
    return [sorted(keys[i] for i in part) for part in np.array_split(order, folds)]


def nested_shots(pool: list[str], ks: list[int], seed: int) -> dict[int, list[str]]:
    """Shot sets of each size in ks: prefixes of one seeded permutation of the pool (so nested)."""
    order = [pool[i] for i in np.random.default_rng(seed).permutation(len(pool))]
    if max(ks) > len(order):
        raise ValueError(f"k={max(ks)} exceeds the shot pool of {len(order)} molecules")
    return {k: order[:k] for k in ks}


def pack_shots(keys: list[str], pool: dict[str, np.ndarray], index: dict[str, int], targets: dict[str, list[float]]) -> dict:
    rows = [index[k] for k in keys]
    offsets = pool["atom_offsets"]
    counts = np.array([offsets[r + 1] - offsets[r] for r in rows], dtype=np.int64)
    return {"numbers": np.concatenate([pool["numbers"][offsets[r]:offsets[r + 1]] for r in rows]).astype(np.uint8),
            "positions": np.concatenate([pool["positions"][offsets[r]:offsets[r + 1]] for r in rows]).astype(np.float32),
            "atom_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            "molecule_key": np.array(keys),
            "targets": np.array([targets[k] for k in keys], dtype=np.float32),
            "target_columns": np.array(ENERGY),
            "split": np.array(["train"] * len(keys))}


def run_config(template: dict, name: str, shots_path: str | None, out_root: str) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template))  # deep copy
    t = cfg["training"]
    t["checkpoint_path"] = f"checkpoints/fewshot_oe62h/{name}.pt"
    t["metrics_csv_path"] = f"{out_root}/{name}.metrics.csv"
    t["summary_json_path"] = f"{out_root}/{name}.summary.json"
    cfg["wandb"]["name"] = f"fewshot_oe62h_{name}"
    cfg["post_inference"]["out_csv"] = f"{out_root}/ext_{name}.csv"
    if shots_path is None:
        cfg.pop("shots", None)
    else:
        cfg["shots"]["packed_path"] = shots_path
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--pool", default="data/processed/external_testsets/pool.npz")
    parser.add_argument("--template", default="configs/uma_fewshot_oe62h.yaml")
    parser.add_argument("--set", default="OE62H")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ks", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/processed/oe62h_fewshot")
    parser.add_argument("--config-dir", default="runs/fewshot_oe62h/configs")
    parser.add_argument("--run-dir", default="runs/fewshot_oe62h")
    args = parser.parse_args()
    with open(args.labels, newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["set"] == args.set]
    targets = {r["molecule_key"]: [float(r[c]) for c in ENERGY] for r in rows if all(r[c] for c in ENERGY)}
    pool = dict(np.load(args.pool, allow_pickle=False))
    index = {str(k): i for i, k in enumerate(pool["molecule_key"])}
    keys = sorted(k for k in targets if k in index)
    folds = fold_assignment(keys, args.folds, args.seed)
    template = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))

    out, config_dir = Path(args.out_dir), Path(args.config_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    configs = [config_dir / "k0.yaml"]
    configs[0].write_text(yaml.safe_dump(run_config(template, "k0", None, args.run_dir), sort_keys=False), encoding="utf-8")
    manifest = {"set": args.set, "molecules": len(keys), "ks": args.ks, "seed": args.seed, "folds": []}
    for f, held_out in enumerate(folds):
        shot_pool = sorted(set(keys) - set(held_out))
        shots = nested_shots(shot_pool, args.ks, args.seed + 1 + f)
        manifest["folds"].append({"held_out": held_out, "shots": {str(k): v for k, v in shots.items()}})
        for k, chosen in shots.items():
            path = out / f"fold{f}_k{k}.npz"
            np.savez(path, **pack_shots(chosen, pool, index, targets))
            cfg_path = config_dir / f"f{f}_k{k}.yaml"
            cfg_path.write_text(yaml.safe_dump(run_config(template, f"f{f}_k{k}", str(path), args.run_dir), sort_keys=False),
                                encoding="utf-8")
            configs.append(cfg_path)
    (out / "folds.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (config_dir / "configs.txt").write_text("\n".join(str(c) for c in configs) + "\n", encoding="utf-8")
    print(f"{len(keys)} {args.set} molecules, {args.folds} folds of {[len(f) for f in folds]}; {len(configs)} configs")


if __name__ == "__main__":
    main()
