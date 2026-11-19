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

Wave G cross-tests (TASK-320): --set OE62L draws the shots from OE62L instead (its effect on OE62H shows
whether the OE62H gain is a dataset-level offset). --exclude-element S replaces the folds with --draws
independent shot sets drawn only from molecules without S, so every S molecule is unseen: does the S
error still fall? --tag prefixes run names (checkpoints, CSVs, W&B), --no-control skips k0.yaml, and
--wandb-group overrides the template's W&B group.
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


def exclusion_draws(keys: list[str], elements: dict[str, set[str]], element: str, ks: list[int], draws: int,
                    seed: int) -> list[dict]:
    """Shot sets drawn only from molecules without `element`; every other molecule is held out."""
    pool = sorted(k for k in keys if element not in elements[k])
    out = []
    for d in range(draws):
        shots = nested_shots(pool, ks, seed + 100 + d)
        used = set(shots[max(ks)])
        out.append({"held_out": sorted(k for k in keys if k not in used), "shots": shots})
    return out


def run_config(template: dict, name: str, shots_path: str | None, out_root: str, wandb_group: str | None = None) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template))  # deep copy
    if wandb_group:
        cfg["wandb"]["group"] = wandb_group
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
    parser.add_argument("--tag", default="", help="prefix for run names, e.g. oe62l_")
    parser.add_argument("--no-control", action="store_true", help="do not write the k0 control config")
    parser.add_argument("--exclude-element", default=None, help="draw shots only from molecules without it")
    parser.add_argument("--draws", type=int, default=3, help="shot sets with --exclude-element")
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args()
    with open(args.labels, newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["set"] == args.set]
    targets = {r["molecule_key"]: [float(r[c]) for c in ENERGY] for r in rows if all(r[c] for c in ENERGY)}
    elements = {r["molecule_key"]: set(r["elements"].split("-")) for r in rows}
    pool = dict(np.load(args.pool, allow_pickle=False))
    index = {str(k): i for i, k in enumerate(pool["molecule_key"])}
    keys = sorted(k for k in targets if k in index)
    template = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))

    out, config_dir = Path(args.out_dir), Path(args.config_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    configs = []
    if not args.no_control:
        configs.append(config_dir / f"{args.tag}k0.yaml")
        configs[0].write_text(yaml.safe_dump(run_config(template, f"{args.tag}k0", None, args.run_dir, args.wandb_group),
                                             sort_keys=False), encoding="utf-8")
    manifest = {"set": args.set, "molecules": len(keys), "ks": args.ks, "seed": args.seed, "tag": args.tag}
    if args.exclude_element:
        # "folds" here are independent draws; each one's held_out is every molecule not used as a shot.
        units = exclusion_draws(keys, elements, args.exclude_element, args.ks, args.draws, args.seed)
        manifest.update(mode="exclude", exclude_element=args.exclude_element, unit="d")
    else:
        units = []
        for f, held_out in enumerate(fold_assignment(keys, args.folds, args.seed)):
            shot_pool = sorted(set(keys) - set(held_out))
            units.append({"held_out": held_out, "shots": nested_shots(shot_pool, args.ks, args.seed + 1 + f)})
        manifest.update(mode="folds", unit="f")
    manifest["folds"] = []
    for u, unit in enumerate(units):
        manifest["folds"].append({"held_out": unit["held_out"], "shots": {str(k): v for k, v in unit["shots"].items()}})
        for k, chosen in unit["shots"].items():
            name = f"{args.tag}{manifest['unit']}{u}_k{k}"
            path = out / f"{name}.npz"
            np.savez(path, **pack_shots(chosen, pool, index, targets))
            cfg_path = config_dir / f"{name}.yaml"
            cfg_path.write_text(yaml.safe_dump(run_config(template, name, str(path), args.run_dir, args.wandb_group),
                                               sort_keys=False), encoding="utf-8")
            configs.append(cfg_path)
    (out / "folds.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (config_dir / "configs.txt").write_text("\n".join(str(c) for c in configs) + "\n", encoding="utf-8")
    print(f"{len(keys)} {args.set} molecules, {manifest['mode']}: {[len(u['held_out']) for u in units]} held out; "
          f"{len(configs)} configs")


if __name__ == "__main__":
    main()
