#!/usr/bin/env python3
"""Predict S1 with the Baum et al. ViSNetGWBSE excitation checkpoints (TASK-271).

Their code (github.com/daoiradrio/ViSNetGWBSE @ daca34e, no licence; kept under data/external and
never committed) imports itself as `ViSNetGW`, so --package-root must hold a ViSNetGW link to the
repo. Hyperparameters come from their config.yaml, with n_feat, r_cut, n_bas, n_mp and l_max read
from the checkpoint name (README pattern, e.g. gapbse5M128502532). Checkpoints are plain state
dicts and load with weights_only=True. For each pool (.npz with numbers, positions, atom_offsets,
molecule_key) and checkpoint, writes <out-dir>/<pool stem>.<checkpoint stem>.csv with
molecule_key and prediction_S1_eV.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_checkpoint_name(stem: str) -> dict:
    """prop, n_pre (millions), and the architecture fields encoded in a checkpoint file name."""
    match = re.fullmatch(r"([a-z]+?)(\d+)M(\d{7,})", stem)
    if not match:
        raise ValueError(f"unrecognised checkpoint name {stem!r}")
    prop, npre, tail = match.groups()
    n_pre = float("0." + npre[1:]) if npre.startswith("0") and len(npre) > 1 else float(npre)
    return {"prop": prop, "n_pre_millions": n_pre, "hidden_channels": int(tail[:-6]), "cutoff": int(tail[-6:-4]) / 10,
            "num_rbf": int(tail[-4:-2]), "num_layers": int(tail[-2]), "lmax": int(tail[-1])}


def load_model(repo: Path, checkpoint: Path):
    from omegaconf import OmegaConf
    from ViSNetGW.model.visnet import create_model

    cfg = OmegaConf.load(repo / "config.yaml")
    for field, value in parse_checkpoint_name(checkpoint.stem).items():
        if field in cfg.model:
            cfg.model[field] = value
    model = create_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def predict(model, pool: dict, device, batch_size: int) -> np.ndarray:
    numbers, positions, offsets = pool["numbers"], pool["positions"], pool["atom_offsets"]
    out = []
    for start in range(0, offsets.size - 1, batch_size):
        stop = min(start + batch_size, offsets.size - 1)
        a, b = int(offsets[start]), int(offsets[stop])
        counts = torch.as_tensor(np.diff(offsets[start:stop + 1]))
        data = {"z": torch.as_tensor(numbers[a:b].astype(np.int64), device=device),
                "pos": torch.as_tensor(positions[a:b], dtype=torch.float32, device=device),
                "batch": torch.repeat_interleave(torch.arange(stop - start), counts).to(device)}
        out.append(model(data)[0].view(-1).float().cpu())
    return torch.cat(out).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="data/external/ViSNetGWBSE")
    parser.add_argument("--package-root", default="data/external/visnetgw_pkg")
    parser.add_argument("--checkpoint-dir", default="data/external/ViSNetGWBSE/checkpoints/excitation_energies")
    parser.add_argument("--pool", action="append", required=True)
    parser.add_argument("--out-dir", default="runs/head_to_head/visnetgwbse")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.package_root).resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in sorted(Path(args.checkpoint_dir).glob("*.ckpt")):
        model = load_model(Path(args.repo), checkpoint).to(device)
        for pool_path in args.pool:
            started = time.perf_counter()
            pool = dict(np.load(pool_path, allow_pickle=False))
            values = predict(model, pool, device, args.batch_size)
            path = out_dir / f"{Path(pool_path).stem}.{checkpoint.stem}.csv"
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["molecule_key", "prediction_S1_eV"])
                writer.writerows((str(k), f"{v:.6f}") for k, v in zip(pool["molecule_key"], values))
            print(f"{checkpoint.name} on {pool_path}: {values.size} molecules in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
