#!/usr/bin/env python3
"""Predict with saved ViSNet checkpoints over a packed .npz molecule pool (TASK-257, DOC-248 L1).

The pool holds numbers, positions, atom_offsets and molecule_key (scripts/build_distill_pool.py).
For each --checkpoint/--out pair it writes molecule_key plus prediction_<column> for the
checkpoint's target columns (energies in eV, oscillators as log1p f), in pool order.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn_excited.models.visnet import build_visnet  # noqa: E402


class PoolDataset(torch.utils.data.Dataset):
    def __init__(self, pool):
        self.numbers, self.positions, self.offsets = pool["numbers"], pool["positions"], pool["atom_offsets"]

    def __len__(self):
        return self.offsets.size - 1

    def __getitem__(self, i):
        from torch_geometric.data import Data

        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return Data(z=torch.as_tensor(self.numbers[a:b].astype(np.int64)),
                    pos=torch.as_tensor(self.positions[a:b], dtype=torch.float32))


@torch.no_grad()
def predict(checkpoint: str, loader, device) -> tuple[list[str], np.ndarray]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kwargs = dict(state["model_kwargs"])
    columns = list(kwargs["target_columns"])
    model = build_visnet(**kwargs)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    outputs = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.z, batch.pos, batch.batch)
        out = out[0] if isinstance(out, tuple) else out
        outputs.append(out.view(batch.num_graphs, -1).float().cpu())
    return columns, torch.cat(outputs).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--out", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.out):
        raise SystemExit("give one --out per --checkpoint")
    from torch_geometric.loader import DataLoader

    pool = dict(np.load(args.pool, allow_pickle=False))
    keys = [str(k) for k in pool["molecule_key"]]
    loader = DataLoader(PoolDataset(pool), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for checkpoint, out in zip(args.checkpoint, args.out):
        started = time.perf_counter()
        columns, values = predict(checkpoint, loader, device)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["molecule_key", *[f"prediction_{c}" for c in columns]])
            for key, row in zip(keys, values):
                writer.writerow([key, *(f"{v:.6f}" for v in row)])
        print(f"{checkpoint} -> {out}: {len(keys)} molecules in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
