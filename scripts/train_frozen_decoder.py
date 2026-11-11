#!/usr/bin/env python3
"""Train one bet A spectral decoder on frozen per-atom UMA features (TASK-277 R0/R1, TASK-310).

Features come from scripts/extract_uma_features.py (<features>.features.npy + .meta.npz). The decoder
(src/gnn_excited/models/spectral_decoders.py: mlp control, r0 tridiagonal, r1 configuration space)
is trained on the train split with the UMA trainer's loss (energy + log1p f MSE, 0.1 gap, 0.01
ordering), selected on val energy MAE, and scored on test. The whole feature matrix is held on the GPU
in float16; batches gather their molecules' atoms by index.

Writes <out>.summary.json (val/test metrics, including sorted S-agg, per-state MAE, the adjacent
inversion rate and errors on near-degenerate adjacent pairs) and <out>.{val,test}_predictions.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn_excited.losses import qm9gwbse_loss  # noqa: E402
from gnn_excited.models.spectral_decoders import SpectralDecoder  # noqa: E402

ENERGY = [f"S{i}_eV" for i in range(1, 6)]
OSC = [f"log1p_S{i}_f" for i in range(1, 6)]
COLUMNS = [c for pair in zip(ENERGY, OSC) for c in pair]  # loss layout, as in the UMA configs
NEAR_DEGENERATE_EV = 0.1


def gather(starts: torch.Tensor, counts: torch.Tensor):
    """Atom row indices and the molecule index of each atom, for molecules with these atom ranges."""
    node_batch = torch.repeat_interleave(torch.arange(counts.numel(), device=counts.device), counts)
    first = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    atoms = torch.repeat_interleave(starts, counts) + torch.arange(int(counts.sum()), device=counts.device) - first
    return atoms, node_batch


def energy_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Sorted S-agg, per-state MAE, inversion rate of the raw predictions, near-degenerate pair errors."""
    srt = np.sort(pred, axis=1)
    err = np.abs(srt - target)
    gaps_t, gaps_p = np.diff(target, axis=1), np.diff(srt, axis=1)
    near = gaps_t < NEAR_DEGENERATE_EV
    state_near = np.zeros_like(target, dtype=bool)
    state_near[:, :-1] |= near
    state_near[:, 1:] |= near
    return {"S_agg_sorted": float(err.mean()), "per_state": err.mean(axis=0).round(5).tolist(),
            "adjacent_inversion_rate": float((np.diff(pred, axis=1) < 0).mean()),
            "gap_mae": float(np.abs(gaps_p - gaps_t).mean()),
            "near_degenerate_pairs": int(near.sum()),
            "near_degenerate_gap_mae": float(np.abs(gaps_p - gaps_t)[near].mean()) if near.any() else None,
            "near_degenerate_state_mae": float(err[state_near].mean()) if state_near.any() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="prefix of <prefix>.features.npy / .meta.npz")
    parser.add_argument("--kind", choices=["mlp", "r0", "r1"], required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", required=True, help="output prefix")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--configs", type=int, default=16)
    parser.add_argument("--offdiag-scale", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=float, default=5)
    parser.add_argument("--gap-weight", type=float, default=0.1)
    parser.add_argument("--ordering-weight", type=float, default=0.01)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = np.load(f"{args.features}.meta.npz", allow_pickle=False)
    features = torch.from_numpy(np.load(f"{args.features}.features.npy", mmap_mode="r")[:]).to(device)
    offsets = torch.as_tensor(meta["atom_offsets"], dtype=torch.long, device=device)
    starts, counts = offsets[:-1], offsets[1:] - offsets[:-1]
    available = [str(c) for c in meta["target_columns"]]
    targets = torch.as_tensor(meta["targets"][:, [available.index(c) for c in COLUMNS]], dtype=torch.float32, device=device)
    split = meta["split"].astype(str)
    keys = [str(k) for k in meta["molecule_key"]]
    idx = {s: torch.as_tensor(np.flatnonzero(split == s), device=device) for s in ("train", "val", "test")}
    train_t = targets[idx["train"]]
    e_cols, o_cols = list(range(0, 10, 2)), list(range(1, 10, 2))
    model = SpectralDecoder(features.shape[1], train_t[:, e_cols].mean(0).tolist(), train_t[:, e_cols].std(0).tolist(),
                            train_t[:, o_cols].mean(0).tolist(), train_t[:, o_cols].std(0).clamp_min(1e-6).tolist(),
                            kind=args.kind, hidden=args.hidden, configs=args.configs,
                            offdiag_scale=args.offdiag_scale).to(device)

    def predict(rows: torch.Tensor):
        atoms, node_batch = gather(starts[rows], counts[rows])
        energies, osc = model(features[atoms].float(), node_batch, rows.numel())
        out = torch.empty(rows.numel(), 10, device=device)
        out[:, e_cols], out[:, o_cols] = energies, osc
        return out

    @torch.no_grad()
    def evaluate(name: str):
        model.eval()
        preds = torch.cat([predict(chunk) for chunk in torch.split(idx[name], 2048)])
        model.train()
        return preds

    steps_per_epoch = idx["train"].numel() // args.batch_size
    total, warmup = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) * (
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warmup) / max(1, total - warmup))))))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best, best_state, history, started = math.inf, None, [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = idx["train"][torch.randperm(idx["train"].numel(), generator=generator).to(device)]
        running = 0.0
        for step in range(steps_per_epoch):
            rows = order[step * args.batch_size:(step + 1) * args.batch_size]
            loss = qm9gwbse_loss(predict(rows), targets[rows], COLUMNS, gap_weight=args.gap_weight,
                                 ordering_weight=args.ordering_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss)
        val = evaluate("val")
        val_mae = float((val[:, e_cols] - targets[idx["val"]][:, e_cols]).abs().mean())
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch, "val_energy_eV_mae": val_mae})
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch} loss {running / steps_per_epoch:.5f} val {val_mae:.5f} "
                  f"best {best:.5f}@{best_epoch} {time.perf_counter() - started:.0f}s", flush=True)

    if best_state is None:
        raise FloatingPointError("validation MAE was never finite")
    model.load_state_dict(best_state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary = {"features": args.features, "kind": args.kind, "seed": args.seed, "args": vars(args),
               "best_epoch": best_epoch, "seconds": time.perf_counter() - started,
               "parameters": sum(p.numel() for p in model.parameters()), "history": history}
    for name in ("val", "test"):
        pred = evaluate(name).cpu().numpy()
        target = targets[idx[name]].cpu().numpy()
        summary[name] = energy_metrics(pred[:, e_cols], target[:, e_cols])
        summary[name]["energy_eV_mae_unsorted"] = float(np.abs(pred[:, e_cols] - target[:, e_cols]).mean())
        summary[name]["log1p_osc_mae"] = float(np.abs(pred[:, o_cols] - target[:, o_cols]).mean())
        with open(f"{args.out}.{name}_predictions.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["molecule_key", *[f"{p}_{c}" for c in COLUMNS for p in ("target", "prediction")]])
            for row, m in enumerate(idx[name].cpu().numpy()):
                writer.writerow([keys[m], *[f"{v:.6f}" for j in range(10) for v in (target[row, j], pred[row, j])]])
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("TEST " + json.dumps(summary["test"]), flush=True)


if __name__ == "__main__":
    main()
