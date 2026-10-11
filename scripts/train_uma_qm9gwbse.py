#!/usr/bin/env python3
"""Fine-tune the UMA-S foundation backbone on QM9GWBSE excited states (TASK-211).

Runs in envs/uma-fairchem (fairchem-core 2.22; no torch_geometric/h5py), on the
packed .npz from scripts/export_qm9gwbse_packed.py. The eSCNMDMoeBackbone
(uma-s-1p2p1, 291M params, ~6.4M active per molecule) is loaded with
otf_graph so it builds its own radius graph; its l=0 node embeddings feed
fresh energy/oscillator heads, mean-pooled per molecule like ViSNetOnePass.
Loss and metrics are gnn_excited.losses.qm9gwbse_loss and
gnn_excited.metrics.batch_mae, so test energy_eV_mae compares directly with
the ViSNet incumbent (0.059560). Env, charge/spin, loading and MoLE-speed
lessons: Stele KNOW-219..222.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn_excited.losses import qm9gwbse_loss, target_layout  # noqa: E402
from gnn_excited.metrics import batch_mae  # noqa: E402

BACKBONE_OVERRIDES = {
    "backbone": {
        "otf_graph": True,
        "always_use_pbc": False,
        "radius_pbc_version": 2,
        "activation_checkpointing": False,
        "regress_forces": False,
        "regress_stress": False,
        "execution_mode": "general",
    }
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def molecule_data(numbers: np.ndarray, positions: np.ndarray):
    """Closed-shell neutral molecule for the omol task (spin is multiplicity, so 1)."""
    from fairchem.core.datasets.atomic_data import AtomicData

    n = int(numbers.size)
    return AtomicData(
        pos=torch.from_numpy(np.asarray(positions, dtype=np.float32)),
        atomic_numbers=torch.from_numpy(numbers.astype(np.int64)),
        cell=torch.zeros(1, 3, 3),
        pbc=torch.zeros(1, 3, dtype=torch.bool),
        natoms=torch.tensor([n]),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        cell_offsets=torch.empty(0, 3),
        nedges=torch.tensor([0]),
        charge=torch.tensor([0]),
        spin=torch.tensor([1]),
        fixed=torch.zeros(n, dtype=torch.long),
        tags=torch.zeros(n, dtype=torch.long),
        dataset="omol",
    )


class PackedMolecules(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray, target_index: np.ndarray):
        self.numbers = arrays["numbers"]
        self.positions = arrays["positions"]
        self.offsets = arrays["atom_offsets"]
        self.targets = arrays["targets"][:, target_index].astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int):
        m = int(self.indices[i])
        start, end = int(self.offsets[m]), int(self.offsets[m + 1])
        return molecule_data(self.numbers[start:end], self.positions[start:end]), torch.from_numpy(self.targets[m]), m


def collate(items):
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    datas, targets, rows = zip(*items)
    return atomicdata_list_to_batch(list(datas)), torch.stack(targets), torch.as_tensor(rows)


def _head(embed_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )
    # Zero final layer: the model starts at the train-set target mean.
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class UMAExcitedState(nn.Module):
    """UMA backbone l=0 embeddings -> per-atom heads -> per-molecule mean, in target units."""

    def __init__(self, backbone, target_columns, target_mean, target_std, embed_dim: int = 128, head_hidden: int = 256):
        super().__init__()
        self.backbone = backbone
        self.target_columns = tuple(target_columns)
        self.energy_indices, self.oscillator_indices = target_layout(self.target_columns)
        self.energy_head = _head(embed_dim, head_hidden, len(self.energy_indices))
        self.oscillator_head = _head(embed_dim, head_hidden, len(self.oscillator_indices)) if self.oscillator_indices else None
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))

    def heads(self) -> list[nn.Module]:
        return [h for h in (self.energy_head, self.oscillator_head) if h is not None]

    def forward(self, batch):
        num_molecules = int(batch.natoms.numel())
        embedding = self.backbone(batch)
        scalars = embedding["node_embedding"][:, 0, :].float()
        node_batch = embedding["batch"]
        counts = torch.bincount(node_batch, minlength=num_molecules).clamp_min(1).unsqueeze(1).to(scalars.dtype)

        def pool(per_atom):
            summed = per_atom.new_zeros(num_molecules, per_atom.size(1)).index_add_(0, node_batch, per_atom)
            return summed / counts

        out = scalars.new_empty(num_molecules, len(self.target_columns))
        out[:, list(self.energy_indices)] = pool(self.energy_head(scalars))
        if self.oscillator_head is not None:
            out[:, list(self.oscillator_indices)] = pool(self.oscillator_head(scalars))
        return out * self.target_std + self.target_mean


def patch_mole_split_forward() -> None:
    """O(B) backward for MOLE (stock per-system slicing is O(B^2); KNOW-222)."""
    from fairchem.core.models.uma.nn import mole as mole_mod

    stock = mole_mod.MOLE.forward
    if getattr(stock, "_gnn_excited_split", False):
        return

    def split_forward(self, x):
        g = self.global_mole_tensors
        sizes = g.mole_sizes.tolist()
        if g.ac_start_idx != 0 or x.shape[0] != sum(sizes):
            return stock(self, x)
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum("eoi, be->boi", self.weights, g.expert_mixing_coefficients)
        parts = [F.linear(xi, wi, self.bias) for xi, wi in zip(torch.split(x, sizes, 0), weights.unbind(0))]
        return torch.cat(parts, 0)

    split_forward._gnn_excited_split = True
    mole_mod.MOLE.forward = split_forward


def load_backbone(checkpoint: str):
    from fairchem.core.units.mlip_unit.utils import load_inference_model

    model, _ = load_inference_model(checkpoint, overrides=BACKBONE_OVERRIDES, use_ema=True)
    backbone = model.module.backbone
    backbone.requires_grad_(True)
    return backbone


def lr_lambda_factory(total_steps: int, warmup_steps: int, start_factor: float, min_factor: float):
    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return start_factor + (1.0 - start_factor) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def loss_kwargs(loss_cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "energy_weight": float(loss_cfg.get("energy_weight", 1.0)),
        "oscillator_weight": float(loss_cfg.get("oscillator_weight", 1.0)),
        "gap_weight": float(loss_cfg.get("gap_weight", 0.0)),
        "ordering_weight": float(loss_cfg.get("ordering_weight", 0.0)),
        "ordering_margin": float(loss_cfg.get("ordering_margin", 0.0)),
    }


@torch.no_grad()
def evaluate(model, loader, target_columns, loss_cfg, device, autocast_dtype=None, keys=None, predictions_path=None):
    model.eval()
    predictions, targets, rows = [], [], []
    total_loss, total_n = 0.0, 0
    for batch, target, row in loader:
        batch, target = batch.to(device), target.to(device)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            pred = model(batch)
        pred = pred.float()
        loss = qm9gwbse_loss(pred, target, target_columns, **loss_kwargs(loss_cfg))
        total_loss += float(loss) * target.size(0)
        total_n += target.size(0)
        predictions.append(pred.cpu())
        targets.append(target.cpu())
        rows.append(row)
    pred_all, target_all = torch.cat(predictions), torch.cat(targets)
    metrics = batch_mae(pred_all, target_all, tuple(target_columns))
    metrics["loss"] = total_loss / max(total_n, 1)
    if metrics.get("ordering_comparison_count"):
        metrics["ordering_violation_rate"] = metrics["ordering_violation_count"] / metrics["ordering_comparison_count"]
    if predictions_path is not None and keys is not None:
        row_all = torch.cat(rows).numpy()
        with open(predictions_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["molecule_key", *[f"{p}_{c}" for c in target_columns for p in ("target", "prediction")]])
            for k, m in enumerate(row_all):
                values = [v for j in range(len(target_columns)) for v in (float(target_all[k, j]), float(pred_all[k, j]))]
                writer.writerow([keys[m], *values])
    return metrics


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for record in history:
        fields.extend(key for key in record if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def run(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    model_cfg, data_cfg, train_cfg = config["model"], config["dataset"], config["training"]
    loss_cfg = config.get("loss") or {}
    target_columns = tuple(config["targets"]["columns"])
    seed = int(train_cfg.get("seed", 17))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision(str(train_cfg.get("matmul_precision", "high")))
    device = torch.device(train_cfg.get("device", "cuda"))

    arrays = dict(np.load(data_cfg["packed_path"], allow_pickle=False))
    available = [str(c) for c in arrays["target_columns"]]
    target_index = np.asarray([available.index(c) for c in target_columns])
    split = arrays["split"].astype(str)
    split_idx = {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}
    for name, limit in (data_cfg.get("max_molecules") or {}).items():
        split_idx[name] = split_idx[name][: int(limit)]
    train_targets = arrays["targets"][split_idx["train"]][:, target_index]
    target_mean = train_targets.mean(axis=0)
    target_std = np.maximum(train_targets.std(axis=0), 1e-6)

    loader_kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 8)),
        "collate_fn": collate,
        "pin_memory": False,
        "persistent_workers": int(train_cfg.get("num_workers", 8)) > 0,
    }
    datasets = {name: PackedMolecules(arrays, idx, target_index) for name, idx in split_idx.items()}
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], shuffle=True, drop_last=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)

    if bool(train_cfg.get("mole_split_patch", True)):
        patch_mole_split_forward()
    backbone = load_backbone(model_cfg["checkpoint_path"])
    embed_dim = int(getattr(backbone, "sphere_channels", model_cfg.get("embed_dim", 128)))
    model = UMAExcitedState(
        backbone, target_columns, target_mean, target_std,
        embed_dim=embed_dim, head_hidden=int(model_cfg.get("head_hidden", 256)),
    ).to(device)

    head_params = [p for head in model.heads() for p in head.parameters()]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()), "lr": float(train_cfg["backbone_lr"]),
             "weight_decay": float(train_cfg.get("backbone_weight_decay", 0.0))},
            {"params": head_params, "lr": float(train_cfg["head_lr"]),
             "weight_decay": float(train_cfg.get("head_weight_decay", 1e-3))},
        ]
    )
    epochs = int(train_cfg["epochs"])
    steps_per_epoch = len(train_loader)
    sched_cfg = train_cfg.get("scheduler") or {}
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda_factory(
            epochs * steps_per_epoch,
            int(float(sched_cfg.get("warmup_epochs", 1)) * steps_per_epoch),
            float(sched_cfg.get("start_factor", 0.01)),
            float(sched_cfg.get("min_factor", 0.01)),
        ),
    )
    autocast_dtype = torch.bfloat16 if bool(train_cfg.get("bf16", False)) else None
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    skip_nonfinite = bool(train_cfg.get("skip_nonfinite_batches", True))
    max_hours = float(train_cfg.get("max_hours", 1e9))

    checkpoint_path = Path(train_cfg["checkpoint_path"])
    metrics_path = Path(train_cfg["metrics_csv_path"])
    summary_path = Path(train_cfg["summary_json_path"])
    for path in (checkpoint_path, metrics_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    wandb_cfg = config.get("wandb") or {}
    if wandb_cfg.get("enabled"):
        try:
            import wandb

            wandb_run = wandb.init(
                entity=wandb_cfg.get("entity"), project=wandb_cfg.get("project"), name=wandb_cfg.get("name"),
                group=wandb_cfg.get("group"), tags=wandb_cfg.get("tags"), config=config,
            )
        except Exception as exc:  # keep training if W&B is unreachable
            print(f"W&B disabled: {type(exc).__name__}: {exc}", flush=True)

    summary: dict[str, Any] = {
        "status": "running",
        "started_at": _utc_now(),
        "config_path": config_path,
        "config": config,
        "split_sizes": {name: int(idx.size) for name, idx in split_idx.items()},
        "target_columns": list(target_columns),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "parameters": {
            "backbone": sum(p.numel() for p in model.backbone.parameters()),
            "heads": sum(p.numel() for p in head_params),
        },
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "wandb_run_id": getattr(wandb_run, "id", None),
    }
    print(json.dumps({k: summary[k] for k in ("split_sizes", "parameters", "gpu")}), flush=True)

    history: list[dict[str, Any]] = []
    best_val, best_epoch, skipped = math.inf, None, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running, seen = 0.0, 0
        for step, (batch, target, _) in enumerate(train_loader, start=1):
            batch, target = batch.to(device), target.to(device)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                pred = model(batch)
            loss = qm9gwbse_loss(pred.float(), target, target_columns, **loss_kwargs(loss_cfg))
            if not torch.isfinite(loss):
                if not skip_nonfinite:
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch} step {step}")
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += float(loss) * target.size(0)
            seen += target.size(0)
            if step % int(train_cfg.get("log_every_steps", 200)) == 0:
                rate = seen / (time.perf_counter() - epoch_start)
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {running / seen:.5f} {rate:.0f} mol/s", flush=True)
        train_seconds = time.perf_counter() - epoch_start
        val_metrics = evaluate(model, val_loader, target_columns, loss_cfg, device, autocast_dtype)
        record = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val_metrics["loss"],
            "val_energy_eV_mae": val_metrics.get("energy_eV_mae"),
            "val_log1p_oscillator_strength_mae": val_metrics.get("log1p_oscillator_strength_mae"),
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "train_seconds": train_seconds,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "skipped_nonfinite_batches": skipped,
        }
        history.append(record)
        if val_metrics["loss"] < best_val:
            best_val, best_epoch = val_metrics["loss"], epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": config,
                    "target_columns": list(target_columns),
                },
                checkpoint_path,
            )
        print(json.dumps(record), flush=True)
        if wandb_run is not None:
            wandb_run.log({**record, **{f"val_{k}": v for k, v in val_metrics.items()}}, step=epoch)
        write_history(metrics_path, history)
        summary.update(updated_at=_utc_now(), latest_epoch=epoch, best_epoch=best_epoch, best_val_loss=best_val,
                       latest_val_metrics=val_metrics)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        elapsed_hours = (time.perf_counter() - started) / 3600
        if epoch < epochs and elapsed_hours + 1.2 * record["epoch_seconds"] / 3600 > max_hours:
            summary["stop_reason"] = f"max_hours {max_hours} reached after epoch {epoch}"
            print(summary["stop_reason"], flush=True)
            break

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    keys = [str(k) for k in arrays["molecule_key"]]
    predictions_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".test_predictions.csv"))
    test_metrics = evaluate(model, test_loader, target_columns, loss_cfg, device, autocast_dtype, keys, predictions_path)
    summary.update(status="completed", completed_at=_utc_now(), test_metrics=test_metrics,
                   predictions_csv_path=str(predictions_path))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("TEST " + json.dumps(test_metrics), flush=True)
    if wandb_run is not None:
        wandb_run.summary.update({f"test_{k}": v for k, v in test_metrics.items()} | {"best_epoch": best_epoch})
        wandb_run.finish()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    run(config, args.config)


if __name__ == "__main__":
    main()
