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
import re
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


def molecule_data(numbers: np.ndarray, positions: np.ndarray, charge: int = 0, spin: int = 1):
    """One molecule for the omol task. spin is the multiplicity (1 for closed shell)."""
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
        charge=torch.tensor([int(charge)]),
        spin=torch.tensor([int(spin)]),
        fixed=torch.zeros(n, dtype=torch.long),
        tags=torch.zeros(n, dtype=torch.long),
        dataset="omol",
    )


def _charge_spin(arrays: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Optional per-molecule `charge` and `spin` (multiplicity) arrays; neutral singlets otherwise."""
    return (np.asarray(arrays.get("charge", np.zeros(n)), dtype=np.int64),
            np.asarray(arrays.get("spin", np.ones(n)), dtype=np.int64))


class PackedMolecules(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray, target_index: np.ndarray):
        self.numbers = arrays["numbers"]
        self.positions = arrays["positions"]
        self.offsets = arrays["atom_offsets"]
        self.targets = arrays["targets"][:, target_index].astype(np.float32)
        self.charge, self.spin = _charge_spin(arrays, self.targets.shape[0])
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int):
        m = int(self.indices[i])
        start, end = int(self.offsets[m]), int(self.offsets[m + 1])
        data = molecule_data(self.numbers[start:end], self.positions[start:end], self.charge[m], self.spin[m])
        return data, torch.from_numpy(self.targets[m]), m


class PoolMolecules(Dataset):
    """Unlabeled packed pool (numbers, positions, atom_offsets; optional charge, spin) for inference."""

    def __init__(self, pool: dict[str, np.ndarray]):
        self.numbers, self.positions, self.offsets = pool["numbers"], pool["positions"], pool["atom_offsets"]
        self.charge, self.spin = _charge_spin(pool, int(self.offsets.size - 1))

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def __getitem__(self, i: int):
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        data = molecule_data(self.numbers[start:end], self.positions[start:end], self.charge[i], self.spin[i])
        return data, torch.zeros(1), i


@torch.no_grad()
def predict_pool(model, pool_path: str, out_csv: str, target_columns, device, batch_size: int, num_workers: int) -> int:
    """Write molecule_key + prediction_<column> for every pool molecule, in pool order."""
    pool = dict(np.load(pool_path, allow_pickle=False))
    keys = [str(k) for k in pool["molecule_key"]]
    loader = DataLoader(PoolMolecules(pool), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate, persistent_workers=False)
    model.eval()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", *[f"prediction_{c}" for c in target_columns]])
        for batch, _, rows in loader:
            pred = model(batch.to(device)).float().cpu().numpy()
            for row, values in zip(rows.tolist(), pred):
                writer.writerow([keys[row], *(f"{v:.6f}" for v in values)])
                written += 1
    return written


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


def spherical_invariants(x: torch.Tensor) -> torch.Tensor:
    """[N, (lmax+1)^2, C] spherical features -> [N, (lmax+1)*C]: the l=0 channel plus per-l channel norms."""
    lmax = int(round(x.shape[1] ** 0.5)) - 1
    if (lmax + 1) ** 2 != x.shape[1]:
        raise ValueError(f"expected (lmax+1)^2 spherical components, got {x.shape[1]}")
    parts = [x[:, 0, :]] + [x[:, l * l:(l + 1) ** 2, :].norm(dim=1) for l in range(1, lmax + 1)]
    return torch.cat(parts, dim=-1)


def _capture_block_output(module, inputs, output):
    # Stored on the block itself (not a closure) so deep copies (EMA) capture their own outputs.
    module._gnn_captured = output[0] if isinstance(output, tuple) else output


class _PooledHead(nn.Module):
    """Per-atom trunk, then mean+std (first two cumulants) pooling per molecule, then a zero-init output layer."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, per_atom, node_batch, num_molecules):
        h = self.trunk(per_atom)
        counts = torch.bincount(node_batch, minlength=num_molecules).clamp_min(1).unsqueeze(1).to(h.dtype)
        mean = h.new_zeros(num_molecules, h.size(1)).index_add_(0, node_batch, h) / counts
        second = h.new_zeros(num_molecules, h.size(1)).index_add_(0, node_batch, h * h) / counts
        std = (second - mean * mean).clamp_min(0.0).add(1e-6).sqrt()
        return self.out(torch.cat([mean, std], dim=-1))


class UMAExcitedState(nn.Module):
    """UMA backbone embeddings -> heads -> per-molecule outputs, in target units.

    readout="l0_mean" (TASK-211): l=0 node features, per-atom heads, per-molecule mean.
    readout="rich" (TASK-250, DOC-248 L3): l=0 plus per-l norms of the final embedding and of the
    last `readout_layers` block outputs, mean+std pooled.

    aux_columns (TASK-287, rich readout only) adds a second-fidelity energy head, called with
    forward(batch, fidelity="aux"). aux_mode="heads": the fidelities share only the backbone.
    aux_mode="delta": the GW energies are the aux prediction plus a learned correction, so GW
    predictions for chemistry seen only in the aux data inherit the aux head's trend.
    """

    def __init__(self, backbone, target_columns, target_mean, target_std, embed_dim: int = 128, head_hidden: int = 256,
                 readout: str = "l0_mean", readout_layers: int = 0, lmax: int = 2,
                 aux_columns=None, aux_mean=None, aux_std=None, aux_mode: str = "heads"):
        super().__init__()
        self.backbone = backbone
        self.readout = readout
        self.readout_layers = int(readout_layers)
        self.target_columns = tuple(target_columns)
        self.energy_indices, self.oscillator_indices = target_layout(self.target_columns)
        if readout == "l0_mean":
            if self.readout_layers:
                raise ValueError("readout_layers requires readout='rich'")
            self.energy_head = _head(embed_dim, head_hidden, len(self.energy_indices))
            self.oscillator_head = _head(embed_dim, head_hidden, len(self.oscillator_indices)) if self.oscillator_indices else None
        elif readout == "rich":
            in_dim = embed_dim * (lmax + 1) * (1 + self.readout_layers)
            self.energy_head = _PooledHead(in_dim, head_hidden, len(self.energy_indices))
            self.oscillator_head = _PooledHead(in_dim, head_hidden, len(self.oscillator_indices)) if self.oscillator_indices else None
            if self.readout_layers:
                blocks = getattr(backbone, "blocks", None)
                if blocks is None or len(blocks) < self.readout_layers + 1:
                    raise ValueError("readout_layers needs backbone.blocks with enough blocks")
                for k in range(1, self.readout_layers + 1):
                    blocks[-1 - k].register_forward_hook(_capture_block_output)
        else:
            raise ValueError(f"unknown readout {readout!r}")
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))
        self.aux_mode = aux_mode if aux_columns else None
        self.aux_head = None
        if self.aux_mode is not None:
            if readout != "rich" or aux_mode not in ("heads", "delta"):
                raise ValueError("aux targets need readout='rich' and aux_mode 'heads' or 'delta'")
            if aux_mode == "delta" and [self.target_columns[i] for i in self.energy_indices] != list(aux_columns):
                raise ValueError("aux_mode='delta' needs aux columns equal to the GW energy columns, in state order")
            self.aux_head = _PooledHead(in_dim, head_hidden, len(aux_columns))
            self.register_buffer("aux_mean", torch.as_tensor(aux_mean, dtype=torch.float32))
            self.register_buffer("aux_std", torch.as_tensor(aux_std, dtype=torch.float32))

    def heads(self) -> list[nn.Module]:
        return [h for h in (self.energy_head, self.oscillator_head, self.aux_head) if h is not None]

    def rich_features(self, embedding) -> torch.Tensor:
        """Per-atom rich-readout input: invariants of the final embedding and the captured block outputs."""
        xs = [embedding["node_embedding"]]
        xs += [self.backbone.blocks[-1 - k]._gnn_captured for k in range(1, self.readout_layers + 1)]
        return torch.cat([spherical_invariants(x.float()) for x in xs], dim=-1)

    def forward(self, batch, fidelity: str = "gw"):
        num_molecules = int(batch.natoms.numel())
        embedding = self.backbone(batch)
        node_batch = embedding["batch"]
        if self.readout == "rich":
            per_atom = self.rich_features(embedding)
            if fidelity == "aux":
                return self.aux_head(per_atom, node_batch, num_molecules) * self.aux_std + self.aux_mean
            energy = self.energy_head(per_atom, node_batch, num_molecules)
            out = per_atom.new_empty(num_molecules, len(self.target_columns))
            out[:, list(self.energy_indices)] = energy
            if self.oscillator_head is not None:
                out[:, list(self.oscillator_indices)] = self.oscillator_head(per_atom, node_batch, num_molecules)
            out = out * self.target_std + self.target_mean
            if self.aux_mode == "delta":
                e = list(self.energy_indices)
                aux = self.aux_head(per_atom, node_batch, num_molecules) * self.aux_std + self.aux_mean
                out[:, e] = aux + energy * self.target_std[e] + (self.target_mean[e] - self.aux_mean)
            return out
        scalars = embedding["node_embedding"][:, 0, :].float()
        counts = torch.bincount(node_batch, minlength=num_molecules).clamp_min(1).unsqueeze(1).to(scalars.dtype)

        def pool(per_atom):
            summed = per_atom.new_zeros(num_molecules, per_atom.size(1)).index_add_(0, node_batch, per_atom)
            return summed / counts

        out = scalars.new_empty(num_molecules, len(self.target_columns))
        out[:, list(self.energy_indices)] = pool(self.energy_head(scalars))
        if self.oscillator_head is not None:
            out[:, list(self.oscillator_indices)] = pool(self.oscillator_head(scalars))
        return out * self.target_std + self.target_mean


def backbone_param_groups(backbone: nn.Module, base_lr: float, weight_decay: float, layer_decay: float | None) -> list[dict]:
    """One AdamW group, or layer-wise LR decay: top block gets base_lr, each block below x layer_decay.

    Parameters outside backbone.blocks.<i> count as embeddings (depth 0) unless their name starts
    with norm/output, which count as above the last block.
    """
    named = [(n, p) for n, p in backbone.named_parameters() if p.requires_grad]
    if not layer_decay:
        return [{"params": [p for _, p in named], "lr": base_lr, "weight_decay": weight_decay, "name": "backbone"}]
    pattern = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    block_ids = [int(m.group(1)) for n, _ in named if (m := pattern.search(n))]
    if not block_ids:
        raise ValueError("layer_lr_decay needs backbone parameters under blocks.<i>")
    top = max(block_ids) + 2
    by_depth: dict[int, list] = {}
    for name, param in named:
        match = pattern.search(name)
        if match:
            depth = int(match.group(1)) + 1
        else:
            depth = top if name.split(".")[0].startswith(("norm", "output")) else 0
        by_depth.setdefault(depth, []).append(param)
    return [{"params": params, "lr": base_lr * layer_decay ** (top - depth), "weight_decay": weight_decay,
             "name": f"backbone_depth{depth}"} for depth, params in sorted(by_depth.items())]


def average_state_dicts(states: list[dict], weights: list[float] | None = None) -> dict:
    """Weighted average of floating tensors (a model soup); other entries are taken from the first state."""
    weights = [1.0 / len(states)] * len(states) if weights is None else [w / sum(weights) for w in weights]
    out = {}
    for key, ref in states[0].items():
        if torch.is_tensor(ref) and ref.is_floating_point():
            out[key] = sum(w * s[key].float() for w, s in zip(weights, states)).to(ref.dtype)
        else:
            out[key] = ref
    return out


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


@torch.no_grad()
def evaluate_aux(model, loader, device, autocast_dtype=None) -> float:
    """Mean absolute error (eV) of the aux-fidelity head over all aux columns."""
    model.eval()
    total, count = 0.0, 0
    for batch, target, _ in loader:
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            pred = model(batch.to(device), fidelity="aux")
        total += float((pred.float() - target.to(device)).abs().sum())
        count += target.numel()
    return total / max(count, 1)


def cycle(loader):
    """Endless iterator over a DataLoader (a fresh shuffle each pass)."""
    while True:
        yield from loader


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

    # aux (TASK-287): {packed_path, columns, mode, weight, batch_size}. A second-fidelity packed set
    # (train/val split column); each GW step is followed by one aux step on the same optimizer.
    aux_cfg = config.get("aux") or {}
    aux_columns, aux_mean, aux_std, aux_loaders = None, None, None, {}
    if aux_cfg:
        aux_arrays = dict(np.load(aux_cfg["packed_path"], allow_pickle=False))
        aux_available = [str(c) for c in aux_arrays["target_columns"]]
        aux_columns = tuple(aux_cfg.get("columns") or aux_available)
        aux_index = np.asarray([aux_available.index(c) for c in aux_columns])
        aux_split = aux_arrays["split"].astype(str)
        aux_idx = {name: np.flatnonzero(aux_split == name) for name in ("train", "val")}
        aux_train_targets = aux_arrays["targets"][aux_idx["train"]][:, aux_index]
        aux_mean, aux_std = aux_train_targets.mean(axis=0), np.maximum(aux_train_targets.std(axis=0), 1e-6)
        aux_kwargs = {**loader_kwargs, "batch_size": int(aux_cfg.get("batch_size", train_cfg["batch_size"]))}
        aux_loaders = {
            "train": DataLoader(PackedMolecules(aux_arrays, aux_idx["train"], aux_index), shuffle=True, drop_last=True,
                                generator=torch.Generator().manual_seed(seed + 1), **aux_kwargs),
            "val": DataLoader(PackedMolecules(aux_arrays, aux_idx["val"], aux_index), shuffle=False, **aux_kwargs),
        }

    # shots (TASK-310): a few extra GW-labelled molecules (energy columns only, e.g. OE62H) for the GW
    # head: {packed_path, columns, weight, batch_size}. Every row of the file is used; each GW step is
    # followed by one shots step, so a small set is revisited many times per epoch.
    shots_cfg = config.get("shots") or {}
    shots_loader, shots_columns, shots_positions = None, None, None
    if shots_cfg:
        shots_arrays = dict(np.load(shots_cfg["packed_path"], allow_pickle=False))
        shots_available = [str(c) for c in shots_arrays["target_columns"]]
        shots_columns = tuple(shots_cfg.get("columns") or shots_available)
        if any(c not in target_columns or not c.endswith("_eV") for c in shots_columns):
            raise ValueError("shots columns must be energy columns among the GW target columns")
        shots_positions = [target_columns.index(c) for c in shots_columns]
        shots_rows = np.arange(shots_arrays["targets"].shape[0])
        shots_loader = DataLoader(
            PackedMolecules(shots_arrays, shots_rows, np.asarray([shots_available.index(c) for c in shots_columns])),
            shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed + 2),
            **{**loader_kwargs, "batch_size": min(int(shots_cfg.get("batch_size", 8)), shots_rows.size),
               "num_workers": 2, "persistent_workers": True})

    if bool(train_cfg.get("mole_split_patch", True)):
        patch_mole_split_forward()
    backbone = load_backbone(model_cfg["checkpoint_path"])
    embed_dim = int(getattr(backbone, "sphere_channels", model_cfg.get("embed_dim", 128)))
    # A fixed head_init_seed makes runs that differ only in data order share initial weights (soup-able).
    head_init_seed = train_cfg.get("head_init_seed")
    if head_init_seed is not None:
        torch.manual_seed(int(head_init_seed))
    model = UMAExcitedState(
        backbone, target_columns, target_mean, target_std,
        embed_dim=embed_dim, head_hidden=int(model_cfg.get("head_hidden", 256)),
        readout=str(model_cfg.get("readout", "l0_mean")), readout_layers=int(model_cfg.get("readout_layers", 0)),
        lmax=int(model_cfg.get("lmax", 2)),
        aux_columns=aux_columns, aux_mean=aux_mean, aux_std=aux_std, aux_mode=str(aux_cfg.get("mode", "heads")),
    ).to(device)
    if head_init_seed is not None:
        torch.manual_seed(seed)
    # training.init_checkpoint (TASK-310): start from a trained model's full state (heads and buffers
    # included), e.g. to fine-tune. The model must be built with the same targets, readout and aux.
    init_checkpoint = train_cfg.get("init_checkpoint")
    if init_checkpoint:
        model.load_state_dict(torch.load(init_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
        print(f"initialised from {init_checkpoint}", flush=True)

    # inference: {pool_path, checkpoint_paths, out_csvs}: teacher predictions on an unlabeled pool, then exit.
    inference_cfg = config.get("inference") or {}
    if inference_cfg:
        paths, outs = list(inference_cfg["checkpoint_paths"]), list(inference_cfg["out_csvs"])
        if len(paths) != len(outs):
            raise ValueError("inference.checkpoint_paths and inference.out_csvs must pair up")
        results = {}
        for path, out in zip(paths, outs):
            started = time.perf_counter()
            state = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
            model.load_state_dict(state)
            n = predict_pool(model, inference_cfg["pool_path"], out, target_columns, device,
                             int(inference_cfg.get("batch_size", train_cfg["batch_size"])), int(train_cfg.get("num_workers", 8)))
            results[path] = {"out_csv": out, "molecules": n, "seconds": time.perf_counter() - started}
            print(f"INFERENCE {path} -> {out}: {n} molecules in {results[path]['seconds']:.0f}s", flush=True)
        return {"status": "inference_completed", "results": results}

    head_params = [p for head in model.heads() for p in head.parameters()]
    backbone_groups = backbone_param_groups(
        model.backbone, float(train_cfg["backbone_lr"]), float(train_cfg.get("backbone_weight_decay", 0.0)),
        train_cfg.get("layer_lr_decay"),
    )
    print("backbone LR groups: " + json.dumps([(g["name"], len(g["params"]), g["lr"]) for g in backbone_groups]), flush=True)
    optimizer = torch.optim.AdamW(
        backbone_groups
        + [{"params": head_params, "lr": float(train_cfg["head_lr"]),
            "weight_decay": float(train_cfg.get("head_weight_decay", 1e-3)), "name": "heads"}]
    )
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "val_loss"))
    ema_decay = train_cfg.get("ema_decay")
    ema = None
    if ema_decay:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(ema_decay)), use_buffers=True)
    # evaluation.checkpoint_path: score an existing checkpoint on val and test, no training.
    # evaluation.soup_checkpoint_paths (+ optional soup_weights): score the weight average of several.
    eval_cfg = config.get("evaluation") or {}
    eval_checkpoint = eval_cfg.get("checkpoint_path")
    soup_paths = list(eval_cfg.get("soup_checkpoint_paths") or [])
    epochs = 0 if (eval_checkpoint or soup_paths) else int(train_cfg["epochs"])
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
        "evaluation_checkpoint_path": eval_checkpoint,
        "soup_checkpoint_paths": soup_paths or None,
        "checkpoint_metric": checkpoint_metric,
        "ema_decay": ema_decay,
        "head_init_seed": head_init_seed,
        "backbone_lr_groups": [(g["name"], g["lr"]) for g in backbone_groups],
    }
    print(json.dumps({k: summary[k] for k in ("split_sizes", "parameters", "gpu")}), flush=True)

    if aux_cfg:
        summary["aux"] = {"packed_path": aux_cfg["packed_path"], "mode": model.aux_mode, "columns": list(aux_columns),
                          "weight": float(aux_cfg.get("weight", 1.0)),
                          "split_sizes": {name: len(loader.dataset) for name, loader in aux_loaders.items()}}
    aux_weight = float(aux_cfg.get("weight", 1.0))
    aux_iter = cycle(aux_loaders["train"]) if aux_cfg and epochs else None
    if shots_cfg:
        summary["shots"] = {"packed_path": shots_cfg["packed_path"], "columns": list(shots_columns),
                            "weight": float(shots_cfg.get("weight", 1.0)), "molecules": len(shots_loader.dataset)}
    shots_weight = float(shots_cfg.get("weight", 1.0))
    shots_iter = cycle(shots_loader) if shots_loader is not None and epochs else None

    history: list[dict[str, Any]] = []
    best_val, best_epoch, skipped = math.inf, None, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running, seen, aux_running, aux_seen, shots_running, shots_seen = 0.0, 0, 0.0, 0, 0.0, 0
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
            if aux_iter is not None:
                # Separate backward pass, so peak memory stays at one batch.
                aux_batch, aux_target, _ = next(aux_iter)
                aux_batch, aux_target = aux_batch.to(device), aux_target.to(device)
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    aux_pred = model(aux_batch, fidelity="aux")
                aux_loss = qm9gwbse_loss(aux_pred.float(), aux_target, aux_columns, **loss_kwargs(loss_cfg))
                if torch.isfinite(aux_loss):
                    (aux_weight * aux_loss).backward()
                    aux_running += float(aux_loss) * aux_target.size(0)
                    aux_seen += aux_target.size(0)
            if shots_iter is not None:
                shot_batch, shot_target, _ = next(shots_iter)
                shot_batch, shot_target = shot_batch.to(device), shot_target.to(device)
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    shot_pred = model(shot_batch)[:, shots_positions]
                shot_loss = qm9gwbse_loss(shot_pred.float(), shot_target, shots_columns, **loss_kwargs(loss_cfg))
                if torch.isfinite(shot_loss):
                    (shots_weight * shot_loss).backward()
                    shots_running += float(shot_loss) * shot_target.size(0)
                    shots_seen += shot_target.size(0)
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            running += float(loss) * target.size(0)
            seen += target.size(0)
            if step % int(train_cfg.get("log_every_steps", 200)) == 0:
                rate = seen / (time.perf_counter() - epoch_start)
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {running / seen:.5f} {rate:.0f} mol/s", flush=True)
        train_seconds = time.perf_counter() - epoch_start
        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate(eval_model, val_loader, target_columns, loss_cfg, device, autocast_dtype)
        record = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val_metrics["loss"],
            "val_energy_eV_mae": val_metrics.get("energy_eV_mae"),
            **{f"val_{k}": val_metrics[k] for k in ("singlet_energy_eV_mae", "triplet_energy_eV_mae") if k in val_metrics},
            "val_log1p_oscillator_strength_mae": val_metrics.get("log1p_oscillator_strength_mae"),
            "lr_backbone": max(g["lr"] for g in optimizer.param_groups[:-1]),
            "lr_head": optimizer.param_groups[-1]["lr"],
            "train_seconds": train_seconds,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "skipped_nonfinite_batches": skipped,
        }
        if aux_cfg:
            record["aux_train_loss"] = aux_running / max(aux_seen, 1)
            record["aux_val_energy_eV_mae"] = evaluate_aux(eval_model, aux_loaders["val"], device, autocast_dtype)
            record["epoch_seconds"] = time.perf_counter() - epoch_start
        if shots_iter is not None:
            record["shots_train_loss"] = shots_running / max(shots_seen, 1)
        history.append(record)
        # checkpoint_metric "last_epoch" keeps the final epoch (a fixed fine-tuning schedule).
        selection_value = -float(epoch) if checkpoint_metric == "last_epoch" else record.get(checkpoint_metric)
        if selection_value is None or not math.isfinite(float(selection_value)):
            raise ValueError(f"checkpoint_metric {checkpoint_metric!r} missing or non-finite; record keys: {sorted(record)}")
        if selection_value < best_val:
            best_val, best_epoch = float(selection_value), epoch
            torch.save(
                {
                    "model_state_dict": eval_model.state_dict(),
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
        summary.update(updated_at=_utc_now(), latest_epoch=epoch, best_epoch=best_epoch, best_checkpoint_metric_value=best_val,
                       latest_val_metrics=val_metrics)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        elapsed_hours = (time.perf_counter() - started) / 3600
        if epoch < epochs and elapsed_hours + 1.2 * record["epoch_seconds"] / 3600 > max_hours:
            summary["stop_reason"] = f"max_hours {max_hours} reached after epoch {epoch}"
            print(summary["stop_reason"], flush=True)
            break

    if soup_paths:
        states = [torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"] for p in soup_paths]
        model.load_state_dict(average_state_dicts(states, eval_cfg.get("soup_weights")))
        best_epoch = None
    else:
        best = torch.load(eval_checkpoint or checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        if eval_checkpoint:
            best_epoch = best.get("epoch")
    keys = [str(k) for k in arrays["molecule_key"]]
    predictions_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".test_predictions.csv"))
    val_predictions_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".val_predictions.csv"))
    val_metrics = evaluate(model, val_loader, target_columns, loss_cfg, device, autocast_dtype, keys, val_predictions_path)
    test_metrics = evaluate(model, test_loader, target_columns, loss_cfg, device, autocast_dtype, keys, predictions_path)
    summary.update(status="completed", completed_at=_utc_now(), best_epoch=best_epoch, val_metrics=val_metrics,
                   test_metrics=test_metrics, predictions_csv_path=str(predictions_path),
                   val_predictions_csv_path=str(val_predictions_path))
    if aux_cfg:
        summary["aux"]["val_energy_eV_mae"] = evaluate_aux(model, aux_loaders["val"], device, autocast_dtype)
    # post_inference: {pool_path, out_csv} or a list of them: the selected model also predicts unlabeled
    # pools (TASK-287; lists since TASK-310). summary["post_inference"] mirrors the config's shape.
    post_cfg = config.get("post_inference") or []
    done = []
    for item in [post_cfg] if isinstance(post_cfg, dict) else post_cfg:
        n = predict_pool(model, item["pool_path"], item["out_csv"], target_columns, device,
                         int(item.get("batch_size", train_cfg["batch_size"])), int(train_cfg.get("num_workers", 8)))
        done.append({"pool_path": item["pool_path"], "out_csv": item["out_csv"], "molecules": n})
    if done:
        summary["post_inference"] = done[0] if isinstance(post_cfg, dict) else done
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
