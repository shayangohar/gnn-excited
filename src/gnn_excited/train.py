from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    DataLoader = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from gnn_excited.data.pyg_dataset import QCDGES1Dataset, deterministic_split
from gnn_excited.models.dimenetpp import build_dimenetpp


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row.get("status") == "ok"]


def _subset_keys(rows: list[dict[str, str]], indices: list[int]) -> list[str]:
    return [rows[i]["molecule_key"] for i in indices]


def batch_mae(pred, target) -> dict[str, float]:
    energy_mae = (pred[:, 0] - target[:, 0]).abs().mean().item()
    log_f_mae = (pred[:, 1] - target[:, 1]).abs().mean().item()
    f_mae = (torch.expm1(pred[:, 1]).clamp_min(0) - torch.expm1(target[:, 1])).abs().mean().item()
    return {"S1_eV_mae": energy_mae, "log1p_S1_f_mae": log_f_mae, "S1_f_mae": f_mae}


def evaluate(model, loader, device: str) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "S1_eV_mae": 0.0, "log1p_S1_f_mae": 0.0, "S1_f_mae": 0.0, "n": 0}
    loss_fn = torch.nn.MSELoss(reduction="mean")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.y.view(-1, 2)
            pred = model(batch.z, batch.pos, batch.batch).view(-1, 2)
            loss = loss_fn(pred, target)
            metrics = batch_mae(pred, target)
            batch_n = target.shape[0]
            totals["loss"] += loss.item() * batch_n
            for key, value in metrics.items():
                totals[key] += value * batch_n
            totals["n"] += batch_n
    n = max(totals.pop("n"), 1)
    return {key: value / n for key, value in totals.items()}


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError("Training requires torch and torch_geometric.") from _IMPORT_ERROR

    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    model_kwargs = dict(config["model"])
    device = train_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is False")

    rows = _load_manifest_rows(dataset_cfg["manifest_path"])
    max_rows = dataset_cfg.get("max_manifest_molecules")
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    train_idx, val_idx, test_idx = deterministic_split(
        rows,
        seed=int(dataset_cfg["split_seed"]),
        train_fraction=float(dataset_cfg["train_fraction"]),
        val_fraction=float(dataset_cfg["val_fraction"]),
    )
    train_ds = QCDGES1Dataset(dataset_cfg["hdf5_path"], dataset_cfg["manifest_path"], _subset_keys(rows, train_idx))
    val_ds = QCDGES1Dataset(dataset_cfg["hdf5_path"], dataset_cfg["manifest_path"], _subset_keys(rows, val_idx))
    test_ds = QCDGES1Dataset(dataset_cfg["hdf5_path"], dataset_cfg["manifest_path"], _subset_keys(rows, test_idx))

    train_loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False)

    model = build_dimenetpp(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    loss_fn = torch.nn.MSELoss()
    history = []
    best_val = math.inf
    checkpoint_path = Path(train_cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.y.view(-1, 2)
            pred = model(batch.z, batch.pos, batch.batch).view(-1, 2)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_n = target.shape[0]
            total_loss += loss.item() * batch_n
            total_n += batch_n

        val_metrics = evaluate(model, val_loader, device) if len(val_ds) else {"loss": float("nan")}
        record = {"epoch": epoch, "train_loss": total_loss / max(total_n, 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(record)
        val_loss = val_metrics.get("loss", math.inf)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "config": config,
                    "history": history,
                },
                checkpoint_path,
            )

    test_metrics = evaluate(model, test_loader, device) if len(test_ds) else {}
    return {"history": history, "test_metrics": test_metrics, "checkpoint_path": str(checkpoint_path)}
