from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_metrics_csv_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".metrics.csv")


def _default_summary_json_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".summary.json")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _run_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_run_metadata(device: str) -> dict[str, Any]:
    git_status = _run_command(["git", "status", "--short"])
    metadata: dict[str, Any] = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": _run_command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "git_status_short": git_status or "",
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
        },
    }
    if torch is not None:
        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["cuda_version"] = torch.version.cuda
        metadata["cudnn_version"] = torch.backends.cudnn.version()
        metadata["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else None
        )
    try:
        import torch_geometric
    except ModuleNotFoundError:
        metadata["torch_geometric_version"] = None
    else:
        metadata["torch_geometric_version"] = torch_geometric.__version__
    return metadata


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_summary_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


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

    config_path = Path(config_path)
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]
    model_kwargs = dict(config["model"])
    device = train_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is False")

    checkpoint_path = Path(train_cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = Path(train_cfg.get("metrics_csv_path") or _default_metrics_csv_path(checkpoint_path))
    summary_json_path = Path(train_cfg.get("summary_json_path") or _default_summary_json_path(checkpoint_path))

    rows = _load_manifest_rows(dataset_cfg["manifest_path"])
    manifest_ok_rows = len(rows)
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
    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch: int | None = None
    started_at = _utc_now()
    run_summary: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "metrics_csv_path": str(metrics_csv_path),
        "summary_json_path": str(summary_json_path),
        "device": device,
        "environment": collect_run_metadata(device),
        "manifest_ok_rows": manifest_ok_rows,
        "dataset_rows_used": len(rows),
        "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "config": config,
        "best_epoch": None,
        "best_val_loss": None,
        "latest_epoch": 0,
        "latest_metrics": None,
        "test_metrics": None,
    }
    write_summary_json(summary_json_path, run_summary)

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
        print(record, flush=True)
        write_history_csv(metrics_csv_path, history)

        val_loss = val_metrics.get("loss", math.inf)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "config": config,
                    "history": history,
                },
                checkpoint_path,
            )
        run_summary.update(
            {
                "updated_at": _utc_now(),
                "best_epoch": best_epoch,
                "best_val_loss": best_val if math.isfinite(best_val) else None,
                "latest_epoch": epoch,
                "latest_metrics": record,
            }
        )
        write_summary_json(summary_json_path, run_summary)

    test_metrics = evaluate(model, test_loader, device) if len(test_ds) else {}
    run_summary.update(
        {
            "status": "completed",
            "completed_at": _utc_now(),
            "updated_at": _utc_now(),
            "test_metrics": test_metrics,
        }
    )
    write_summary_json(summary_json_path, run_summary)
    return {
        "history": history,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "metrics_csv_path": str(metrics_csv_path),
        "summary_json_path": str(summary_json_path),
    }
