from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float_or_none(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def summarize_run(summary_path: Path) -> dict[str, Any]:
    summary = _load_summary(summary_path)
    metrics_path = Path(summary.get("metrics_csv_path") or summary_path.with_suffix(".metrics.csv"))
    if not metrics_path.is_absolute():
        metrics_path = summary_path.parent.parent / metrics_path
    rows = _load_metrics(metrics_path)
    best_loss_row = min(rows, key=lambda row: float(row["val_loss"])) if rows else {}
    best_energy_row = min(rows, key=lambda row: float(row["val_S1_eV_mae"])) if rows else {}
    environment = summary.get("environment", {})
    return {
        "summary_path": str(summary_path),
        "dataset_rows_used": summary.get("dataset_rows_used"),
        "split_sizes": summary.get("split_sizes"),
        "status": summary.get("status"),
        "stopped_early": summary.get("stopped_early"),
        "stop_reason": summary.get("stop_reason"),
        "best_epoch": summary.get("best_epoch"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_loss_epoch_from_csv": best_loss_row.get("epoch"),
        "best_val_S1_eV_mae": _float_or_none(best_energy_row, "val_S1_eV_mae"),
        "best_val_S1_eV_mae_epoch": best_energy_row.get("epoch"),
        "final_epoch": summary.get("latest_epoch"),
        "final_val_loss": (summary.get("latest_metrics") or {}).get("val_loss"),
        "final_val_S1_eV_mae": (summary.get("latest_metrics") or {}).get("val_S1_eV_mae"),
        "test_metrics": summary.get("test_metrics"),
        "git_commit": environment.get("git_commit") or summary.get("git_commit"),
        "git_dirty": environment.get("git_dirty") if "environment" in summary else summary.get("git_dirty"),
        "cuda_device_name": environment.get("cuda_device_name") or summary.get("cuda_device_name"),
        "slurm_job_id": (environment.get("slurm") or {}).get("job_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed training run summaries.")
    parser.add_argument("summaries", nargs="+", type=Path, help="One or more run summary JSON paths")
    args = parser.parse_args()

    for summary_path in args.summaries:
        print(json.dumps(summarize_run(summary_path), indent=2, sort_keys=True))

