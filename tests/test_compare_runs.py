from __future__ import annotations

import csv
import json
from pathlib import Path

from gnn_excited.cli.compare_runs import summarize_run


def test_summarize_run_reports_best_rows(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    metrics_path = runs_dir / "example.metrics.csv"
    summary_path = runs_dir / "example.summary.json"

    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "val_loss", "val_S1_eV_mae"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "val_loss": 0.4, "val_S1_eV_mae": 0.5})
        writer.writerow({"epoch": 2, "val_loss": 0.2, "val_S1_eV_mae": 0.3})
        writer.writerow({"epoch": 3, "val_loss": 0.3, "val_S1_eV_mae": 0.1})

    summary_path.write_text(
        json.dumps(
            {
                "dataset_rows_used": 10000,
                "metrics_csv_path": "runs/example.metrics.csv",
                "best_epoch": 2,
                "best_val_loss": 0.2,
                "latest_epoch": 3,
                "latest_metrics": {"val_loss": 0.3, "val_S1_eV_mae": 0.1},
                "test_metrics": {"S1_eV_mae": 0.25},
                "environment": {"git_commit": "abc", "git_dirty": False, "slurm": {"job_id": "123"}},
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_run(summary_path)

    assert summary["dataset_rows_used"] == 10000
    assert summary["best_val_loss_epoch_from_csv"] == "2"
    assert summary["best_val_S1_eV_mae_epoch"] == "3"
    assert summary["test_metrics"] == {"S1_eV_mae": 0.25}
    assert summary["slurm_job_id"] == "123"
