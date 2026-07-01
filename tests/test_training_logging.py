from __future__ import annotations

import csv
import json
from pathlib import Path

from gnn_excited.train import collect_run_metadata, write_history_csv, write_summary_json


def test_write_history_csv_uses_union_of_metric_keys(tmp_path: Path) -> None:
    output = tmp_path / "metrics.csv"
    history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 3.0},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.5, "val_S1_eV_mae": 0.4},
    ]

    write_history_csv(output, history)

    with output.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["epoch"] == "1"
    assert rows[1]["val_S1_eV_mae"] == "0.4"


def test_write_summary_json_serializes_paths(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    payload = {"checkpoint_path": tmp_path / "model.pt", "split_sizes": {"train": 8}}

    write_summary_json(output, payload)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["checkpoint_path"].endswith("model.pt")
    assert data["split_sizes"] == {"train": 8}


def test_collect_run_metadata_includes_reproducibility_fields(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    metadata = collect_run_metadata("cpu")

    assert metadata["python_version"]
    assert "git_dirty" in metadata
    assert "torch_version" in metadata
    assert "torch_geometric_version" in metadata
    assert metadata["slurm"]["job_id"] == "12345"
