from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest

from gnn_excited.train import (
    WandbRun,
    build_loss_weights,
    build_scheduler,
    classify_validation_improvement,
    collect_run_metadata,
    seed_everything,
    weighted_mse_loss,
    write_history_csv,
    write_summary_json,
)


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


def test_seed_everything_repeats_python_numpy_and_torch_rngs() -> None:
    torch = pytest.importorskip("torch")

    first_metadata = seed_everything(17)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )
    second_metadata = seed_everything(17)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first_metadata["seed"] == 17
    assert second_metadata["python_hash_seed"] == "17"
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_build_scheduler_supports_reduce_on_plateau() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)

    scheduler = build_scheduler(
        optimizer,
        {"type": "reduce_on_plateau", "factor": 0.5, "patience": 0, "min_lr": 1e-6},
    )

    assert scheduler is not None
    scheduler.step(1.0)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_build_scheduler_rejects_unknown_type() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)

    with pytest.raises(ValueError, match="Unsupported scheduler type"):
        build_scheduler(optimizer, {"type": "cosine"})


def test_validation_improvement_separates_checkpoint_from_early_stopping() -> None:
    improvement = classify_validation_improvement(
        val_loss=0.99995,
        best_val=1.0,
        early_stopping_best_val=1.0,
        min_delta=0.0001,
    )

    assert improvement["checkpoint_improved"] is True
    assert improvement["early_stopping_improved"] is False


def test_wandb_run_disabled_without_importing_wandb() -> None:
    run = WandbRun({"wandb": {"enabled": False}}, {"config_path": "config.yaml"})

    assert run.enabled is False
    assert run.metadata() is None


def test_batch_mae_reports_multistate_metrics() -> None:
    torch = pytest.importorskip('torch')
    from gnn_excited.train import batch_mae

    pred = torch.tensor([[1.0, 0.0, 3.0, 0.0]])
    target = torch.tensor([[2.0, 0.0, 1.0, 0.0]])
    metrics = batch_mae(pred, target, ('S1_eV', 'log1p_S1_f', 'S2_eV', 'log1p_S2_f'))

    assert metrics['S1_eV_mae'] == pytest.approx(1.0)
    assert metrics['S2_eV_mae'] == pytest.approx(2.0)
    assert metrics['energy_eV_mae'] == pytest.approx(1.5)
    assert metrics['oscillator_strength_mae'] == pytest.approx(0.0)


def test_build_loss_weights_supports_energy_oscillator_defaults() -> None:
    config = {
        'loss': {
            'type': 'weighted_mse',
            'energy_weight': 1.5,
            'oscillator_weight': 1.0,
        }
    }

    weights = build_loss_weights(config, ('S1_eV', 'log1p_S1_f', 'S2_eV', 'log1p_S2_f'))

    assert weights == pytest.approx((1.5, 1.0, 1.5, 1.0))


def test_build_loss_weights_allows_per_column_overrides() -> None:
    config = {
        'loss': {
            'type': 'weighted_mse',
            'energy_weight': 1.5,
            'oscillator_weight': 1.0,
            'weights': {'S1_eV': 2.0},
        }
    }

    weights = build_loss_weights(config, ('S1_eV', 'log1p_S1_f', 'S2_eV', 'log1p_S2_f'))

    assert weights == pytest.approx((2.0, 1.0, 1.5, 1.0))


def test_weighted_mse_loss_normalizes_by_weight_sum() -> None:
    torch = pytest.importorskip('torch')
    pred = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 0.0]])

    assert weighted_mse_loss(pred, target, (3.0, 1.0)).item() == pytest.approx(0.75)
