from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _coerce_metric_value(value: str) -> float | int | str:
    if value == "":
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _resolve_metrics_path(summary_path: Path, summary: dict[str, Any]) -> Path:
    metrics_path = Path(summary["metrics_csv_path"])
    if metrics_path.is_absolute():
        return metrics_path
    return summary_path.parent.parent / metrics_path


def backfill_run(summary_path: Path, entity: str, project: str, group: str | None, dry_run: bool = False) -> str:
    summary = _load_json(summary_path)
    metrics_path = _resolve_metrics_path(summary_path, summary)
    metrics = _load_metrics(metrics_path)
    run_name = summary_path.stem.replace(".summary", "")
    if dry_run:
        return f"would upload {run_name}: {len(metrics)} epochs from {metrics_path}"

    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        group=group,
        job_type="backfill",
        config={
            "summary": summary,
            "source_summary_path": str(summary_path),
            "source_metrics_path": str(metrics_path),
        },
        reinit=True,
    )
    try:
        for row in metrics:
            payload = {key: _coerce_metric_value(value) for key, value in row.items() if key != "epoch"}
            run.log(payload, step=int(row["epoch"]))
        test_metrics = summary.get("test_metrics") or {}
        if test_metrics:
            run.log({f"test_{key}": value for key, value in test_metrics.items()})
        if summary.get("best_epoch") is not None:
            run.summary["best_epoch"] = summary["best_epoch"]
        if summary.get("best_val_loss") is not None:
            run.summary["best_val_loss"] = summary["best_val_loss"]
        run.summary["dataset_rows_used"] = summary.get("dataset_rows_used")
        run.summary["status"] = summary.get("status")
        run.summary["source_summary_path"] = str(summary_path)
    finally:
        run.finish()
    return f"uploaded {run_name}: {len(metrics)} epochs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill completed local training runs into Weights & Biases.")
    parser.add_argument("summaries", nargs="+", type=Path, help="Run summary JSON paths")
    parser.add_argument("--entity", default="shayangohar2007-fresno-state")
    parser.add_argument("--project", default="gnn-excited")
    parser.add_argument("--group", default="historical")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for summary_path in args.summaries:
        print(backfill_run(summary_path, args.entity, args.project, args.group, args.dry_run), flush=True)


if __name__ == "__main__":
    main()
