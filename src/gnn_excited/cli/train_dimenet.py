from __future__ import annotations

import argparse

from gnn_excited.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small DimeNet++ model on QCDGE A_9.")
    parser.add_argument("--config", default="configs/small_cpu.yaml")
    args = parser.parse_args()
    try:
        result = train_from_config(args.config)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from None
    print(result)
