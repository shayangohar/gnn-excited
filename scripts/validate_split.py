from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gnn_excited.data.pyg_dataset import explicit_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate persisted split columns against a processed manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--columns", nargs="+", default=["random_split", "scaffold_split"])
    args = parser.parse_args()

    with Path(args.manifest).open("r", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status") == "ok"]
    report: dict[str, object] = {"manifest_ok_rows": len(rows), "splits": {}}
    for column in args.columns:
        train, val, test = explicit_split(rows, args.splits, column)
        report["splits"][column] = {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(train) + len(val) + len(test),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
