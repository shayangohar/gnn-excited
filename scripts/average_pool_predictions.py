#!/usr/bin/env python3
"""Average several models' pool prediction CSVs into one seed-ensemble CSV (TASK-320).

Each input has molecule_key + prediction_<column> columns (scripts/train_uma_qm9gwbse.py predict_pool).
Molecules present in every input are kept, in the first input's order; every prediction column is the
mean over inputs. Used to score 3-seed ensembles with scripts/geometry_training_report.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tables = []
    for path in args.inputs:
        with open(path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = [c for c in reader.fieldnames if c != "molecule_key"]
            tables.append({row["molecule_key"]: [float(row[c]) for c in columns] for row in reader})
    keys = [k for k in tables[0] if all(k in t for t in tables[1:])]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_key", *columns])
        for k in keys:
            writer.writerow([k, *(f"{v:.6f}" for v in np.mean([t[k] for t in tables], axis=0))])
    print(f"{len(keys)} molecules averaged over {len(tables)} inputs -> {args.out}")


if __name__ == "__main__":
    main()
