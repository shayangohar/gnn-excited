from __future__ import annotations

import argparse
from pathlib import Path

from gnn_excited.data.qm9gwbse import build_qm9gwbse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact QM9GWBSE HDF5 and validation metadata from ZIP archives.")
    parser.add_argument("--raw-dir", default="data/raw/qm9gwbse")
    parser.add_argument("--out-dir", default="data/processed/qm9gwbse")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--identity-csv", default=None, help="Explicit qm9_id-to-SMILES mapping; never inferred from molecule keys")
    parser.add_argument("--qcdge-identity-csv", default=None, help="Optional explicit QCDGE identity table for overlap audit")
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--skip-md5-validation", action="store_true", help="Skip official Zenodo MD5 checks (synthetic fixtures only)")
    args = parser.parse_args()
    result = build_qm9gwbse(
        Path(args.raw_dir), Path(args.out_dir), seed=args.seed,
        train_fraction=args.train_fraction, val_fraction=args.val_fraction,
        identity_csv=args.identity_csv, qcdge_identity_csv=args.qcdge_identity_csv, max_count=args.max_count,
        validate_md5=not args.skip_md5_validation,
    )
    print(result)


if __name__ == "__main__":
    main()
