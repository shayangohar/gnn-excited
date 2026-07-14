from __future__ import annotations

import argparse
import json

from gnn_excited.data.audit import run_qcdge_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit QCDGE identities, checksums, deduplication, manifests, and data splits."
    )
    parser.add_argument("--csv", required=True, help="Path to final_all.csv")
    parser.add_argument("--hdf5", required=True, help="Path to final_all.hdf5")
    parser.add_argument("--manifest", required=True, help="Existing target manifest to filter")
    parser.add_argument("--out-dir", required=True, help="Directory for audit reports and split tables")
    parser.add_argument("--dedup-manifest", required=True, help="Filtered output manifest path")
    parser.add_argument("--checksum-file", default=None, help="Optional upstream SHA512SUM path")
    parser.add_argument("--compressed-hdf5", default=None, help="Optional .hdf5.gz to hash after decompression")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    fractions = (args.train_fraction, args.val_fraction, 1.0 - args.train_fraction - args.val_fraction)
    report = run_qcdge_audit(
        csv_path=args.csv,
        hdf5_path=args.hdf5,
        source_manifest_path=args.manifest,
        output_dir=args.out_dir,
        deduplicated_manifest_path=args.dedup_manifest,
        checksum_path=args.checksum_file,
        compressed_hdf5_path=args.compressed_hdf5,
        seed=args.seed,
        fractions=fractions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
