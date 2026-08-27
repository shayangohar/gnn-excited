#!/usr/bin/env python3
"""Audit OMol25 raw LMDB for conformer multiplicity per molecule identity."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ase.db import connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb-dir", required=True)
    parser.add_argument("--max-shards", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=20000)
    args = parser.parse_args()

    shards = sorted(Path(args.lmdb_dir).rglob("*.aselmdb"))[: args.max_shards]
    print(f"shards: {len(shards)}", flush=True)

    identity_key = None
    per_identity: Counter = Counter()
    total = 0
    for shard in shards:
        db = connect(str(shard))
        for row in db.select():
            data = getattr(row, "_data", None) or {}
            if identity_key is None:
                candidates = [k for k in data if "mol" in k.lower() or "conf" in k.lower() or "smil" in k.lower() or k in ("name", "id", "molecule_id", "conformer_id")]
                print(f"data keys (sample): {sorted(data.keys())[:20]}", flush=True)
                print(f"identity candidates: {candidates}", flush=True)
                identity_key = candidates[0] if candidates else None
            if identity_key is None:
                per_identity[row.id] += 1
            else:
                per_identity[str(data.get(identity_key))] += 1
            total += 1
            if total >= args.max_rows:
                break
        if total >= args.max_rows:
            break

    counts = list(per_identity.values())
    mult = Counter(counts)
    print(f"rows scanned: {total}", flush=True)
    print(f"identity field used: {identity_key}", flush=True)
    print(f"unique identities: {len(per_identity)}", flush=True)
    print(f"rows-per-identity distribution (multiplicity: num identities): {sorted(mult.items())}", flush=True)
    print(f"fraction identities with >1 conformer: {sum(v for k, v in mult.items() if k > 1) / len(per_identity):.4f}", flush=True)


if __name__ == "__main__":
    main()
