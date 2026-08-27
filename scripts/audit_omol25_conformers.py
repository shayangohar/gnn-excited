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
    parser.add_argument("--max-rows", type=int, default=6000)
    args = parser.parse_args()

    shards = sorted(Path(args.lmdb_dir).rglob("*.aselmdb"))[: args.max_shards]
    print(f"shards: {len(shards)}", flush=True)

    samples: dict[str, Counter] = {}
    identity: Counter = Counter()
    total = 0
    for shard in shards:
        db = connect(str(shard))
        for row in db.select():
            data = getattr(row, "_data", None) or {}
            for key in ("data_id", "composition", "charge", "reference_source", "source", "spin"):
                samples.setdefault(key, Counter())[str(data.get(key))[:40]] += 1
            src = str(data.get("source"))
            base = src.rsplit("_confo", 1)[0] if "_confo" in src else src.split("/")[-1]
            ident = base or str(data.get("composition"))
            identity[ident] += 1
            total += 1
            if total >= args.max_rows:
                break
        if total >= args.max_rows:
            break

    for key, counter in samples.items():
        print(f"--- {key} top values: {sorted(counter.items(), key=lambda kv: -kv[1])[:4]}", flush=True)

    counts = list(identity.values())
    mult = Counter(counts)
    print(f"rows scanned: {total}", flush=True)
    print(f"unique (composition|charge) identities: {len(identity)}", flush=True)
    print(f"rows-per-identity distribution (multiplicity: num identities): {sorted(mult.items())}", flush=True)
    singles = sum(v for k, v in mult.items() if k == 1)
    print(f"fraction with >1 row: {1 - singles / len(identity):.4f}", flush=True)


if __name__ == "__main__":
    main()
