#!/usr/bin/env python3
"""Lock dev and test halves of the external test sets (TASK-328, DOC-327 section 5).

Until now PC9, OE62L, OE62H and the external QM9 set served as both development and test sets: their
numbers chose he4/he5, the heads-vs-delta design and the few-shot setup. From this split on, every
decision and every few-shot label uses the dev half only; the test half is scored once, for the paper.

Strata: OE62H by its set of non-CHONF elements (signatures with fewer than --min-stratum molecules are
pooled as "rare"); the other sets by heavy-atom-count tercile within the set. Each stratum is shuffled
with a seed derived from --seed and the stratum name and split in half; odd strata alternate which half
gets the extra molecule. Writes --out (molecule_key, set, half, stratum, heavy_atoms) and prints the
counts per set and half and the file's sha256, which is recorded in Stele.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

SETS = ("QM9", "PC9", "OE62L", "OE62H")


def strata(rows: list[dict], min_stratum: int) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in SETS:
        members = [r for r in rows if r["set"] == s]
        if s == "OE62H":
            sig = {r["molecule_key"]: "+".join(sorted(set(r["elements"].split("-")) - set("HCNOF"))) for r in members}
            counts = Counter(sig.values())
            for r in members:
                name = sig[r["molecule_key"]] if counts[sig[r["molecule_key"]]] >= min_stratum else "rare"
                groups[f"OE62H:{name}"].append(r)
        else:
            sizes = sorted(int(r["heavy_atoms"]) for r in members)
            cuts = (sizes[len(sizes) // 3], sizes[2 * len(sizes) // 3])
            for r in members:
                h = int(r["heavy_atoms"])
                groups[f"{s}:size{0 if h < cuts[0] else 1 if h < cuts[1] else 2}"].append(r)
    return groups


def assign(groups: dict[str, list[dict]], seed: int) -> dict[str, tuple[str, str]]:
    halves, extra_to_dev = {}, True
    for name in sorted(groups):
        keys = sorted(r["molecule_key"] for r in groups[name])
        random.Random(f"{seed}:{name}").shuffle(keys)
        n_dev = len(keys) // 2 + (len(keys) % 2 if extra_to_dev else 0)
        if len(keys) % 2:
            extra_to_dev = not extra_to_dev
        for i, k in enumerate(keys):
            halves[k] = ("dev" if i < n_dev else "test", name)
    return halves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/processed/external_testsets/labels.csv")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--min-stratum", type=int, default=4)
    parser.add_argument("--out", default="configs/splits/external_dev_test.csv")
    args = parser.parse_args()
    with open(args.labels, newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["set"] in SETS]
    halves = assign(strata(rows, args.min_stratum), args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["molecule_key", "set", "half", "stratum", "heavy_atoms"])
        for r in sorted(rows, key=lambda r: (SETS.index(r["set"]), r["molecule_key"])):
            writer.writerow([r["molecule_key"], r["set"], *halves[r["molecule_key"]], r["heavy_atoms"]])
    counts = Counter((r["set"], halves[r["molecule_key"]][0]) for r in rows)
    for s in SETS:
        print(f"{s:6s} dev {counts[(s, 'dev')]:4d}  test {counts[(s, 'test')]:4d}")
    print("sha256", hashlib.sha256(out.read_bytes()).hexdigest(), out)


if __name__ == "__main__":
    main()
