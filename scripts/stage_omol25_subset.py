#!/usr/bin/env python3
"""Stage OMol25 subsets from the gated Hugging Face repository.

The repo requires a HF account with Meta's license accepted plus a read token.
Token sources (first match wins): $HF_TOKEN, then /u/mgohar/.omol25_hf_token
(mode 600). The token never enters Git, Slurm files, or logs.

Usage:
  python scripts/stage_omol25_subset.py --list-only
  python scripts/stage_omol25_subset.py --download relative/file/path [more...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "facebook/omol25"
DEFAULT_DEST = Path("/work/hdd/bhzu/gnn-excited/data/omol25")


def resolve_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path("/u/mgohar/.omol25_hf_token")
    if token_path.exists():
        return token_path.read_text().strip()
    sys.exit(
        "No HF token found. Set $HF_TOKEN or write the token to "
        "/u/mgohar/.omol25_hf_token (chmod 600). Accept the license at "
        "https://huggingface.co/datasets/facebook/omol25 first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true", help="print inventory and exit")
    parser.add_argument(
        "--download",
        nargs="*",
        default=[],
        help="repo-relative files to fetch into DEST",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    api = HfApi(token=resolve_token())
    listing = api.list_repo_tree(REPO, repo_type="dataset", recursive=False)
    entries = []
    for entry in listing:
        if getattr(entry, "size", None) is not None:
            entries.append((entry.path, entry.size or 0))
    total_bytes = sum(size for _, size in entries)
    print(f"top-level entries: {len(entries)} | visible bytes: {total_bytes / 1e9:.2f} GB")
    for path, size in sorted(entries, key=lambda item: item[1]):
        print(f"{size / 1e9:10.3f} GB  {path}")

    if args.list_only or not args.download:
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    for relative in args.download:
        local = hf_hub_download(
            REPO,
            relative,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN") or resolve_token(),
            local_dir=args.dest,
        )
        print(f"downloaded: {local}")


if __name__ == "__main__":
    main()
