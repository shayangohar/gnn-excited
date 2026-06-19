from __future__ import annotations

import argparse
import json

from gnn_excited.inference.smiles import predict_smiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict S1 properties from a SMILES string.")
    parser.add_argument("smiles")
    parser.add_argument("--checkpoint", default="checkpoints/dimenetpp_a9_small.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        result = predict_smiles(args.smiles, args.checkpoint, device=args.device)
    except (ModuleNotFoundError, ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"Prediction failed: {exc}") from None
    print(json.dumps(result, indent=2))
