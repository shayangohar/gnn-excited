#!/usr/bin/env python3
"""Frozen per-atom UMA features for the bet A decoder ladder (TASK-277 / TASK-310).

Writes, for every molecule of --packed (all splits, packed order), the rich-readout input of
scripts/train_uma_qm9gwbse.py: spherical invariants (l=0 channels and per-l norms) of the final node
embedding and of the last --readout-layers block outputs (128 x 3 x 2 = 768 values per atom for UMA-S).

  --checkpoint <trained .pt> --config <its training YAML>   features of a fine-tuned backbone
  --checkpoint <uma-s-1p2p1.pt> --pretrained                features of the pretrained backbone

Outputs <out>.features.npy (float16, [atoms, dim], row order = packed atom order) and <out>.meta.npz
(atom_offsets, molecule_key, split, targets, target_columns), so decoders train without the backbone.
Runs in envs/uma-fairchem.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from train_uma_qm9gwbse import PackedMolecules, UMAExcitedState, collate, load_backbone, patch_mole_split_forward

DEFAULT_COLUMNS = ["S1_eV", "log1p_S1_f", "S2_eV", "log1p_S2_f", "S3_eV", "log1p_S3_f",
                   "S4_eV", "log1p_S4_f", "S5_eV", "log1p_S5_f"]


def build_model(args, arrays) -> UMAExcitedState:
    if args.pretrained:
        backbone = load_backbone(args.checkpoint)
        columns, readout_layers, lmax, aux = DEFAULT_COLUMNS, args.readout_layers, 2, None
    else:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        backbone = load_backbone(config["model"]["checkpoint_path"])
        columns = list(config["targets"]["columns"])
        readout_layers, lmax = int(config["model"].get("readout_layers", 0)), int(config["model"].get("lmax", 2))
        aux = config.get("aux")
        if config["model"].get("readout") != "rich":
            raise SystemExit("feature extraction needs a rich-readout model")
    n = len(columns)
    aux_n = len(aux["columns"]) if aux else 0
    model = UMAExcitedState(backbone, columns, np.zeros(n), np.ones(n),
                            embed_dim=int(getattr(backbone, "sphere_channels", 128)), readout="rich",
                            readout_layers=readout_layers, lmax=lmax,
                            aux_columns=aux["columns"] if aux else None, aux_mean=np.zeros(aux_n), aux_std=np.ones(aux_n),
                            aux_mode=aux.get("mode", "heads") if aux else "heads")
    if not args.pretrained:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict(state)
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed", default="data/processed/qm9gwbse/qm9gwbse_packed.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="training YAML of --checkpoint (fine-tuned models)")
    parser.add_argument("--pretrained", action="store_true", help="--checkpoint is a fairchem UMA checkpoint")
    parser.add_argument("--readout-layers", type=int, default=1, help="pretrained only; fine-tuned use their config")
    parser.add_argument("--out", required=True, help="output prefix, e.g. data/processed/uma_features/ft_richind_s17")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N molecules only (smoke tests)")
    args = parser.parse_args()
    if not args.pretrained and not args.config:
        raise SystemExit("--config is required unless --pretrained")
    device = torch.device("cuda")
    patch_mole_split_forward()
    arrays = dict(np.load(args.packed, allow_pickle=False))
    n_mol = arrays["molecule_key"].shape[0] if args.limit is None else min(args.limit, arrays["molecule_key"].shape[0])
    model = build_model(args, arrays).to(device).eval()
    available = [str(c) for c in arrays["target_columns"]]
    dataset = PackedMolecules(arrays, np.arange(n_mol), np.arange(len(available)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    offsets = arrays["atom_offsets"][: n_mol + 1].astype(np.int64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features = None
    cursor, started = 0, time.perf_counter()
    for step, (batch, _, rows) in enumerate(loader, start=1):
        embedding = model.backbone(batch.to(device))
        per_atom = model.rich_features(embedding).cpu().numpy().astype(np.float16)
        if features is None:
            features = np.lib.format.open_memmap(f"{out}.features.npy", mode="w+", dtype=np.float16,
                                                 shape=(int(offsets[-1]), per_atom.shape[1]))
        expected = int(offsets[int(rows[-1]) + 1] - offsets[int(rows[0])])
        if per_atom.shape[0] != expected or int(offsets[int(rows[0])]) != cursor:
            raise RuntimeError(f"atom count mismatch at batch {step}: {per_atom.shape[0]} vs {expected}")
        features[cursor:cursor + per_atom.shape[0]] = per_atom
        cursor += per_atom.shape[0]
        if step % 200 == 0:
            print(f"{int(rows[-1]) + 1}/{n_mol} molecules, {time.perf_counter() - started:.0f}s", flush=True)
    features.flush()
    if cursor != int(offsets[-1]):
        raise RuntimeError(f"wrote {cursor} atoms, expected {int(offsets[-1])}")
    np.savez(f"{out}.meta.npz", atom_offsets=offsets, molecule_key=arrays["molecule_key"][:n_mol],
             split=arrays["split"][:n_mol], targets=arrays["targets"][:n_mol], target_columns=arrays["target_columns"])
    info = {"checkpoint": args.checkpoint, "pretrained": args.pretrained, "molecules": int(n_mol), "atoms": cursor,
            "dim": int(features.shape[1]), "seconds": time.perf_counter() - started}
    Path(f"{out}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
