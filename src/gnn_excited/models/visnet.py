"""One-pass ViSNet transfer model for paired QM9GWBSE targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    import torch
    from torch import nn
    from torch_geometric.nn.models import ViSNet
    from torch_geometric.utils import scatter
except ModuleNotFoundError as exc:  # pragma: no cover - optional ML dependency.
    torch = None
    nn = None
    ViSNet = None
    scatter = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from gnn_excited.losses import target_layout


if nn is not None:

    class ViSNetOnePass(nn.Module):
        """Evaluate one shared ViSNet encoder, then split energy/oscillator heads."""

        def __init__(self, target_columns: Sequence[str], hidden_channels: int = 128, **kwargs: Any):
            super().__init__()
            self.target_columns = tuple(str(column) for column in target_columns)
            self.energy_indices, self.oscillator_indices = target_layout(self.target_columns)
            self.hidden_channels = int(hidden_channels)
            encoder_kwargs = dict(kwargs)
            encoder_kwargs.pop("output_channels", None)
            encoder_kwargs.pop("out_channels", None)
            if "layers" in encoder_kwargs and "num_layers" not in encoder_kwargs:
                encoder_kwargs["num_layers"] = encoder_kwargs.pop("layers")
            encoder_kwargs["hidden_channels"] = self.hidden_channels
            self.encoder = ViSNet(**encoder_kwargs)
            self.energy_head = nn.Linear(self.hidden_channels, len(self.energy_indices))
            self.oscillator_head = nn.Linear(self.hidden_channels, len(self.oscillator_indices))

        @property
        def energy_readout(self):
            return self.energy_head

        @property
        def oscillator_readout(self):
            return self.oscillator_head

        def forward(self, z, pos, batch=None):
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            representation = getattr(self.encoder, "representation_model", None)
            if representation is None:
                representation = getattr(self.encoder, "representation", None)
            if representation is None:
                raise RuntimeError("Installed PyG ViSNet has no representation_model; cannot expose a shared readout")
            embedding, _ = representation(z, pos, batch)
            embedding = scatter(embedding, batch, dim=0, reduce=getattr(self.encoder, "reduce_op", "sum"))
            energy = self.energy_head(embedding)
            oscillator = self.oscillator_head(embedding)
            output = embedding.new_empty((embedding.size(0), len(self.target_columns)))
            output[:, list(self.energy_indices)] = energy
            output[:, list(self.oscillator_indices)] = oscillator
            return output

        def freeze_encoder(self) -> None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

        def unfreeze_encoder(self) -> None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)


else:

    class ViSNetOnePass:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("ViSNetOnePass requires torch and torch_geometric") from _IMPORT_ERROR


def build_visnet(**kwargs: Any):
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError("ViSNet requires torch and torch_geometric") from _IMPORT_ERROR
    target_columns = kwargs.pop("target_columns", None)
    if target_columns is None:
        raise ValueError("build_visnet requires target_columns")
    return ViSNetOnePass(target_columns=target_columns, **kwargs)


def _checkpoint_state(checkpoint: str | Path | dict[str, Any], map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dictionary or a mapping containing model_state_dict")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain model_state_dict/state_dict")
    return state


def load_transfer_checkpoint(
    model: ViSNetOnePass,
    checkpoint: str | Path | dict[str, Any],
    *,
    mode: str = "readout_only",
    map_location: str | torch.device = "cpu",
) -> ViSNetOnePass:
    """Load a same-architecture checkpoint in frozen-readout or full mode."""
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError("Checkpoint loading requires torch and torch_geometric") from _IMPORT_ERROR
    mode = {"readout": "readout_only", "frozen_readout": "readout_only", "full": "full_finetune", "finetune": "full_finetune"}.get(str(mode).lower(), str(mode).lower())
    if mode not in {"readout_only", "full_finetune"}:
        raise ValueError("transfer mode must be readout_only or full_finetune")
    state = _checkpoint_state(checkpoint, map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Incompatible ViSNet checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if mode == "readout_only":
        model.freeze_encoder()
        return model

    model.unfreeze_encoder()
    return model
