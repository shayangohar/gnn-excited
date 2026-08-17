"""One-pass ViSNet models for paired QM9GWBSE targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    import torch
    from torch import nn
    from torch_geometric.nn.models import ViSNet
    from torch_geometric.nn.models.visnet import GatedEquivariantBlock
    from torch_geometric.utils import scatter
except ModuleNotFoundError as exc:  # pragma: no cover - optional ML dependency.
    torch = None
    nn = None
    ViSNet = None
    GatedEquivariantBlock = None
    scatter = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from gnn_excited.losses import target_layout


DEBYE_TO_ATOMIC_UNITS = 0.393430307
EV_PER_HARTREE = 27.211386245988
OSCILLATOR_STRENGTH_FACTOR = (2.0 / 3.0) * DEBYE_TO_ATOMIC_UNITS**2 / EV_PER_HARTREE


if nn is not None:

    class MultiTargetEquivariantScalar(nn.Module):
        """ViSNet's gated scalar output extended to multiple targets."""

        def __init__(self, hidden_channels: int, out_channels: int):
            super().__init__()
            self.output_network = nn.ModuleList(
                [
                    GatedEquivariantBlock(
                        hidden_channels,
                        hidden_channels // 2,
                        scalar_activation=True,
                    ),
                    GatedEquivariantBlock(hidden_channels // 2, out_channels),
                ]
            )

        def forward(self, scalar, vector):
            for layer in self.output_network:
                scalar, vector = layer(scalar, vector)
            return scalar + vector.sum() * 0

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
            self.reduce_op = str(encoder_kwargs.pop("reduce_op", "mean"))
            self.encoder = ViSNet(**encoder_kwargs).representation_model
            self.energy_head = MultiTargetEquivariantScalar(self.hidden_channels, len(self.energy_indices))
            self.oscillator_head = MultiTargetEquivariantScalar(self.hidden_channels, len(self.oscillator_indices))

        @property
        def energy_readout(self):
            return self.energy_head

        @property
        def oscillator_readout(self):
            return self.oscillator_head

        def forward(self, z, pos, batch=None):
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            scalar, vector = self.encoder(z, pos, batch)
            energy = scatter(self.energy_head(scalar, vector), batch, dim=0, reduce=self.reduce_op)
            oscillator = scatter(self.oscillator_head(scalar, vector), batch, dim=0, reduce=self.reduce_op)
            output = scalar.new_empty((energy.size(0), len(self.target_columns)))
            output[:, list(self.energy_indices)] = energy
            output[:, list(self.oscillator_indices)] = oscillator
            return output

        def freeze_encoder(self) -> None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

        def unfreeze_encoder(self) -> None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)


    class ViSNetSpectroscopyDecoder(ViSNetOnePass):
        """Decode five queried excited states from one shared ViSNet encoding."""

        def __init__(
            self,
            target_columns: Sequence[str],
            hidden_channels: int = 128,
            num_states: int = 5,
            dipole_reduce_op: str = "sum",
            **kwargs: Any,
        ):
            super().__init__(target_columns, hidden_channels, **kwargs)
            if len(self.energy_indices) != num_states or len(self.oscillator_indices) != num_states:
                raise ValueError(
                    f"Spectroscopy decoder requires {num_states} paired energy/oscillator states"
                )
            del self.energy_head, self.oscillator_head
            self.num_states = int(num_states)
            self.dipole_reduce_op = str(dipole_reduce_op)
            self.state_queries = nn.Embedding(self.num_states, self.hidden_channels)
            self.state_decoder = nn.ModuleList(
                [
                    GatedEquivariantBlock(
                        self.hidden_channels,
                        self.hidden_channels // 2,
                        scalar_activation=True,
                    ),
                    GatedEquivariantBlock(self.hidden_channels // 2, 1),
                ]
            )

        def forward(self, z, pos, batch=None):
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            scalar, vector = self.encoder(z, pos, batch)
            energies = []
            dipoles = []
            for query in self.state_queries.weight:
                state_scalar = scalar + query
                state_vector = vector
                for layer in self.state_decoder:
                    state_scalar, state_vector = layer(state_scalar, state_vector)
                energies.append(scatter(state_scalar, batch, dim=0, reduce=self.reduce_op).squeeze(-1))
                dipoles.append(
                    scatter(state_vector, batch, dim=0, reduce=self.dipole_reduce_op).squeeze(-1)
                )
            energy = torch.stack(energies, dim=1)
            transition_dipole = torch.stack(dipoles, dim=1)
            oscillator = (
                OSCILLATOR_STRENGTH_FACTOR
                * energy.clamp_min(0)
                * transition_dipole.pow(2).sum(dim=-1)
            )
            output = scalar.new_empty((energy.size(0), len(self.target_columns)))
            output[:, list(self.energy_indices)] = energy
            output[:, list(self.oscillator_indices)] = torch.log1p(oscillator)
            return output, transition_dipole


    class ViSNetTransitionDipoleSidecar(ViSNetOnePass):
        """Preserve a trained scalar model while learning equivariant transition dipoles."""

        def __init__(
            self,
            target_columns: Sequence[str],
            hidden_channels: int = 128,
            num_states: int = 5,
            dipole_reduce_op: str = "sum",
            **kwargs: Any,
        ):
            super().__init__(target_columns, hidden_channels, **kwargs)
            if len(self.energy_indices) != num_states or len(self.oscillator_indices) != num_states:
                raise ValueError(
                    f"Transition-dipole sidecar requires {num_states} paired energy/oscillator states"
                )
            self.num_states = int(num_states)
            self.dipole_reduce_op = str(dipole_reduce_op)
            self.state_queries = nn.Embedding(self.num_states, self.hidden_channels)
            self.dipole_decoder = nn.ModuleList(
                [
                    GatedEquivariantBlock(
                        self.hidden_channels,
                        self.hidden_channels // 2,
                        scalar_activation=True,
                    ),
                    GatedEquivariantBlock(self.hidden_channels // 2, 1),
                ]
            )

        def forward(self, z, pos, batch=None):
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            scalar, vector = self.encoder(z, pos, batch)
            energy = scatter(self.energy_head(scalar, vector), batch, dim=0, reduce=self.reduce_op)
            dipoles = []
            for query in self.state_queries.weight:
                state_scalar = scalar + query
                state_vector = vector
                for layer in self.dipole_decoder:
                    state_scalar, state_vector = layer(state_scalar, state_vector)
                dipoles.append(
                    scatter(state_vector, batch, dim=0, reduce=self.dipole_reduce_op).squeeze(-1)
                )
            transition_dipole = torch.stack(dipoles, dim=1)
            oscillator = (
                OSCILLATOR_STRENGTH_FACTOR
                * energy.clamp_min(0)
                * transition_dipole.pow(2).sum(dim=-1)
            )
            output = scalar.new_empty((energy.size(0), len(self.target_columns)))
            output[:, list(self.energy_indices)] = energy
            output[:, list(self.oscillator_indices)] = torch.log1p(oscillator)
            return output, transition_dipole

        def freeze_scalar_model(self) -> None:
            for module in (self.encoder, self.energy_head, self.oscillator_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)


else:

    class ViSNetOnePass:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("ViSNetOnePass requires torch and torch_geometric") from _IMPORT_ERROR

    ViSNetSpectroscopyDecoder = ViSNetOnePass
    ViSNetTransitionDipoleSidecar = ViSNetOnePass


def build_visnet(**kwargs: Any):
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError("ViSNet requires torch and torch_geometric") from _IMPORT_ERROR
    target_columns = kwargs.pop("target_columns", None)
    if target_columns is None:
        raise ValueError("build_visnet requires target_columns")
    spectroscopy_decoder = kwargs.pop("spectroscopy_decoder", False)
    transition_dipole_sidecar = kwargs.pop("transition_dipole_sidecar", False)
    if spectroscopy_decoder and transition_dipole_sidecar:
        raise ValueError("Choose either spectroscopy_decoder or transition_dipole_sidecar")
    model_class = (
        ViSNetTransitionDipoleSidecar
        if transition_dipole_sidecar
        else ViSNetSpectroscopyDecoder if spectroscopy_decoder else ViSNetOnePass
    )
    return model_class(target_columns=target_columns, **kwargs)


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
    """Load a full compatible checkpoint or initialize only the shared encoder."""
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError("Checkpoint loading requires torch and torch_geometric") from _IMPORT_ERROR
    mode = {
        "readout": "readout_only",
        "frozen_readout": "readout_only",
        "full": "full_finetune",
        "finetune": "full_finetune",
        "encoder_only": "encoder_finetune",
        "sidecar": "frozen_sidecar",
    }.get(str(mode).lower(), str(mode).lower())
    if mode not in {"readout_only", "full_finetune", "encoder_finetune", "frozen_sidecar"}:
        raise ValueError(
            "transfer mode must be readout_only, full_finetune, encoder_finetune, "
            "or frozen_sidecar"
        )
    state = _checkpoint_state(checkpoint, map_location)
    if mode == "encoder_finetune":
        encoder_state = {key: value for key, value in state.items() if key.startswith("encoder.")}
        if not encoder_state:
            raise ValueError("Checkpoint does not contain encoder.* parameters")
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        missing_encoder = [key for key in missing if key.startswith("encoder.")]
        if missing_encoder or unexpected:
            raise ValueError(
                f"Incompatible ViSNet encoder checkpoint: missing={sorted(missing_encoder)}, "
                f"unexpected={sorted(unexpected)}"
            )
        model.unfreeze_encoder()
        return model
    missing, unexpected = model.load_state_dict(state, strict=False)
    if mode == "frozen_sidecar":
        allowed_missing = ("state_queries.", "dipole_decoder.")
        incompatible_missing = [key for key in missing if not key.startswith(allowed_missing)]
        if incompatible_missing or unexpected or not hasattr(model, "freeze_scalar_model"):
            raise ValueError(
                f"Incompatible transition-dipole sidecar checkpoint: "
                f"missing={sorted(incompatible_missing)}, unexpected={sorted(unexpected)}"
            )
        model.freeze_scalar_model()
        return model
    if missing or unexpected:
        raise ValueError(f"Incompatible ViSNet checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if mode == "readout_only":
        model.freeze_encoder()
        return model

    model.unfreeze_encoder()
    return model
