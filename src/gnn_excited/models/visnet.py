"""One-pass ViSNet models for paired QM9GWBSE targets."""

from __future__ import annotations

from copy import deepcopy
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

        def __init__(
            self,
            target_columns: Sequence[str],
            hidden_channels: int = 128,
            **kwargs: Any,
        ):
            super().__init__()
            self.target_columns = tuple(str(column) for column in target_columns)
            self.energy_indices, self.oscillator_indices = target_layout(
                self.target_columns
            )
            self.hidden_channels = int(hidden_channels)
            encoder_kwargs = dict(kwargs)
            encoder_kwargs.pop("output_channels", None)
            encoder_kwargs.pop("out_channels", None)
            if "layers" in encoder_kwargs and "num_layers" not in encoder_kwargs:
                encoder_kwargs["num_layers"] = encoder_kwargs.pop("layers")
            encoder_kwargs["hidden_channels"] = self.hidden_channels
            self.reduce_op = str(encoder_kwargs.pop("reduce_op", "mean"))
            self.encoder = ViSNet(**encoder_kwargs).representation_model
            self.energy_head = MultiTargetEquivariantScalar(
                self.hidden_channels, len(self.energy_indices)
            )
            self.oscillator_head = MultiTargetEquivariantScalar(
                self.hidden_channels, len(self.oscillator_indices)
            )

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
            energy = scatter(
                self.energy_head(scalar, vector), batch, dim=0, reduce=self.reduce_op
            )
            oscillator = scatter(
                self.oscillator_head(scalar, vector),
                batch,
                dim=0,
                reduce=self.reduce_op,
            )
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
            if (
                len(self.energy_indices) != num_states
                or len(self.oscillator_indices) != num_states
            ):
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
                energies.append(
                    scatter(state_scalar, batch, dim=0, reduce=self.reduce_op).squeeze(
                        -1
                    )
                )
                dipoles.append(
                    scatter(
                        state_vector, batch, dim=0, reduce=self.dipole_reduce_op
                    ).squeeze(-1)
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
            if (
                len(self.energy_indices) != num_states
                or len(self.oscillator_indices) != num_states
            ):
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

        def _transition_features(self, z, pos, batch, scalar, vector):
            return scalar, vector

        def _query_features(self, z, pos, batch, scalar, vector, query):
            return scalar + query, vector

        def forward(self, z, pos, batch=None):
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            scalar, vector = self.encoder(z, pos, batch)
            energy = scatter(
                self.energy_head(scalar, vector), batch, dim=0, reduce=self.reduce_op
            )
            transition_scalar, transition_vector = self._transition_features(
                z, pos, batch, scalar, vector
            )
            dipoles = []
            for query in self.state_queries.weight:
                state_scalar, state_vector = self._query_features(
                    z, pos, batch, transition_scalar, transition_vector, query
                )
                for layer in self.dipole_decoder:
                    state_scalar, state_vector = layer(state_scalar, state_vector)
                dipoles.append(
                    scatter(
                        state_vector, batch, dim=0, reduce=self.dipole_reduce_op
                    ).squeeze(-1)
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

    class ViSNetTransitionRefinementSidecar(ViSNetTransitionDipoleSidecar):
        """Add one trainable ViSNet interaction only to the transition path."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.transition_refinement = deepcopy(self.encoder.vis_mp_layers[-1])
            self.transition_refinement_gate = nn.Parameter(torch.zeros(()))

        def initialize_transition_refinement(self) -> None:
            self.transition_refinement.load_state_dict(
                self.encoder.vis_mp_layers[-1].state_dict()
            )

        def _refine(self, z, pos, batch, scalar, vector):
            edge_index, edge_weight, edge_vector = self.encoder.distance(pos, batch)
            radial = self.encoder.distance_expansion(edge_weight)
            mask = edge_index[0] != edge_index[1]
            edge_vector[mask] = edge_vector[mask] / edge_weight[mask].unsqueeze(1)
            edge_vector = self.encoder.sphere(edge_vector)
            edge_features = self.encoder.edge_embedding(edge_index, radial, scalar)
            delta_scalar, delta_vector, _ = self.transition_refinement(
                scalar,
                vector,
                edge_index,
                edge_weight,
                edge_features,
                edge_vector,
            )
            return (
                scalar + self.transition_refinement_gate * delta_scalar,
                vector + self.transition_refinement_gate * delta_vector,
            )

        def _transition_features(self, z, pos, batch, scalar, vector):
            return self._refine(z, pos, batch, scalar, vector)

    class ViSNetStateConditionedTransitionRefinementSidecar(
        ViSNetTransitionRefinementSidecar
    ):
        """Inject each state query before the shared transition interaction."""

        def _transition_features(self, z, pos, batch, scalar, vector):
            return scalar, vector

        def _query_features(self, z, pos, batch, scalar, vector, query):
            return self._refine(z, pos, batch, scalar + query, vector)

else:

    class ViSNetOnePass:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "ViSNetOnePass requires torch and torch_geometric"
            ) from _IMPORT_ERROR

    ViSNetSpectroscopyDecoder = ViSNetOnePass
    ViSNetTransitionDipoleSidecar = ViSNetOnePass
    ViSNetTransitionRefinementSidecar = ViSNetOnePass
    ViSNetStateConditionedTransitionRefinementSidecar = ViSNetOnePass


def build_visnet(**kwargs: Any):
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "ViSNet requires torch and torch_geometric"
        ) from _IMPORT_ERROR
    target_columns = kwargs.pop("target_columns", None)
    if target_columns is None:
        raise ValueError("build_visnet requires target_columns")
    spectroscopy_decoder = kwargs.pop("spectroscopy_decoder", False)
    transition_dipole_sidecar = kwargs.pop("transition_dipole_sidecar", False)
    transition_refinement_sidecar = kwargs.pop("transition_refinement_sidecar", False)
    state_conditioned_transition_refinement = kwargs.pop(
        "state_conditioned_transition_refinement", False
    )
    if (
        sum(
            map(
                bool,
                (
                    spectroscopy_decoder,
                    transition_dipole_sidecar,
                    transition_refinement_sidecar,
                    state_conditioned_transition_refinement,
                ),
            )
        )
        > 1
    ):
        raise ValueError("Choose one ViSNet spectroscopy architecture")
    model_class = (
        ViSNetStateConditionedTransitionRefinementSidecar
        if state_conditioned_transition_refinement
        else (
            ViSNetTransitionRefinementSidecar
            if transition_refinement_sidecar
            else (
                ViSNetTransitionDipoleSidecar
                if transition_dipole_sidecar
                else (
                    ViSNetSpectroscopyDecoder if spectroscopy_decoder else ViSNetOnePass
                )
            )
        )
    )
    return model_class(target_columns=target_columns, **kwargs)


def _checkpoint_state(
    checkpoint: str | Path | dict[str, Any], map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must be a state dictionary or a mapping containing model_state_dict"
        )
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
        raise ModuleNotFoundError(
            "Checkpoint loading requires torch and torch_geometric"
        ) from _IMPORT_ERROR
    mode = {
        "readout": "readout_only",
        "frozen_readout": "readout_only",
        "full": "full_finetune",
        "finetune": "full_finetune",
        "encoder_only": "encoder_finetune",
        "sidecar": "frozen_sidecar",
    }.get(str(mode).lower(), str(mode).lower())
    if mode not in {
        "readout_only",
        "full_finetune",
        "encoder_finetune",
        "frozen_sidecar",
    }:
        raise ValueError(
            "transfer mode must be readout_only, full_finetune, encoder_finetune, "
            "or frozen_sidecar"
        )
    state = _checkpoint_state(checkpoint, map_location)
    if mode == "encoder_finetune":
        encoder_state = {
            key: value for key, value in state.items() if key.startswith("encoder.")
        }
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
        allowed_missing = (
            "state_queries.",
            "dipole_decoder.",
            "transition_refinement.",
            "transition_refinement_gate",
        )
        incompatible_missing = [
            key for key in missing if not key.startswith(allowed_missing)
        ]
        if (
            incompatible_missing
            or unexpected
            or not hasattr(model, "freeze_scalar_model")
        ):
            raise ValueError(
                f"Incompatible transition-dipole sidecar checkpoint: "
                f"missing={sorted(incompatible_missing)}, unexpected={sorted(unexpected)}"
            )
        if hasattr(model, "initialize_transition_refinement"):
            model.initialize_transition_refinement()
        model.freeze_scalar_model()
        return model
    if missing or unexpected:
        raise ValueError(
            f"Incompatible ViSNet checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if mode == "readout_only":
        model.freeze_encoder()
        return model

    model.unfreeze_encoder()
    return model
