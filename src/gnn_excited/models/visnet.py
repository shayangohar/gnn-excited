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

    class ViSNetElectronicDescriptorDelta(ViSNetTransitionDipoleSidecar):
        """Learn per-root electronic residuals on top of a frozen scalar ViSNet."""

        def __init__(
            self,
            *args,
            descriptor_dim: int,
            delta_use_geometry_context: bool = True,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.descriptor_dim = int(descriptor_dim)
            self.delta_use_geometry_context = bool(delta_use_geometry_context)
            self.energy_state_queries = nn.Embedding(
                self.num_states, self.hidden_channels
            )
            self.energy_descriptor_projection = nn.Sequential(
                nn.Linear(self.descriptor_dim, self.hidden_channels),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels),
            )
            self.dipole_descriptor_projection = nn.Sequential(
                nn.Linear(self.descriptor_dim, self.hidden_channels),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels),
            )
            self.energy_delta_head = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, 1),
            )
            nn.init.zeros_(self.energy_delta_head[-1].weight)
            nn.init.zeros_(self.energy_delta_head[-1].bias)

        def forward(self, z, pos, batch=None, electronic_descriptors=None):
            if electronic_descriptors is None:
                raise ValueError(
                    "Electronic descriptor delta model requires electronic_descriptors"
                )
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            scalar, vector = self.encoder(z, pos, batch)
            baseline_energy = scatter(
                self.energy_head(scalar, vector),
                batch,
                dim=0,
                reduce=self.reduce_op,
            )
            batch_size = baseline_energy.size(0)
            descriptors = electronic_descriptors.view(
                batch_size, self.num_states, self.descriptor_dim
            )
            energy_features = self.energy_descriptor_projection(descriptors)
            if self.delta_use_geometry_context:
                geometry_context = scatter(
                    scalar, batch, dim=0, reduce=self.reduce_op
                ).unsqueeze(1)
                energy_features = energy_features + geometry_context
            energy_features = (
                energy_features + self.energy_state_queries.weight.unsqueeze(0)
            )
            energy = baseline_energy + self.energy_delta_head(
                energy_features
            ).squeeze(-1)

            dipole_features = self.dipole_descriptor_projection(descriptors)
            dipoles = []
            for state, query in enumerate(self.state_queries.weight):
                state_scalar = scalar + query + dipole_features[:, state, :][batch]
                state_vector = vector
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
            output = scalar.new_empty((batch_size, len(self.target_columns)))
            output[:, list(self.energy_indices)] = energy
            output[:, list(self.oscillator_indices)] = torch.log1p(oscillator)
            return output, transition_dipole

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

    class ViSNetLatentHamiltonian(ViSNetOnePass):
        """Frozen scalar ViSNet whose state energies are the sorted eigenvalues of a
        learned molecular tridiagonal Hamiltonian (ordering by construction)."""

        def __init__(
            self,
            target_columns: Sequence[str],
            hidden_channels: int = 128,
            num_states: int = 5,
            pooling: Sequence[str] = ("mean", "sum"),
            hamiltonian_hidden: int = 256,
            anchor_production_energies: bool = False,
            per_atom_parameters: bool = False,
            offdiag_scale: float = 0.5,
            diag_bias_low: float = 3.0,
            diag_bias_high: float = 8.0,
            **kwargs: Any,
        ):
            super().__init__(target_columns, hidden_channels, **kwargs)
            if len(self.energy_indices) != int(num_states):
                raise ValueError(
                    f"Latent-Hamiltonian decoder requires {num_states} energy states"
                )
            self.num_states = int(num_states)
            poolings = tuple(str(op) for op in pooling)
            if not poolings or any(op not in ("mean", "sum") for op in poolings):
                raise ValueError("pooling must contain only 'mean'/'sum' operations")
            self.poolings = poolings
            self.anchor_production_energies = bool(anchor_production_energies)
            self.per_atom_parameters = bool(per_atom_parameters)
            if self.per_atom_parameters and self.anchor_production_energies:
                raise ValueError(
                    "per_atom_parameters with anchor_production_energies is not supported"
                )
            self.offdiag_scale = float(offdiag_scale)
            feature_dim = (
                self.hidden_channels
                if self.per_atom_parameters
                else self.hidden_channels * len(self.poolings)
            )
            if self.anchor_production_energies:
                feature_dim += self.num_states
            width = int(hamiltonian_hidden)
            self.hamiltonian_mlp = nn.Sequential(
                nn.Linear(feature_dim, width),
                nn.SiLU(),
                nn.Linear(width, width // 2),
                nn.SiLU(),
                nn.Linear(width // 2, 2 * self.num_states - 1),
            )
            # Deterministic near-data initialization: zero weights with diagonal
            # biases spanning the excitation range, so the initial Hamiltonian is
            # diagonal with strictly ascending eigenvalues near the label scale.
            final = self.hamiltonian_mlp[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            with torch.no_grad():
                final.bias[: self.num_states] = torch.linspace(
                    float(diag_bias_low), float(diag_bias_high), self.num_states
                )

        def freeze_scalar_model(self) -> None:
            for module in (self.encoder, self.energy_head, self.oscillator_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

        def forward(self, z, pos, batch=None):
            if batch is None:
                batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            scalar, vector = self.encoder(z, pos, batch)
            if self.per_atom_parameters:
                # Per-atom readout (as in the production heads) followed by
                # mean aggregation: per-atom expressivity before pooling.
                parameters = scatter(
                    self.hamiltonian_mlp(scalar), batch, dim=0, reduce="mean"
                )
            else:
                pooled = [
                    scatter(scalar, batch, dim=0, reduce=op) for op in self.poolings
                ]
                features = (
                    pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=-1)
                )
                if self.anchor_production_energies:
                    production_energy = scatter(
                        self.energy_head(scalar, vector),
                        batch,
                        dim=0,
                        reduce=self.reduce_op,
                    )
                    features = torch.cat(
                        [features, production_energy.detach()], dim=-1
                    )
                parameters = self.hamiltonian_mlp(features)
            states = torch.arange(self.num_states, device=parameters.device)
            hamiltonian = parameters.new_zeros(
                (parameters.size(0), self.num_states, self.num_states)
            )
            hamiltonian[:, states, states] = parameters[:, : self.num_states]
            offdiagonal = self.offdiag_scale * torch.tanh(parameters[:, self.num_states :])
            adjacent = states[:-1]
            hamiltonian[:, adjacent, adjacent + 1] = offdiagonal
            hamiltonian[:, adjacent + 1, adjacent] = offdiagonal
            energy = torch.linalg.eigvalsh(hamiltonian)
            oscillator = scatter(
                self.oscillator_head(scalar, vector),
                batch,
                dim=0,
                reduce=self.reduce_op,
            )
            output = scalar.new_empty((energy.size(0), len(self.target_columns)))
            output[:, list(self.energy_indices)] = energy.to(output.dtype)
            output[:, list(self.oscillator_indices)] = oscillator.to(output.dtype)
            return output

else:

    class ViSNetOnePass:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "ViSNetOnePass requires torch and torch_geometric"
            ) from _IMPORT_ERROR

    ViSNetSpectroscopyDecoder = ViSNetOnePass
    ViSNetTransitionDipoleSidecar = ViSNetOnePass
    ViSNetElectronicDescriptorDelta = ViSNetOnePass
    ViSNetTransitionRefinementSidecar = ViSNetOnePass
    ViSNetStateConditionedTransitionRefinementSidecar = ViSNetOnePass
    ViSNetLatentHamiltonian = ViSNetOnePass


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
    electronic_descriptor_delta = kwargs.pop("electronic_descriptor_delta", False)
    transition_refinement_sidecar = kwargs.pop("transition_refinement_sidecar", False)
    state_conditioned_transition_refinement = kwargs.pop(
        "state_conditioned_transition_refinement", False
    )
    latent_hamiltonian = kwargs.pop("latent_hamiltonian", False)
    if (
        sum(
            map(
                bool,
                (
                    spectroscopy_decoder,
                    transition_dipole_sidecar,
                    electronic_descriptor_delta,
                    transition_refinement_sidecar,
                    state_conditioned_transition_refinement,
                    latent_hamiltonian,
                ),
            )
        )
        > 1
    ):
        raise ValueError("Choose one ViSNet spectroscopy architecture")
    if electronic_descriptor_delta:
        model_class = ViSNetElectronicDescriptorDelta
    elif state_conditioned_transition_refinement:
        model_class = ViSNetStateConditionedTransitionRefinementSidecar
    elif transition_refinement_sidecar:
        model_class = ViSNetTransitionRefinementSidecar
    elif transition_dipole_sidecar:
        model_class = ViSNetTransitionDipoleSidecar
    elif spectroscopy_decoder:
        model_class = ViSNetSpectroscopyDecoder
    elif latent_hamiltonian:
        model_class = ViSNetLatentHamiltonian
    else:
        model_class = ViSNetOnePass
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
            "energy_state_queries.",
            "energy_descriptor_projection.",
            "dipole_descriptor_projection.",
            "energy_delta_head.",
            "transition_refinement.",
            "transition_refinement_gate",
            "hamiltonian_mlp.",
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
