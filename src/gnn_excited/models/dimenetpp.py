from __future__ import annotations

from typing import Callable

import torch
from torch import nn

try:
    from torch.nn import Linear
    from torch_geometric.nn import radius_graph
    from torch_geometric.nn.inits import glorot_orthogonal
    from torch_geometric.nn.models.dimenet import triplets
    from torch_geometric.nn.models import DimeNetPlusPlus
    from torch_geometric.nn.resolver import activation_resolver
    from torch_geometric.utils import scatter
except ModuleNotFoundError as exc:  # pragma: no cover
    Linear = None
    DimeNetPlusPlus = None
    glorot_orthogonal = None
    radius_graph = None
    triplets = None
    activation_resolver = None
    scatter = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_DimeNetPlusPlusBase = DimeNetPlusPlus if DimeNetPlusPlus is not None else nn.Module

UNDERSCORE = chr(95)
EV_SUFFIX = UNDERSCORE + chr(101) + chr(86)
F_SUFFIX = UNDERSCORE + chr(102)
LOG1P_PREFIX = chr(108) + chr(111) + chr(103) + chr(49) + chr(112) + UNDERSCORE
OUT_CHANNELS = chr(111) + chr(117) + chr(116) + UNDERSCORE + chr(99) + chr(104) + chr(97) + chr(110) + chr(110) + chr(101) + chr(108) + chr(115)
HEAD_TYPE = chr(104) + chr(101) + chr(97) + chr(100) + UNDERSCORE + chr(116) + chr(121) + chr(112) + chr(101)
OUTPUT_HEAD = chr(111) + chr(117) + chr(116) + chr(112) + chr(117) + chr(116) + UNDERSCORE + chr(104) + chr(101) + chr(97) + chr(100)
TARGET_COLUMNS = chr(116) + chr(97) + chr(114) + chr(103) + chr(101) + chr(116) + UNDERSCORE + chr(99) + chr(111) + chr(108) + chr(117) + chr(109) + chr(110) + chr(115)
SINGLE = chr(115) + chr(105) + chr(110) + chr(103) + chr(108) + chr(101)
STANDARD = chr(115) + chr(116) + chr(97) + chr(110) + chr(100) + chr(97) + chr(114) + chr(100)
SPLIT_ENERGY_OSCILLATOR = chr(115) + chr(112) + chr(108) + chr(105) + chr(116) + UNDERSCORE + chr(101) + chr(110) + chr(101) + chr(114) + chr(103) + chr(121) + UNDERSCORE + chr(111) + chr(115) + chr(99) + chr(105) + chr(108) + chr(108) + chr(97) + chr(116) + chr(111) + chr(114)
SPLIT_HEADS = chr(115) + chr(112) + chr(108) + chr(105) + chr(116) + UNDERSCORE + chr(104) + chr(101) + chr(97) + chr(100) + chr(115)
SHARED_SPLIT_ENERGY_OSCILLATOR = 'shared' + UNDERSCORE + 'split' + UNDERSCORE + 'energy' + UNDERSCORE + 'oscillator'
SHARED_SPLIT_HEADS = 'shared' + UNDERSCORE + 'split' + UNDERSCORE + 'heads'
SPLIT_OUTPUT_ENERGY_OSCILLATOR = 'split' + UNDERSCORE + 'output' + UNDERSCORE + 'energy' + UNDERSCORE + 'oscillator'
STATE_CONDITIONED_GEOMETRY = 'state' + UNDERSCORE + 'conditioned' + UNDERSCORE + 'geometry'
STATE_CONDITIONED = 'state' + UNDERSCORE + 'conditioned'


def _energy_oscillator_indices(target_columns, out_channels: int) -> tuple[list[int], list[int]]:
    if target_columns is None:
        if out_channels % 2:
            raise ValueError('Split energy/oscillator heads require an even output dimension.')
        return list(range(0, out_channels, 2)), list(range(1, out_channels, 2))

    energy_indices = [idx for idx, column in enumerate(target_columns) if column.endswith(EV_SUFFIX)]
    oscillator_indices = [
        idx for idx, column in enumerate(target_columns) if column.startswith(LOG1P_PREFIX) and column.endswith(F_SUFFIX)
    ]
    if len(energy_indices) + len(oscillator_indices) != len(target_columns):
        raise ValueError('All target columns must be energy *_eV or log1p_*_f oscillator targets.')
    if not energy_indices or not oscillator_indices:
        raise ValueError('Split energy/oscillator heads require at least one target of each type.')
    return energy_indices, oscillator_indices


def _state_target_layout(target_columns) -> tuple[tuple[str, int, int], ...]:
    if target_columns is None:
        raise ValueError('State-conditioned heads require explicit target_columns.')

    states: dict[str, dict[str, int]] = {}
    for idx, column in enumerate(target_columns):
        column = str(column)
        if column.endswith(EV_SUFFIX):
            state = column[: -len(EV_SUFFIX)]
            property_name = 'energy'
        elif column.startswith(LOG1P_PREFIX) and column.endswith(F_SUFFIX):
            state = column[len(LOG1P_PREFIX) : -len(F_SUFFIX)]
            property_name = 'oscillator'
        else:
            raise ValueError('State-conditioned targets must be energy *_eV or log1p_*_f oscillator targets.')
        if not state:
            raise ValueError(f'Could not infer an electronic state from target column {column!r}.')
        state_targets = states.setdefault(state, {})
        if property_name in state_targets:
            raise ValueError(f'Duplicate {property_name} target for electronic state {state}.')
        state_targets[property_name] = idx

    def state_sort_key(item: tuple[str, dict[str, int]]) -> tuple[str, int, str]:
        state = item[0]
        prefix = ''.join(character for character in state if not character.isdigit())
        suffix = ''.join(character for character in state if character.isdigit())
        return prefix, int(suffix) if suffix else -1, state

    layout = []
    for state, state_targets in sorted(states.items(), key=state_sort_key):
        missing = {'energy', 'oscillator'} - state_targets.keys()
        if missing:
            raise ValueError(f'Electronic state {state} is missing targets for: {sorted(missing)}.')
        layout.append((state, state_targets['energy'], state_targets['oscillator']))
    if not layout:
        raise ValueError('State-conditioned heads require at least one electronic state.')
    return tuple(layout)


class SplitOutputPPBlock(nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        target_dim: int,
        energy_indices: list[int],
        oscillator_indices: list[int],
        num_layers: int,
        act: Callable,
        output_initializer: str = 'zeros',
    ):
        if output_initializer not in {'zeros', 'glorot_orthogonal'}:
            raise ValueError('output_initializer must be "zeros" or "glorot_orthogonal".')
        super().__init__()
        self.act = act
        self.output_initializer = output_initializer
        self.target_dim = target_dim
        self.register_buffer('energy_indices', torch.tensor(energy_indices, dtype=torch.long), persistent=False)
        self.register_buffer('oscillator_indices', torch.tensor(oscillator_indices, dtype=torch.long), persistent=False)
        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = nn.ModuleList([Linear(out_emb_channels, out_emb_channels) for _ in range(num_layers)])
        self.energy_lin = Linear(out_emb_channels, len(energy_indices), bias=False)
        self.oscillator_lin = Linear(out_emb_channels, len(oscillator_indices), bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        for lin in (self.energy_lin, self.oscillator_lin):
            if self.output_initializer == 'zeros':
                lin.weight.data.fill_(0)
            elif self.output_initializer == 'glorot_orthogonal':
                glorot_orthogonal(lin.weight, scale=2.0)

    def forward(self, x, rbf, i, num_nodes: int | None = None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        energy = self.energy_lin(x)
        oscillator = self.oscillator_lin(x)
        output = x.new_empty((x.size(0), self.target_dim))
        output[:, self.energy_indices.to(output.device)] = energy
        output[:, self.oscillator_indices.to(output.device)] = oscillator
        return output


class SplitEnergyOscillatorDimeNetPlusPlus(nn.Module):
    def __init__(self, target_columns=None, **kwargs):
        super().__init__()
        out_channels = int(kwargs.get(OUT_CHANNELS))
        energy_indices, oscillator_indices = _energy_oscillator_indices(target_columns, out_channels)

        base_kwargs = dict(kwargs)
        base_kwargs.pop(OUT_CHANNELS, None)
        self.register_buffer('energy_indices', torch.tensor(energy_indices, dtype=torch.long), persistent=False)
        self.register_buffer('oscillator_indices', torch.tensor(oscillator_indices, dtype=torch.long), persistent=False)
        self.target_dim = out_channels
        self.energy_model = DimeNetPlusPlus(**base_kwargs, out_channels=len(energy_indices))
        self.oscillator_model = DimeNetPlusPlus(**base_kwargs, out_channels=len(oscillator_indices))

    def forward(self, z, pos, batch=None):
        energy = self.energy_model(z, pos, batch)
        oscillator = self.oscillator_model(z, pos, batch)
        output = energy.new_empty((energy.size(0), self.target_dim))
        output[:, self.energy_indices.to(output.device)] = energy
        output[:, self.oscillator_indices.to(output.device)] = oscillator
        return output


class SharedBackboneSplitEnergyOscillatorDimeNetPlusPlus(_DimeNetPlusPlusBase):
    def __init__(self, target_columns=None, **kwargs):
        out_channels = int(kwargs.get(OUT_CHANNELS))
        energy_indices, oscillator_indices = _energy_oscillator_indices(target_columns, out_channels)
        num_blocks = int(kwargs['num_blocks'])
        num_radial = int(kwargs['num_radial'])
        hidden_channels = int(kwargs['hidden_channels'])
        out_emb_channels = int(kwargs['out_emb_channels'])
        num_output_layers = int(kwargs.get('num_output_layers', 3))
        output_initializer = str(kwargs.get('output_initializer', 'zeros'))
        act = activation_resolver(kwargs.get('act', 'swish'))
        super().__init__(**kwargs)
        self.register_buffer('energy_indices', torch.tensor(energy_indices, dtype=torch.long), persistent=False)
        self.register_buffer('oscillator_indices', torch.tensor(oscillator_indices, dtype=torch.long), persistent=False)
        self.target_dim = out_channels
        self.output_blocks = nn.ModuleList(
            [
                SplitOutputPPBlock(
                    num_radial=num_radial,
                    hidden_channels=hidden_channels,
                    out_emb_channels=out_emb_channels,
                    target_dim=out_channels,
                    energy_indices=energy_indices,
                    oscillator_indices=oscillator_indices,
                    num_layers=num_output_layers,
                    act=act,
                    output_initializer=output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )


class StateConditionedGeometryDimeNetPlusPlus(_DimeNetPlusPlusBase):
    """Geometry-only DimeNet++ that explicitly evaluates each state as E(R, s).

    A learned state identity is injected into the edge representation before the
    first interaction and after every interaction block. The geometry backbone
    remains shared across states, while every output block has separate energy
    and oscillator-strength projections.
    """

    def __init__(self, target_columns=None, **kwargs):
        target_columns = None if target_columns is None else tuple(str(column) for column in target_columns)
        layout = _state_target_layout(target_columns)
        target_dim = int(kwargs.get(OUT_CHANNELS))
        if target_dim != len(target_columns):
            raise ValueError('State-conditioned out_channels must match target_columns.')

        configured_num_states = kwargs.pop('num_states', None)
        if configured_num_states is not None and int(configured_num_states) != len(layout):
            raise ValueError('num_states must match the number of electronic states in target_columns.')

        num_blocks = int(kwargs['num_blocks'])
        num_radial = int(kwargs['num_radial'])
        hidden_channels = int(kwargs['hidden_channels'])
        out_emb_channels = int(kwargs['out_emb_channels'])
        num_output_layers = int(kwargs.get('num_output_layers', 3))
        output_initializer = str(kwargs.get('output_initializer', 'zeros'))
        act = activation_resolver(kwargs.get('act', 'swish'))

        base_kwargs = dict(kwargs)
        base_kwargs[OUT_CHANNELS] = 2
        super().__init__(**base_kwargs)

        self.target_dim = target_dim
        self.state_labels = tuple(state for state, _, _ in layout)
        self.energy_indices = tuple(energy_idx for _, energy_idx, _ in layout)
        self.oscillator_indices = tuple(oscillator_idx for _, _, oscillator_idx in layout)
        self.state_embedding = nn.Embedding(len(layout), hidden_channels)
        self.state_projections = nn.ModuleList(
            [Linear(hidden_channels, hidden_channels, bias=False) for _ in range(num_blocks + 1)]
        )
        self.output_blocks = nn.ModuleList(
            [
                SplitOutputPPBlock(
                    num_radial=num_radial,
                    hidden_channels=hidden_channels,
                    out_emb_channels=out_emb_channels,
                    target_dim=2,
                    energy_indices=[0],
                    oscillator_indices=[1],
                    num_layers=num_output_layers,
                    act=act,
                    output_initializer=output_initializer,
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self._reset_state_parameters()

    def _reset_state_parameters(self) -> None:
        nn.init.normal_(self.state_embedding.weight, mean=0.0, std=self.state_embedding.embedding_dim ** -0.5)
        for projection in self.state_projections:
            glorot_orthogonal(projection.weight, scale=2.0)

    def forward(self, z, pos, batch=None):
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=z.size(0))

        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        angle = torch.atan2(
            torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1),
            (pos_ij * pos_jk).sum(dim=-1),
        )
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        geometry_embedding = self.emb(z, rbf, i, j)

        predictions = [None] * self.target_dim
        for state_idx, (energy_idx, oscillator_idx) in enumerate(
            zip(self.energy_indices, self.oscillator_indices, strict=True)
        ):
            state_embedding = self.state_embedding.weight[state_idx]
            x = geometry_embedding + self.state_projections[0](state_embedding).unsqueeze(0)
            per_atom = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
            for block_idx, (interaction_block, output_block) in enumerate(
                zip(self.interaction_blocks, self.output_blocks[1:], strict=True),
                start=1,
            ):
                x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
                x = x + self.state_projections[block_idx](state_embedding).unsqueeze(0)
                per_atom = per_atom + output_block(x, rbf, i, num_nodes=pos.size(0))

            if batch is None:
                state_output = per_atom.sum(dim=0, keepdim=True)
            else:
                state_output = scatter(per_atom, batch, dim=0, reduce='sum')
            predictions[energy_idx] = state_output[:, 0]
            predictions[oscillator_idx] = state_output[:, 1]

        output = torch.stack(predictions, dim=-1)
        return output.squeeze(0) if batch is None else output


def build_dimenetpp(**kwargs):
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError() from _IMPORT_ERROR

    head_type = str(kwargs.pop(HEAD_TYPE, kwargs.pop(OUTPUT_HEAD, SINGLE)))
    target_columns_raw = kwargs.pop(TARGET_COLUMNS, None)
    if head_type in {SINGLE, STANDARD}:
        return DimeNetPlusPlus(**kwargs)
    if head_type in {SPLIT_ENERGY_OSCILLATOR, SPLIT_HEADS}:
        target_columns = None if target_columns_raw is None else tuple(str(column) for column in target_columns_raw)
        return SplitEnergyOscillatorDimeNetPlusPlus(target_columns=target_columns, **kwargs)
    if head_type in {SHARED_SPLIT_ENERGY_OSCILLATOR, SHARED_SPLIT_HEADS, SPLIT_OUTPUT_ENERGY_OSCILLATOR}:
        target_columns = None if target_columns_raw is None else tuple(str(column) for column in target_columns_raw)
        return SharedBackboneSplitEnergyOscillatorDimeNetPlusPlus(target_columns=target_columns, **kwargs)
    if head_type in {STATE_CONDITIONED_GEOMETRY, STATE_CONDITIONED}:
        target_columns = None if target_columns_raw is None else tuple(str(column) for column in target_columns_raw)
        return StateConditionedGeometryDimeNetPlusPlus(target_columns=target_columns, **kwargs)
    raise ValueError(f'Unsupported DimeNet++ head_type: {head_type}')
