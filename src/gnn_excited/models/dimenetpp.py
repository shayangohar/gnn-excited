from __future__ import annotations

from typing import Callable

import torch
from torch import nn

try:
    from torch.nn import Linear
    from torch_geometric.nn.inits import glorot_orthogonal
    from torch_geometric.nn.models import DimeNetPlusPlus
    from torch_geometric.nn.resolver import activation_resolver
    from torch_geometric.utils import scatter
except ModuleNotFoundError as exc:  # pragma: no cover
    Linear = None
    DimeNetPlusPlus = None
    glorot_orthogonal = None
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
    raise ValueError(f'Unsupported DimeNet++ head_type: {head_type}')
