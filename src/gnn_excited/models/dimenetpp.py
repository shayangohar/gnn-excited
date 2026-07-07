from __future__ import annotations

import torch
from torch import nn

try:
    from torch_geometric.nn.models import DimeNetPlusPlus
except ModuleNotFoundError as exc:  # pragma: no cover
    DimeNetPlusPlus = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

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


class SplitEnergyOscillatorDimeNetPlusPlus(nn.Module):
    def __init__(self, target_columns=None, **kwargs):
        super().__init__()
        out_channels = int(kwargs.get(OUT_CHANNELS))
        if target_columns is None:
            if out_channels % 2:
                raise ValueError()
            energy_indices = list(range(0, out_channels, 2))
            oscillator_indices = list(range(1, out_channels, 2))
        else:
            energy_indices = [idx for idx, column in enumerate(target_columns) if column.endswith(EV_SUFFIX)]
            oscillator_indices = [
                idx
                for idx, column in enumerate(target_columns)
                if column.startswith(LOG1P_PREFIX) and column.endswith(F_SUFFIX)
            ]
            if len(energy_indices) + len(oscillator_indices) != len(target_columns):
                raise ValueError()
        if not energy_indices or not oscillator_indices:
            raise ValueError()

        base_kwargs = dict(kwargs)
        base_kwargs.pop(OUT_CHANNELS, None)
        self.energy_indices = torch.tensor(energy_indices, dtype=torch.long)
        self.oscillator_indices = torch.tensor(oscillator_indices, dtype=torch.long)
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
    raise ValueError()
