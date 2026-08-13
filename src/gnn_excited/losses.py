"""Shared losses for paired excited-state energy/oscillator targets."""

from __future__ import annotations

import re
from typing import Sequence

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - optional ML dependency.
    torch = None
    F = None


def target_layout(target_columns: Sequence[str]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return energy and oscillator indices in ascending state order."""
    energy: list[tuple[int, int]] = []
    oscillator: list[tuple[int, int]] = []
    for index, column in enumerate(target_columns):
        text = str(column)
        match = re.search(r"(?:S|T)(\d+)", text, re.IGNORECASE)
        state = int(match.group(1)) if match else index
        if text.endswith("_eV"):
            energy.append((state, index))
        elif text.endswith("_f") or "osc" in text.lower() or "dip" in text.lower():
            oscillator.append((state, index))
        else:
            raise ValueError(f"Unsupported target column {column!r}; expected *_eV or oscillator/*_f")
    if not energy or not oscillator:
        raise ValueError("Targets must contain at least one energy and oscillator column")
    return tuple(index for _, index in sorted(energy)), tuple(index for _, index in sorted(oscillator))


def gap_ordering_loss(
    prediction,
    target,
    energy_indices: Sequence[int],
    *,
    gap_weight: float = 1.0,
    ordering_weight: float = 1.0,
    ordering_margin: float = 0.0,
):
    """Return adjacent-energy gap MSE and soft ascending-order penalty."""
    if len(energy_indices) < 2:
        zero = prediction.new_zeros(())
        return {"energy_gap_mse": zero, "energy_order_penalty": zero}
    pred_energy = prediction[:, list(energy_indices)]
    target_energy = target[:, list(energy_indices)]
    pred_gap = pred_energy[:, 1:] - pred_energy[:, :-1]
    target_gap = target_energy[:, 1:] - target_energy[:, :-1]
    gap = F.mse_loss(pred_gap, target_gap)
    # Penalize E_i >= E_{i+1}; margin allows a physically meaningful minimum gap.
    ordering = F.relu(pred_energy[:, :-1] - pred_energy[:, 1:] + float(ordering_margin)).pow(2).mean()
    return {
        "energy_gap_mse": gap * float(gap_weight),
        "energy_order_penalty": ordering * float(ordering_weight),
    }


def qm9gwbse_loss(
    prediction,
    target,
    target_columns: Sequence[str],
    *,
    energy_weight: float = 1.0,
    oscillator_weight: float = 1.0,
    gap_weight: float = 0.0,
    ordering_weight: float = 0.0,
    ordering_margin: float = 0.0,
    return_components: bool = False,
):
    """Direct paired-property MSE plus optional adjacent-gap/order terms."""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("qm9gwbse_loss requires torch")
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have the same shape (batch, targets)")
    energy_indices, oscillator_indices = target_layout(target_columns)
    direct_energy = F.mse_loss(prediction[:, list(energy_indices)], target[:, list(energy_indices)])
    direct_oscillator = F.mse_loss(prediction[:, list(oscillator_indices)], target[:, list(oscillator_indices)])
    gap_terms = gap_ordering_loss(
        prediction,
        target,
        energy_indices,
        gap_weight=gap_weight,
        ordering_weight=ordering_weight,
        ordering_margin=ordering_margin,
    )
    components = {
        "energy_mse": direct_energy,
        "oscillator_mse": direct_oscillator,
        **gap_terms,
    }
    total = (
        float(energy_weight) * direct_energy
        + float(oscillator_weight) * direct_oscillator
        + gap_terms["energy_gap_mse"]
        + gap_terms["energy_order_penalty"]
    )
    components["total"] = total
    return components if return_components else total
