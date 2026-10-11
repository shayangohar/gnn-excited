"""Excited-state regression metrics shared by train.py and the UMA finetune.

Kept free of torch_geometric/h5py imports so the fairchem environment can use
the exact same S-agg (energy_eV_mae) definition as the ViSNet pipeline.
"""

from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional ML dependency.
    torch = None

from gnn_excited.losses import same_spin_adjacent_pairs


def _physical_oscillator_column(column: str) -> str | None:
    if column.startswith('log1p_') and column.endswith('_f'):
        return column.removeprefix('log1p_')
    return None


def batch_mae(
    pred,
    target,
    target_columns: tuple[str, ...],
    include_physical_oscillator_metrics: bool = True,
    target_scale: float = 1.0,
) -> dict[str, float]:
    """Report MAEs, unscaling energy/gap metrics back to physical units.

    Some datasets train on scaled targets (e.g. OMol25 gaps divided by
    GAP_SCALE_EV); pass that divisor as ``target_scale`` so energy and
    adjacent-gap MAEs come out in true eV. Oscillator metrics are
    dimensionless/log-space and are never scaled; counts are unaffected.
    """
    scale = float(target_scale)
    metrics: dict[str, float] = {}
    energy_maes: list[float] = []
    log_osc_maes: list[float] = []
    osc_maes: list[float] = []
    for idx, column in enumerate(target_columns):
        mae = (pred[:, idx] - target[:, idx]).abs().mean().item()
        if column.endswith('_eV'):
            mae = mae * scale
            energy_maes.append(mae)
        metrics[f'{column}_mae'] = mae
        physical_osc_column = _physical_oscillator_column(column)
        if physical_osc_column is not None:
            log_osc_maes.append(mae)
            if include_physical_oscillator_metrics:
                osc_mae = (
                    torch.expm1(pred[:, idx].double()).clamp_min(0)
                    - torch.expm1(target[:, idx].double())
                ).abs().mean().item()
                metrics[f'{physical_osc_column}_mae'] = osc_mae
                osc_maes.append(osc_mae)
    if energy_maes:
        metrics['energy_eV_mae'] = sum(energy_maes) / len(energy_maes)
    if log_osc_maes:
        metrics['log1p_oscillator_strength_mae'] = sum(log_osc_maes) / len(log_osc_maes)
    if osc_maes:
        metrics['oscillator_strength_mae'] = sum(osc_maes) / len(osc_maes)
    energy_indices = [index for index, column in enumerate(target_columns) if column.endswith('_eV')]
    if len(energy_indices) > 1:
        pred_energy = pred[:, energy_indices]
        target_energy = target[:, energy_indices]
        pred_gaps = pred_energy[:, 1:] - pred_energy[:, :-1]
        target_gaps = target_energy[:, 1:] - target_energy[:, :-1]
        gap_errors = (pred_gaps - target_gaps).abs() * scale
        metrics['adjacent_gap_eV_mae'] = gap_errors.mean().item()
        spin_pairs = same_spin_adjacent_pairs(
            [target_columns[index] for index in energy_indices]
        )
        if spin_pairs:
            metrics['ordering_violation_count'] = float(
                sum(int((pred_gaps[:, position] <= 0).sum().item()) for position, _ in spin_pairs)
            )
            metrics['ordering_comparison_count'] = float(
                len(spin_pairs) * pred_gaps.size(0)
            )
        else:
            metrics['ordering_violation_count'] = 0.0
            metrics['ordering_comparison_count'] = 0.0
        for gap_index, (left_index, right_index) in enumerate(zip(energy_indices[:-1], energy_indices[1:])):
            left = target_columns[left_index].removesuffix('_eV')
            right = target_columns[right_index].removesuffix('_eV')
            metrics[f'{left}_{right}_gap_eV_mae'] = gap_errors[:, gap_index].mean().item()
    return metrics
