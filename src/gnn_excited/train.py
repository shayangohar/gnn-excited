from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    DataLoader = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from gnn_excited.data.pyg_dataset import QCDGES1Dataset, deterministic_split, explicit_split
from gnn_excited.data.qm9gwbse import QM9GWBSEDataset, electronic_descriptor_keys
from gnn_excited.data.omol25 import GAP_SCALE_EV, Omol25GapDataset
from gnn_excited.models.dimenetpp import build_dimenetpp
from gnn_excited.models.visnet import build_visnet, load_transfer_checkpoint
from gnn_excited.losses import (
    phase_invariant_vector_squared_error,
    qm9gwbse_loss,
    same_spin_adjacent_pairs,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def seed_everything(seed: int, deterministic_algorithms: bool = False) -> dict[str, Any]:
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms), warn_only=True)
    return {
        'seed': seed,
        'python_hash_seed': os.environ['PYTHONHASHSEED'],
        'deterministic_algorithms': bool(deterministic_algorithms),
        'cudnn_deterministic': bool(getattr(torch.backends.cudnn, 'deterministic', False)),
        'cudnn_benchmark': bool(getattr(torch.backends.cudnn, 'benchmark', False)),
    }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open('r', newline='', encoding='utf-8') as stream:
        return [row for row in csv.DictReader(stream) if row.get('status') == 'ok']


def filter_manifest_exclusions(
    rows: list[dict[str, str]],
    exclusion_path: str | Path | None,
) -> tuple[list[dict[str, str]], int]:
    """Remove manifest rows named by an explicit QCDGE exclusion CSV."""
    if not exclusion_path:
        return rows, 0
    path = Path(exclusion_path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        key_column = "qcdge_molecule_key" if "qcdge_molecule_key" in fieldnames else "molecule_key" if "molecule_key" in fieldnames else None
        if key_column is None:
            raise ValueError(f"Exclusion file {path} must contain qcdge_molecule_key or molecule_key")
        excluded = {(row.get(key_column) or "").strip() for row in reader}
    excluded.discard("")
    filtered = [row for row in rows if row.get("molecule_key", "").strip() not in excluded]
    return filtered, len(rows) - len(filtered)


def _subset_keys(rows: list[dict[str, str]], indices: list[int]) -> list[str]:
    return [rows[i]['molecule_key'] for i in indices]


def _target_columns_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    targets_cfg = config.get('targets') or {}
    columns = targets_cfg.get('columns')
    if columns:
        return tuple(str(column) for column in columns)
    energy = targets_cfg.get('energy', 'S1_eV')
    oscillator = targets_cfg.get('oscillator', 'log1p_S1_f')
    return (str(energy), str(oscillator))


def build_loss_weights(config: dict[str, Any], target_columns: tuple[str, ...]) -> tuple[float, ...] | None:
    loss_cfg = config.get('loss') or {}
    if not loss_cfg:
        return None

    loss_type = str(loss_cfg.get('type', 'mse'))
    supported_loss_types = {'mse', 'mean_mse', 'weighted_mse', 'qm9gwbse'}
    if loss_type == 'qm9gwbse':
        return None
    if loss_type not in supported_loss_types:
        raise ValueError(f'Unsupported loss type: {loss_type}')

    weights_cfg = loss_cfg.get('weights')
    if isinstance(weights_cfg, list):
        if len(weights_cfg) != len(target_columns):
            raise ValueError('loss.weights list length must match target column count')
        weights = [float(weight) for weight in weights_cfg]
    elif weights_cfg is None or isinstance(weights_cfg, dict):
        energy_weight = float(loss_cfg.get('energy_weight', 1.0))
        oscillator_weight = float(loss_cfg.get('oscillator_weight', 1.0))
        weights = []
        for column in target_columns:
            if column.endswith('_eV'):
                weight = energy_weight
            elif _physical_oscillator_column(column) is not None:
                weight = oscillator_weight
            else:
                weight = 1.0
            if isinstance(weights_cfg, dict) and column in weights_cfg:
                weight = float(weights_cfg[column])
            weights.append(weight)
    else:
        raise TypeError('loss.weights must be a mapping, list, or omitted')

    if any(weight <= 0 for weight in weights):
        raise ValueError('loss weights must be positive')

    if loss_type != 'weighted_mse' and all(math.isclose(weight, 1.0) for weight in weights):
        return None
    if loss_type != 'weighted_mse':
        raise ValueError('non-unit loss weights require loss.type: weighted_mse')
    return tuple(weights)


def _normalize_loss_weights(config: dict[str, Any]) -> bool:
    loss_cfg = config.get('loss') or {}
    return bool(loss_cfg.get('normalize', True))


def _training_loss(
    pred,
    target,
    target_columns,
    config,
    loss_weights=None,
    normalize=True,
    predicted_dipoles=None,
    target_dipoles=None,
):
    loss_cfg = config.get('loss') or {}
    if str(loss_cfg.get('type', 'mse')) == 'qm9gwbse':
        return qm9gwbse_loss(
            pred, target, target_columns,
            energy_weight=float(loss_cfg.get('energy_weight', 1.0)),
            oscillator_weight=float(loss_cfg.get('oscillator_weight', 1.0)),
            gap_weight=float(loss_cfg.get('gap_weight', 0.0)),
            ordering_weight=float(loss_cfg.get('ordering_weight', 0.0)),
            ordering_margin=float(loss_cfg.get('ordering_margin', 0.0)),
            predicted_dipoles=predicted_dipoles,
            target_dipoles=target_dipoles,
            dipole_weight=float(loss_cfg.get('dipole_weight', 0.0)),
        )
    return weighted_mse_loss(pred, target, loss_weights, normalize)


def _forward_model(model, batch, target_dim: int):
    descriptors = getattr(batch, "electronic_descriptors", None)
    if descriptors is not None and hasattr(model, "descriptor_dim"):
        output = model(
            batch.z,
            batch.pos,
            batch.batch,
            electronic_descriptors=descriptors,
        )
    else:
        output = model(batch.z, batch.pos, batch.batch)
    if isinstance(output, tuple):
        prediction, transition_dipoles = output
        transition_dipoles = transition_dipoles.view(prediction.shape[0], -1, 3)
    else:
        prediction, transition_dipoles = output, None
    return prediction.view(-1, target_dim), transition_dipoles


def weighted_mse_loss(pred, target, loss_weights=None, normalize: bool = True):
    if loss_weights is None:
        return torch.nn.functional.mse_loss(pred, target)

    weights = torch.as_tensor(loss_weights, dtype=pred.dtype, device=pred.device).view(1, -1)
    if weights.shape[1] != pred.shape[1]:
        raise ValueError('loss weight count must match prediction dimension')

    squared_error = (pred - target).pow(2) * weights
    if normalize:
        denominator = pred.shape[0] * weights.sum().clamp_min(torch.finfo(pred.dtype).eps)
        return squared_error.sum() / denominator
    return squared_error.mean()


def _require_finite(value, name: str, batch=None) -> None:
    if torch.isfinite(value).all():
        return
    keys = getattr(batch, 'molecule_key', None)
    key_text = f'; molecule_keys={list(keys) if isinstance(keys, (list, tuple)) else keys}' if keys is not None else ''
    raise FloatingPointError(f'Non-finite {name}{key_text}')


def _source_subset_from_key(molecule_key: str) -> str:
    prefix = ''.join(ch for ch in str(molecule_key) if ch.isalpha())
    return {
        'Aa': 'A_9',
        'Ab': 'A_10',
        'Ba': 'B_9',
        'Bb': 'B_10',
    }.get(prefix, prefix or 'unknown')


def _subset_key_groups(rows: list[dict[str, str]], indices: list[int]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for index in indices:
        key = rows[index]['molecule_key']
        groups.setdefault(_source_subset_from_key(key), []).append(key)
    return dict(sorted(groups.items()))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_metrics_csv_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix('.metrics.csv')


def _default_summary_json_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix('.summary.json')


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _run_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_run_metadata(device: str) -> dict[str, Any]:
    git_status = _run_command(['git', 'status', '--short'])
    metadata: dict[str, Any] = {
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'git_commit': _run_command(['git', 'rev-parse', 'HEAD']),
        'git_dirty': bool(git_status),
        'git_status_short': git_status or '',
        'slurm': {
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'job_name': os.environ.get('SLURM_JOB_NAME'),
            'partition': os.environ.get('SLURM_JOB_PARTITION'),
            'node_list': os.environ.get('SLURM_JOB_NODELIST'),
            'submit_dir': os.environ.get('SLURM_SUBMIT_DIR'),
        },
    }
    if torch is not None:
        metadata['torch_version'] = torch.__version__
        metadata['cuda_available'] = torch.cuda.is_available()
        metadata['cuda_device_count'] = torch.cuda.device_count()
        metadata['cuda_version'] = torch.version.cuda
        metadata['cudnn_version'] = torch.backends.cudnn.version()
        metadata['cuda_device_name'] = (
            torch.cuda.get_device_name(0) if device == 'cuda' and torch.cuda.is_available() else None
        )
    try:
        import torch_geometric
    except ModuleNotFoundError:
        metadata['torch_geometric_version'] = None
    else:
        metadata['torch_geometric_version'] = torch_geometric.__version__
    return metadata


def _copy_hdf5_to_local_scratch(hdf5_path: str | Path, dataset_cfg: dict[str, Any]) -> Path:
    source = Path(hdf5_path)
    if not bool(dataset_cfg.get('local_copy', False)):
        return source

    configured_root = dataset_cfg.get('local_copy_dir')
    scratch_root = configured_root or os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR')
    if scratch_root is None:
        user = os.environ.get('USER', 'unknown')
        job_id = os.environ.get('SLURM_JOB_ID', 'manual')
        scratch_root = f'/tmp/{user}/gnn-excited-{job_id}'
    scratch_root = os.path.expandvars(str(scratch_root))
    target_dir = Path(scratch_root) / 'gnn_excited_data'
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name

    if target.exists() and target.stat().st_size == source.stat().st_size:
        print(f'Using existing local HDF5 copy: {target}', flush=True)
        return target

    tmp_target = target.with_name(target.name + '.tmp')
    if tmp_target.exists():
        tmp_target.unlink()
    print(f'Copying HDF5 to local scratch: {source} -> {target}', flush=True)
    copy_started = time.perf_counter()
    shutil.copy2(source, tmp_target)
    tmp_target.replace(target)
    copy_seconds = time.perf_counter() - copy_started
    print(f'Finished local HDF5 copy in {copy_seconds:.1f}s', flush=True)
    return target


def _dataloader_kwargs(train_cfg: dict[str, Any], device: str) -> dict[str, Any]:
    dataloader_cfg = train_cfg.get('dataloader') or {}
    num_workers = int(dataloader_cfg.get('num_workers', 0))
    kwargs: dict[str, Any] = {
        'num_workers': num_workers,
        'pin_memory': bool(dataloader_cfg.get('pin_memory', device == 'cuda')),
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(dataloader_cfg.get('persistent_workers', True))
        if 'prefetch_factor' in dataloader_cfg:
            kwargs['prefetch_factor'] = int(dataloader_cfg['prefetch_factor'])
    return kwargs


def _make_loader(dataset, train_cfg: dict[str, Any], device: str, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = _dataloader_kwargs(train_cfg, device)
    if int(kwargs['num_workers']) > 0:
        kwargs['worker_init_fn'] = seed_worker
    return DataLoader(
        dataset,
        batch_size=int(train_cfg['batch_size']),
        shuffle=shuffle,
        generator=generator,
        **kwargs,
    )


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_summary_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as stream:
        json.dump(_json_ready(payload), stream, indent=2, sort_keys=True)
        stream.write('\n')


class WandbRun:
    def __init__(self, config: dict[str, Any], run_summary: dict[str, Any]) -> None:
        self._run = None
        self._wandb = None
        wandb_cfg = config.get('wandb') or {}
        if not wandb_cfg.get('enabled', False):
            return
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise RuntimeError('Config enabled W&B logging, but wandb is not installed.') from exc

        self._wandb = wandb
        init_kwargs = {
            'project': wandb_cfg.get('project', 'gnn-excited'),
            'entity': wandb_cfg.get('entity'),
            'name': wandb_cfg.get('name'),
            'group': wandb_cfg.get('group'),
            'tags': wandb_cfg.get('tags'),
            'mode': wandb_cfg.get('mode'),
            'job_type': wandb_cfg.get('job_type', 'train'),
            'config': _json_ready(
                {
                    'config': config,
                    'run_summary': {
                        key: value
                        for key, value in run_summary.items()
                        if key not in {'latest_metrics', 'test_metrics'}
                    },
                }
            ),
        }
        self._run = wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def metadata(self) -> dict[str, str] | None:
        if self._run is None:
            return None
        return {'id': self._run.id, 'name': self._run.name, 'url': self._run.url}

    def log_epoch(self, record: dict[str, Any]) -> None:
        if self._wandb is None:
            return
        self._wandb.log(record, step=int(record['epoch']))

    def log_test_metrics(self, metrics: dict[str, float], best_epoch: int | None) -> None:
        if self._wandb is None:
            return
        payload = {f'test_{key}': value for key, value in metrics.items()}
        if best_epoch is not None:
            payload['best_epoch'] = best_epoch
        self._wandb.log(payload)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._wandb is None:
            return
        self._wandb.log(metrics)

    def finish(self, exit_code: int = 0) -> None:
        if self._wandb is not None:
            self._wandb.finish(exit_code=exit_code)


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


def evaluate_ensemble(
    model,
    checkpoint_paths,
    loader,
    device: str,
    target_columns,
    config=None,
    target_scale: float = 1.0,
):
    """Compute metrics on the AVERAGED predictions of N checkpoints.

    Members are loaded sequentially into the same model instance; loaders must
    be deterministic (shuffle=False). Per-molecule predictions are concatenated
    within each member and averaged across members before metrics. Only energy
    columns participate in aggregate/ordering metrics; within-spin adjacency
    rules apply. ``target_scale`` converts scaled training targets (e.g.
    OMol25 gaps divided by GAP_SCALE_EV) back to physical units for energy
    and gap MAE reporting; training itself is unaffected.
    """
    scale = float(target_scale)
    target_dim = len(target_columns)
    energy_positions = [
        index for index, column in enumerate(target_columns)
        if str(column).endswith('_eV')
    ]
    spin_pairs = same_spin_adjacent_pairs(
        [target_columns[index] for index in energy_positions]
    )
    member_predictions = []
    target_store = []
    member_aggregates = []
    with torch.no_grad():
        for member_index, path in enumerate(checkpoint_paths):
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state)
            model.eval()
            rows = []
            energy_abs = 0.0
            count = 0
            for batch in loader:
                batch = batch.to(device)
                target = batch.y.view(-1, target_dim)
                prediction, _ = _forward_model(model, batch, target_dim)
                prediction = prediction.detach().float()
                rows.append(prediction)
                energy_abs += float(
                    (
                        prediction[:, energy_positions] - target[:, energy_positions]
                    ).abs().sum().item()
                )
                count += target.shape[0]
                if member_index == 0:
                    target_store.append(target.detach().cpu())
            member_aggregates.append(
                scale * energy_abs / max(count * max(len(energy_positions), 1), 1)
            )
            member_predictions.append(torch.cat(rows, dim=0))
    mean_prediction = torch.stack(member_predictions, dim=0).mean(dim=0)
    target_all = torch.cat(target_store, dim=0).to(mean_prediction.device)

    metrics: dict[str, float] = {}
    for index, column in enumerate(target_columns):
        column_mae = (
            (mean_prediction[:, index] - target_all[:, index]).abs().mean().item()
        )
        if str(column).endswith('_eV'):
            column_mae *= scale
        metrics[f'{column}_mae'] = column_mae
    pred_energy = mean_prediction[:, energy_positions]
    target_energy = target_all[:, energy_positions]
    metrics['energy_eV_mae'] = scale * (pred_energy - target_energy).abs().mean().item()
    prefixes = sorted({str(target_columns[i])[:1] for i in energy_positions})
    for prefix in prefixes:
        positions = [
            p for p in energy_positions if str(target_columns[p]).startswith(prefix)
        ]
        if len(positions) > 1:
            metrics[f'{prefix}manifold_eV_mae'] = (
                scale
                * (mean_prediction[:, positions] - target_all[:, positions])
                .abs()
                .mean()
                .item()
            )
    if len(energy_positions) > 1:
        pred_gaps = pred_energy[:, 1:] - pred_energy[:, :-1]
        target_gaps = target_energy[:, 1:] - target_energy[:, :-1]
        gap_errors = (pred_gaps - target_gaps).abs() * scale
        metrics['adjacent_gap_eV_mae'] = gap_errors.mean().item()
        for gap_index, (left, right) in enumerate(
            zip(energy_positions[:-1], energy_positions[1:])
        ):
            left_name = str(target_columns[left]).removesuffix('_eV')
            right_name = str(target_columns[right]).removesuffix('_eV')
            metrics[f'{left_name}_{right_name}_gap_eV_mae'] = (
                gap_errors[:, gap_index].mean().item()
            )
        if spin_pairs:
            metrics['ordering_violation_count'] = float(
                sum(int((pred_gaps[:, position] <= 0).sum().item()) for position, _ in spin_pairs)
            )
            metrics['ordering_comparison_count'] = float(
                len(spin_pairs) * pred_gaps.size(0)
            )
            metrics['ordering_violation_rate'] = (
                metrics['ordering_violation_count']
                / metrics['ordering_comparison_count']
            )
    for member_index, value in enumerate(member_aggregates):
        metrics[f'member_{member_index}_energy_eV_mae'] = value
    return metrics


def evaluate(
    model,
    loader,
    device: str,
    target_columns: tuple[str, ...],
    loss_weights=None,
    normalize_loss_weights: bool = True,
    config: dict[str, Any] | None = None,
    predictions_csv_path: str | Path | None = None,
    target_scale: float = 1.0,
) -> dict[str, float]:
    model.eval()
    scale = float(target_scale)
    totals: dict[str, float] = {'loss': 0.0, 'n': 0}
    count_metrics = {'ordering_violation_count', 'ordering_comparison_count'}
    target_dim = len(target_columns)
    prediction_rows: list[dict[str, Any]] = []
    energy_errors: list[float] = []
    log_oscillator_errors: list[float] = []
    oscillator_errors: list[float] = []
    oscillator_indices = [
        index for index, column in enumerate(target_columns)
        if _physical_oscillator_column(column) is not None
    ]
    evaluation_cfg = (config or {}).get('evaluation') or {}
    include_physical_oscillator_metrics = bool(
        evaluation_cfg.get('physical_oscillator_metrics', True)
    )
    oscillator_inverse_overflow_count = 0
    max_float64_log = math.log(torch.finfo(torch.float64).max)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.y.view(-1, target_dim)
            _require_finite(target, 'validation targets', batch)
            pred, predicted_dipoles = _forward_model(model, batch, target_dim)
            target_dipoles = (
                batch.transition_dipole.view_as(predicted_dipoles)
                if predicted_dipoles is not None
                else None
            )
            try:
                _require_finite(pred, 'validation predictions', batch)
            except FloatingPointError:
                if bool((config or {}).get('training', {}).get('skip_nonfinite_batches', False)):
                    continue
                raise
            if predicted_dipoles is not None:
                try:
                    _require_finite(predicted_dipoles, 'validation transition dipoles', batch)
                except FloatingPointError:
                    if bool((config or {}).get('training', {}).get('skip_nonfinite_batches', False)):
                        continue
                    raise
            if oscillator_indices:
                oscillator_pred = pred[:, oscillator_indices].double()
                oscillator_inverse_overflow_count += int(
                    (oscillator_pred > max_float64_log).sum().item()
                )
                if include_physical_oscillator_metrics:
                    _require_finite(
                        torch.expm1(oscillator_pred),
                        'physical oscillator predictions',
                        batch,
                    )
            loss = _training_loss(
                pred,
                target,
                target_columns,
                config or {},
                loss_weights,
                normalize_loss_weights,
                predicted_dipoles,
                target_dipoles,
            )
            try:
                _require_finite(loss, 'validation loss', batch)
            except FloatingPointError:
                if bool((config or {}).get('training', {}).get('skip_nonfinite_batches', False)):
                    continue
                raise
            metrics = batch_mae(
                pred,
                target,
                target_columns,
                include_physical_oscillator_metrics,
                target_scale=scale,
            )
            if predicted_dipoles is not None:
                dipole_squared_error = phase_invariant_vector_squared_error(
                    predicted_dipoles, target_dipoles
                )
                metrics['transition_dipole_phase_invariant_mae_D'] = (
                    dipole_squared_error.sqrt().mean().item()
                )
                metrics['transition_dipole_magnitude_mae_D'] = (
                    predicted_dipoles.norm(dim=-1) - target_dipoles.norm(dim=-1)
                ).abs().mean().item()
            batch_n = target.shape[0]
            totals['loss'] += loss.item() * batch_n
            if loss_weights is not None:
                unweighted_loss = weighted_mse_loss(pred, target)
                totals['unweighted_mse_loss'] = totals.get('unweighted_mse_loss', 0.0) + unweighted_loss.item() * batch_n
            for key, value in metrics.items():
                multiplier = 1 if key in count_metrics else batch_n
                totals[key] = totals.get(key, 0.0) + value * multiplier
            totals['n'] += batch_n
            if predictions_csv_path is not None:
                pred_cpu = pred.detach().cpu()
                target_cpu = target.detach().cpu()
                raw_keys = getattr(batch, 'molecule_key', range(batch_n))
                molecule_keys = [raw_keys] if isinstance(raw_keys, str) else list(raw_keys)
                for sample_index, molecule_key in enumerate(molecule_keys):
                    row: dict[str, Any] = {'molecule_key': molecule_key}
                    for target_index, column in enumerate(target_columns):
                        prediction = float(pred_cpu[sample_index, target_index])
                        expected = float(target_cpu[sample_index, target_index])
                        if column.endswith('_eV'):
                            prediction *= scale
                            expected *= scale
                        error = abs(prediction - expected)
                        row[f'target_{column}'] = expected
                        row[f'prediction_{column}'] = prediction
                        row[f'abs_error_{column}'] = error
                        if column.endswith('_eV'):
                            energy_errors.append(error)
                        elif _physical_oscillator_column(column) is not None:
                            log_oscillator_errors.append(error)
                            if include_physical_oscillator_metrics:
                                oscillator_errors.append(abs(math.expm1(prediction) - math.expm1(expected)))
                    prediction_rows.append(row)
    n = max(totals.pop('n'), 1)
    metrics = {key: value if key in count_metrics else value / n for key, value in totals.items()}
    comparisons = metrics.get('ordering_comparison_count', 0.0)
    if comparisons:
        metrics['ordering_violation_rate'] = metrics['ordering_violation_count'] / comparisons
    metrics['oscillator_strength_inverse_overflow_count'] = float(
        oscillator_inverse_overflow_count
    )
    if predictions_csv_path is not None:
        write_history_csv(predictions_csv_path, prediction_rows)
        for name, errors in (
            ('energy_eV_abs_error', energy_errors),
            ('log1p_oscillator_strength_abs_error', log_oscillator_errors),
            ('oscillator_strength_abs_error', oscillator_errors),
        ):
            if errors:
                values = np.asarray(errors, dtype=np.float64)
                metrics[f'{name}_max'] = float(values.max())
                metrics[f'{name}_p95'] = float(np.quantile(values, 0.95))
                metrics[f'{name}_p99'] = float(np.quantile(values, 0.99))
    return metrics

def evaluate_denoising(
    model,
    loader,
    device: str,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Denoising-pretraining validation: per-atom displacement MSE against fresh noise."""
    model.eval()
    totals: dict[str, float] = {'loss': 0.0, 'n': 0}
    train_cfg = (config or {}).get('training') or {}
    sigma_min = float(train_cfg.get('denoise_sigma', 0.1))
    sigma_max = float(train_cfg.get('denoise_sigma_max', 0.6))
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(
                1, device=batch.pos.device
            )
            noise = torch.randn_like(batch.pos) * sigma
            denoised_pos = batch.pos + noise
            disp = model.forward_denoise(batch.z, denoised_pos, batch.batch)
            loss = weighted_mse_loss(disp, -noise)
            _require_finite(loss, 'validation denoise loss', batch)
            n = batch.pos.size(0)
            totals['loss'] += loss.item() * n
            totals['n'] += n
    n = max(totals.pop('n'), 1)
    return {key: value / n for key, value in totals.items()}


def build_scheduler(optimizer, scheduler_cfg: dict[str, Any] | None, total_epochs: int | None = None):
    if not scheduler_cfg:
        return None
    scheduler_type = scheduler_cfg.get('type')
    if scheduler_type in (None, 'none'):
        return None
    if scheduler_type == 'warmup_cosine':
        if total_epochs is None:
            raise ValueError('warmup_cosine scheduler requires total_epochs')
        warmup_epochs = int(scheduler_cfg.get('warmup_epochs', 5))
        if not 0 < warmup_epochs < total_epochs:
            raise ValueError('warmup_epochs must be between 1 and total_epochs - 1')
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(scheduler_cfg.get('start_factor', 1.0 / 300.0)),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=float(scheduler_cfg.get('min_lr', 1e-5)),
        )
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [warmup_epochs])
    if scheduler_type != 'reduce_on_plateau':
        raise ValueError(f'Unsupported scheduler type: {scheduler_type}')
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_cfg.get('mode', 'min')),
        factor=float(scheduler_cfg.get('factor', 0.5)),
        patience=int(scheduler_cfg.get('patience', 8)),
        min_lr=float(scheduler_cfg.get('min_lr', 1e-6)),
    )


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]['lr'])


def classify_validation_improvement(val_loss: float, best_val: float, early_stopping_best_val: float, min_delta: float):
    '''Separate checkpoint-saving improvement from early-stopping improvement.'''
    return {
        'checkpoint_improved': val_loss < best_val,
        'early_stopping_improved': val_loss < early_stopping_best_val - min_delta,
    }


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError('Training requires torch and torch_geometric.') from _IMPORT_ERROR

    config_path = Path(config_path)
    config = load_config(config_path)
    dataset_cfg = config['dataset']
    train_cfg = config['training']
    model_kwargs = dict(config['model'])
    model_type = str(model_kwargs.pop('type', 'dimenet')).lower()
    target_columns = _target_columns_from_config(config)
    configured_out_channels = int(model_kwargs.get('out_channels', len(target_columns)))
    if configured_out_channels != len(target_columns):
        raise ValueError(
            'model.out_channels={} does not match {} targets'.format(configured_out_channels, len(target_columns))
        )
    model_kwargs['out_channels'] = len(target_columns)
    model_kwargs['target_columns'] = target_columns
    loss_weights = build_loss_weights(config, target_columns)
    normalize_loss_weights = _normalize_loss_weights(config)
    loss_weights_tensor = None if loss_weights is None else torch.tensor(loss_weights, dtype=torch.float32)
    device = train_cfg.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Config requested CUDA, but torch.cuda.is_available() is False')

    training_seed = int(train_cfg.get('seed', dataset_cfg.get('split_seed', 0)))
    reproducibility = seed_everything(
        training_seed,
        deterministic_algorithms=bool(train_cfg.get('deterministic_algorithms', False)),
    )

    checkpoint_path = Path(train_cfg['checkpoint_path'])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = Path(train_cfg.get('metrics_csv_path') or _default_metrics_csv_path(checkpoint_path))
    summary_json_path = Path(train_cfg.get('summary_json_path') or _default_summary_json_path(checkpoint_path))

    effective_hdf5_path = _copy_hdf5_to_local_scratch(dataset_cfg['hdf5_path'], dataset_cfg)
    rows = _load_manifest_rows(dataset_cfg['manifest_path'])
    manifest_ok_rows = len(rows)
    electronic_descriptors_path = dataset_cfg.get('electronic_descriptors_path')
    descriptor_rows_missing = 0
    if electronic_descriptors_path:
        available_descriptor_keys = electronic_descriptor_keys(electronic_descriptors_path)
        descriptor_rows_missing = sum(
            row['molecule_key'] not in available_descriptor_keys for row in rows
        )
        rows = [row for row in rows if row['molecule_key'] in available_descriptor_keys]
    exclude_keys_path = dataset_cfg.get('exclude_keys_path')
    rows, excluded_manifest_rows = filter_manifest_exclusions(rows, exclude_keys_path)
    max_rows = dataset_cfg.get('max_manifest_molecules')
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    split_path = dataset_cfg.get('split_path')
    if split_path:
        split_column = str(dataset_cfg.get('split_column', 'random_split'))
        train_idx, val_idx, test_idx = explicit_split(rows, split_path, split_column)
        split_metadata = {
            'type': 'explicit',
            'path': str(split_path),
            'column': split_column,
        }
    else:
        train_idx, val_idx, test_idx = deterministic_split(
            rows,
            seed=int(dataset_cfg['split_seed']),
            train_fraction=float(dataset_cfg['train_fraction']),
            val_fraction=float(dataset_cfg['val_fraction']),
        )
        split_metadata = {
            'type': 'deterministic_random',
            'seed': int(dataset_cfg['split_seed']),
            'train_fraction': float(dataset_cfg['train_fraction']),
            'val_fraction': float(dataset_cfg['val_fraction']),
        }
    test_subset_key_groups = _subset_key_groups(rows, test_idx)
    dataset_type = str(dataset_cfg.get('type', 'qcdge')).lower()
    # OMol25 targets are stored divided by GAP_SCALE_EV for training stability;
    # report all energy/gap MAEs back in true eV (other datasets: scale 1.0).
    target_scale = float(GAP_SCALE_EV) if dataset_type == 'omol25' else 1.0
    if dataset_type == 'omol25':
        dataset_class = Omol25GapDataset
    elif dataset_type in {'qm9gwbse', 'qm9-gwbse'}:
        dataset_class = QM9GWBSEDataset
    else:
        dataset_class = QCDGES1Dataset
    def make_dataset(indices):
        args = (
            effective_hdf5_path,
            dataset_cfg['manifest_path'],
            _subset_keys(rows, indices),
            target_columns,
        )
        if dataset_class is QM9GWBSEDataset:
            return dataset_class(
                *args,
                electronic_descriptors_path=electronic_descriptors_path,
                electronic_descriptor_features=dataset_cfg.get(
                    'electronic_descriptor_features'
                ),
            )
        return dataset_class(*args)

    train_ds = make_dataset(train_idx)
    val_ds = make_dataset(val_idx)
    test_ds = make_dataset(test_idx)

    train_loader = _make_loader(train_ds, train_cfg, device, shuffle=True, seed=training_seed)
    val_loader = _make_loader(val_ds, train_cfg, device, shuffle=False, seed=training_seed + 1)
    test_loader = _make_loader(test_ds, train_cfg, device, shuffle=False, seed=training_seed + 2)

    model_builder = build_visnet if model_type in {'visnet', 'visnet_one_pass'} else build_dimenetpp
    model_kwargs['denoising'] = bool(train_cfg.get('denoising', False))
    model = model_builder(**model_kwargs).to(device)
    evaluation_cfg = config.get('evaluation') or {}
    evaluation_checkpoint = evaluation_cfg.get('checkpoint_path')
    evaluation_checkpoint_paths = list(evaluation_cfg.get('checkpoint_paths') or [])
    if evaluation_checkpoint and evaluation_checkpoint_paths:
        raise ValueError(
            'evaluation.checkpoint_path and evaluation.checkpoint_paths are mutually exclusive'
        )
    evaluation_mode = bool(evaluation_checkpoint) or bool(evaluation_checkpoint_paths)
    transfer_cfg = config.get('transfer') or train_cfg.get('transfer') or {}
    transfer_checkpoint = transfer_cfg.get('checkpoint_path')
    if evaluation_checkpoint and transfer_checkpoint:
        raise ValueError('evaluation.checkpoint_path and transfer.checkpoint_path are mutually exclusive')
    if evaluation_checkpoint_paths and transfer_checkpoint:
        raise ValueError('evaluation.checkpoint_paths and transfer.checkpoint_path are mutually exclusive')
    if transfer_checkpoint:
        transfer_mode = str(transfer_cfg.get('mode', 'readout_only'))
        if model_type not in {'visnet', 'visnet_one_pass'}:
            raise ValueError('Transfer checkpoint loading is currently implemented for ViSNet; DimeNet++ cross-architecture loading is intentionally unsupported.')
        load_transfer_checkpoint(model, transfer_checkpoint, mode=transfer_mode, map_location=device)
    evaluated_checkpoint = None
    if evaluation_checkpoint:
        if model_type not in {'visnet', 'visnet_one_pass'}:
            raise ValueError('Checkpoint-only evaluation is currently implemented for ViSNet')
        evaluated_checkpoint = torch.load(evaluation_checkpoint, map_location=device)
        load_transfer_checkpoint(model, evaluated_checkpoint, mode='full_finetune', map_location=device)
    if loss_weights_tensor is not None:
        loss_weights_tensor = loss_weights_tensor.to(device)
    epochs = int(train_cfg['epochs'])
    if not evaluation_mode and epochs < 1:
        raise ValueError('training.epochs must be positive unless evaluation.checkpoint_path is set')
    optimizer = None
    scheduler = None
    if not evaluation_mode:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg['learning_rate']),
            weight_decay=float(train_cfg.get('weight_decay', 0.0)),
        )
        scheduler = build_scheduler(optimizer, train_cfg.get('scheduler'), epochs)
    history: list[dict[str, Any]] = list((evaluated_checkpoint or {}).get('history') or [])
    best_val = math.inf
    best_epoch: int | None = int(history[-1]['epoch']) if history and 'epoch' in history[-1] else None
    historical_val_losses = [float(record['val_loss']) for record in history if 'val_loss' in record]
    if historical_val_losses:
        best_val = min(historical_val_losses)
    early_stopping_best_val = math.inf
    min_delta = float(train_cfg.get('early_stopping_min_delta', 0.0))
    early_stopping_patience = train_cfg.get('early_stopping_patience')
    early_stopping_patience = int(early_stopping_patience) if early_stopping_patience is not None else None
    max_grad_norm = float(train_cfg.get('max_grad_norm', 0.0))
    denoising = bool(train_cfg.get('denoising', False))
    denoise_sigma = float(train_cfg.get('denoise_sigma', 0.1))
    denoise_sigma_max = float(train_cfg.get('denoise_sigma_max', 0.6))
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = None
    started_at = _utc_now()
    dataloader_cfg = _dataloader_kwargs(train_cfg, device)
    run_summary: dict[str, Any] = {
        'status': 'running',
        'started_at': started_at,
        'updated_at': started_at,
        'config_path': str(config_path),
        'checkpoint_path': str(checkpoint_path),
        'metrics_csv_path': str(metrics_csv_path),
        'summary_json_path': str(summary_json_path),
        'device': device,
        'environment': collect_run_metadata(device),
        'reproducibility': reproducibility,
        'manifest_ok_rows': manifest_ok_rows,
        'electronic_descriptors_path': str(electronic_descriptors_path) if electronic_descriptors_path else None,
        'descriptor_rows_missing': descriptor_rows_missing,
        'excluded_manifest_rows': excluded_manifest_rows,
        'exclude_keys_path': str(exclude_keys_path) if exclude_keys_path else None,
        'evaluation_checkpoint_path': str(evaluation_checkpoint) if evaluation_checkpoint else None,
        'dataset_rows_used': len(rows),
        'hdf5_path': str(dataset_cfg['hdf5_path']),
        'effective_hdf5_path': str(effective_hdf5_path),
        'dataloader': dataloader_cfg,
        'split_sizes': {'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)},
        'split': split_metadata,
        'test_subset_sizes': {subset: len(keys) for subset, keys in test_subset_key_groups.items()},
        'target_columns': list(target_columns),
        'target_dim': len(target_columns),
        'loss': {
            'type': (config.get('loss') or {}).get('type', 'mse'),
            'weights': None if loss_weights is None else list(loss_weights),
            'normalize': normalize_loss_weights,
        },
        'config': config,
        'best_epoch': best_epoch,
        'best_val_loss': best_val if math.isfinite(best_val) else None,
        'latest_epoch': best_epoch or 0,
        'latest_metrics': None,
        'test_metrics': None,
        'stopped_early': False,
        'stop_reason': None,
    }
    write_summary_json(summary_json_path, run_summary)
    wandb_run = WandbRun(config, run_summary)
    if wandb_run.enabled:
        run_summary['wandb'] = wandb_run.metadata()
        write_summary_json(summary_json_path, run_summary)

    failed = False
    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            epoch_learning_rate = _current_lr(optimizer)
            train_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_n = 0
            total_unweighted_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                pred, predicted_dipoles = None, None
                if denoising:
                    sigma = denoise_sigma + (denoise_sigma_max - denoise_sigma) * torch.rand(
                        1, device=batch.pos.device
                    )
                    noise = torch.randn_like(batch.pos) * sigma
                    denoised_pos = batch.pos + noise
                    try:
                        disp = model.forward_denoise(batch.z, denoised_pos, batch.batch)
                        _require_finite(disp, 'training denoise predictions', batch)
                    except FloatingPointError:
                        if bool((config or {}).get('training', {}).get('skip_nonfinite_batches', False)):
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        raise
                    loss = weighted_mse_loss(disp, -noise)
                else:
                    target = batch.y.view(-1, len(target_columns))
                    try:
                        _require_finite(target, 'training targets', batch)
                        pred, predicted_dipoles = _forward_model(model, batch, len(target_columns))
                        target_dipoles = (
                            batch.transition_dipole.view_as(predicted_dipoles)
                            if predicted_dipoles is not None
                            else None
                        )
                        _require_finite(pred, 'training predictions', batch)
                    except FloatingPointError:
                        if bool((config or {}).get('training', {}).get('skip_nonfinite_batches', False)):
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        raise
                    if predicted_dipoles is not None:
                        _require_finite(predicted_dipoles, 'training transition dipoles', batch)
                    loss = _training_loss(
                        pred,
                        target,
                        target_columns,
                        config,
                        loss_weights_tensor,
                        normalize_loss_weights,
                        predicted_dipoles,
                        target_dipoles,
                    )
                _require_finite(loss, 'training loss', batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_grad_norm if max_grad_norm > 0 else math.inf,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    if bool((config or {}).get('training', {}).get('skip_nonfinite_batches', False)):
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    keys = getattr(batch, 'molecule_key', None)
                    raise FloatingPointError(f'Non-finite training gradient norm; molecule_keys={keys}') from exc
                optimizer.step()
                batch_n = batch.pos.size(0) if denoising else target.shape[0]
                total_loss += loss.item() * batch_n
                total_n += batch_n
                if not denoising and loss_weights_tensor is not None:
                    total_unweighted_loss += weighted_mse_loss(pred.detach(), target).item() * batch_n
            train_seconds = time.perf_counter() - train_started

            val_started = time.perf_counter()
            if denoising:
                val_metrics = evaluate_denoising(model, val_loader, device, config) if len(val_ds) else {'loss': float('nan')}
            else:
                val_metrics = evaluate(model, val_loader, device, target_columns, loss_weights_tensor, normalize_loss_weights, config, target_scale=target_scale) if len(val_ds) else {'loss': float('nan')}
            val_seconds = time.perf_counter() - val_started
            epoch_seconds = time.perf_counter() - epoch_started
            val_loss = val_metrics.get('loss', math.inf)
            if not math.isfinite(val_loss):
                raise FloatingPointError(f'Non-finite validation loss at epoch {epoch}')
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            record = {
                'epoch': epoch,
                'train_loss': total_loss / max(total_n, 1),
                'learning_rate': epoch_learning_rate,
                'epoch_seconds': epoch_seconds,
                'train_seconds': train_seconds,
                'val_seconds': val_seconds,
                'train_samples_per_second': total_n / max(train_seconds, 1e-9),
                'val_samples_per_second': len(val_ds) / max(val_seconds, 1e-9) if len(val_ds) else 0.0,
                **{f'val_{key}': value for key, value in val_metrics.items()},
            }
            if loss_weights_tensor is not None:
                record['train_unweighted_mse_loss'] = total_unweighted_loss / max(total_n, 1)
            history.append(record)
            print(record, flush=True)
            write_history_csv(metrics_csv_path, history)
            wandb_run.log_epoch(record)

            improvement = classify_validation_improvement(val_loss, best_val, early_stopping_best_val, min_delta)
            if improvement['checkpoint_improved']:
                best_val = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'model_kwargs': model_kwargs,
                        'config': config,
                        'history': history,
                    },
                    checkpoint_path,
                )

            if improvement['early_stopping_improved']:
                early_stopping_best_val = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            run_summary.update(
                {
                    'updated_at': _utc_now(),
                    'best_epoch': best_epoch,
                    'best_val_loss': best_val if math.isfinite(best_val) else None,
                    'latest_epoch': epoch,
                    'latest_metrics': record,
                    'stopped_early': stopped_early,
                    'stop_reason': stop_reason,
                }
            )
            write_summary_json(summary_json_path, run_summary)

            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                stopped_early = True
                stop_reason = f'validation loss did not improve for {early_stopping_patience} epochs'
                run_summary.update(
                    {
                        'status': 'stopped_early',
                        'updated_at': _utc_now(),
                        'stopped_early': stopped_early,
                        'stop_reason': stop_reason,
                    }
                )
                write_summary_json(summary_json_path, run_summary)
                break

        if not evaluation_checkpoint and best_epoch is not None and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        predictions_csv_path = evaluation_cfg.get('predictions_csv_path')
        if evaluation_checkpoint_paths:
            test_metrics = evaluate_ensemble(
                model,
                evaluation_checkpoint_paths,
                test_loader,
                device,
                target_columns,
                config,
                target_scale=target_scale,
            )
        else:
            test_metrics = evaluate(
            model,
            test_loader,
            device,
            target_columns,
            loss_weights_tensor,
            normalize_loss_weights,
            config,
            predictions_csv_path=predictions_csv_path,
            target_scale=target_scale,
        ) if len(test_ds) else {}
        per_subset_test_metrics: dict[str, dict[str, float]] = {}
        if bool(dataset_cfg.get('report_subset_metrics', False)):
            for subset_index, (subset_name, subset_keys) in enumerate(test_subset_key_groups.items()):
                subset_ds = QCDGES1Dataset(effective_hdf5_path, dataset_cfg['manifest_path'], subset_keys, target_columns)
                subset_loader = _make_loader(
                    subset_ds,
                    train_cfg,
                    device,
                    shuffle=False,
                    seed=training_seed + 100 + subset_index,
                )
                per_subset_test_metrics[subset_name] = evaluate(model, subset_loader, device, target_columns, loss_weights_tensor, normalize_loss_weights, config, target_scale=target_scale)
        wandb_run.log_test_metrics(test_metrics, best_epoch)
        if wandb_run.enabled and per_subset_test_metrics:
            flat_subset_metrics = {
                f'test_subset/{subset_name}/{metric_name}': metric_value
                for subset_name, metrics in per_subset_test_metrics.items()
                for metric_name, metric_value in metrics.items()
            }
            wandb_run.log_metrics(flat_subset_metrics)
        run_summary.update(
            {
                'status': 'completed',
                'completed_at': _utc_now(),
                'updated_at': _utc_now(),
                'stopped_early': stopped_early,
                'stop_reason': stop_reason,
                'test_metrics': test_metrics,
                'per_subset_test_metrics': per_subset_test_metrics,
                'predictions_csv_path': str(predictions_csv_path) if predictions_csv_path else None,
            }
        )
        write_summary_json(summary_json_path, run_summary)
        return {
            'history': history,
            'test_metrics': test_metrics,
            'checkpoint_path': str(checkpoint_path),
            'metrics_csv_path': str(metrics_csv_path),
            'summary_json_path': str(summary_json_path),
        }
    except Exception as exc:
        failed = True
        run_summary.update(
            {
                'status': 'failed',
                'updated_at': _utc_now(),
                'stop_reason': f'{type(exc).__name__}: {exc}',
            }
        )
        write_summary_json(summary_json_path, run_summary)
        raise
    finally:
        wandb_run.finish(exit_code=1 if failed else 0)
