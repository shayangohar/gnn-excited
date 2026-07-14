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
from gnn_excited.models.dimenetpp import build_dimenetpp


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
    supported_loss_types = {'mse', 'mean_mse', 'weighted_mse'}
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

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def _physical_oscillator_column(column: str) -> str | None:
    if column.startswith('log1p_') and column.endswith('_f'):
        return column.removeprefix('log1p_')
    return None


def batch_mae(pred, target, target_columns: tuple[str, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    energy_maes: list[float] = []
    log_osc_maes: list[float] = []
    osc_maes: list[float] = []
    for idx, column in enumerate(target_columns):
        mae = (pred[:, idx] - target[:, idx]).abs().mean().item()
        metrics[f'{column}_mae'] = mae
        if column.endswith('_eV'):
            energy_maes.append(mae)
        physical_osc_column = _physical_oscillator_column(column)
        if physical_osc_column is not None:
            osc_mae = (torch.expm1(pred[:, idx]).clamp_min(0) - torch.expm1(target[:, idx])).abs().mean().item()
            metrics[f'{physical_osc_column}_mae'] = osc_mae
            log_osc_maes.append(mae)
            osc_maes.append(osc_mae)
    if energy_maes:
        metrics['energy_eV_mae'] = sum(energy_maes) / len(energy_maes)
    if log_osc_maes:
        metrics['log1p_oscillator_strength_mae'] = sum(log_osc_maes) / len(log_osc_maes)
    if osc_maes:
        metrics['oscillator_strength_mae'] = sum(osc_maes) / len(osc_maes)
    return metrics


def evaluate(
    model,
    loader,
    device: str,
    target_columns: tuple[str, ...],
    loss_weights=None,
    normalize_loss_weights: bool = True,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {'loss': 0.0, 'n': 0}
    target_dim = len(target_columns)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.y.view(-1, target_dim)
            pred = model(batch.z, batch.pos, batch.batch).view(-1, target_dim)
            loss = weighted_mse_loss(pred, target, loss_weights, normalize_loss_weights)
            metrics = batch_mae(pred, target, target_columns)
            batch_n = target.shape[0]
            totals['loss'] += loss.item() * batch_n
            if loss_weights is not None:
                unweighted_loss = weighted_mse_loss(pred, target)
                totals['unweighted_mse_loss'] = totals.get('unweighted_mse_loss', 0.0) + unweighted_loss.item() * batch_n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_n
            totals['n'] += batch_n
    n = max(totals.pop('n'), 1)
    return {key: value / n for key, value in totals.items()}

def build_scheduler(optimizer, scheduler_cfg: dict[str, Any] | None):
    if not scheduler_cfg:
        return None
    scheduler_type = scheduler_cfg.get('type')
    if scheduler_type in (None, 'none'):
        return None
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
    target_columns = _target_columns_from_config(config)
    configured_out_channels = int(model_kwargs.get('out_channels', len(target_columns)))
    if configured_out_channels != len(target_columns):
        raise ValueError(
            'model.out_channels={} does not match {} targets'.format(configured_out_channels, len(target_columns))
        )
    model_kwargs['out_channels'] = len(target_columns)
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
    train_ds = QCDGES1Dataset(effective_hdf5_path, dataset_cfg['manifest_path'], _subset_keys(rows, train_idx), target_columns)
    val_ds = QCDGES1Dataset(effective_hdf5_path, dataset_cfg['manifest_path'], _subset_keys(rows, val_idx), target_columns)
    test_ds = QCDGES1Dataset(effective_hdf5_path, dataset_cfg['manifest_path'], _subset_keys(rows, test_idx), target_columns)

    train_loader = _make_loader(train_ds, train_cfg, device, shuffle=True, seed=training_seed)
    val_loader = _make_loader(val_ds, train_cfg, device, shuffle=False, seed=training_seed + 1)
    test_loader = _make_loader(test_ds, train_cfg, device, shuffle=False, seed=training_seed + 2)

    model = build_dimenetpp(**model_kwargs).to(device)
    if loss_weights_tensor is not None:
        loss_weights_tensor = loss_weights_tensor.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg['learning_rate']),
        weight_decay=float(train_cfg.get('weight_decay', 0.0)),
    )
    scheduler = build_scheduler(optimizer, train_cfg.get('scheduler'))
    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch: int | None = None
    early_stopping_best_val = math.inf
    min_delta = float(train_cfg.get('early_stopping_min_delta', 0.0))
    early_stopping_patience = train_cfg.get('early_stopping_patience')
    early_stopping_patience = int(early_stopping_patience) if early_stopping_patience is not None else None
    max_grad_norm = float(train_cfg.get('max_grad_norm', 0.0))
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
        'best_epoch': None,
        'best_val_loss': None,
        'latest_epoch': 0,
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

    try:
        for epoch in range(1, int(train_cfg['epochs']) + 1):
            epoch_started = time.perf_counter()
            train_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_n = 0
            total_unweighted_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                target = batch.y.view(-1, len(target_columns))
                pred = model(batch.z, batch.pos, batch.batch).view(-1, len(target_columns))
                loss = weighted_mse_loss(pred, target, loss_weights_tensor, normalize_loss_weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                batch_n = target.shape[0]
                total_loss += loss.item() * batch_n
                total_n += batch_n
                if loss_weights_tensor is not None:
                    total_unweighted_loss += weighted_mse_loss(pred.detach(), target).item() * batch_n
            train_seconds = time.perf_counter() - train_started

            val_started = time.perf_counter()
            val_metrics = evaluate(model, val_loader, device, target_columns, loss_weights_tensor, normalize_loss_weights) if len(val_ds) else {'loss': float('nan')}
            val_seconds = time.perf_counter() - val_started
            epoch_seconds = time.perf_counter() - epoch_started
            val_loss = val_metrics.get('loss', math.inf)
            if scheduler is not None and math.isfinite(val_loss):
                scheduler.step(val_loss)
            record = {
                'epoch': epoch,
                'train_loss': total_loss / max(total_n, 1),
                'learning_rate': _current_lr(optimizer),
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

        if best_epoch is not None and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate(model, test_loader, device, target_columns, loss_weights_tensor, normalize_loss_weights) if len(test_ds) else {}
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
                per_subset_test_metrics[subset_name] = evaluate(model, subset_loader, device, target_columns, loss_weights_tensor, normalize_loss_weights)
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
    finally:
        wandb_run.finish()
