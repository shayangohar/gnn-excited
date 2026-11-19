"""Time the UMA-S QM9GWBSE training step and candidate speedups (TASK-336, KNOW-335).

Builds the same model, loss, AdamW, gradient clipping, LR schedule and EMA as
scripts/train_uma_qm9gwbse.py, runs real training batches, and writes <out>.json with ms/step,
molecules/s, peak memory and the share of wall time spent waiting for data. With --profile-steps
it also writes the torch.profiler tables to <out>.profile.txt.

Variants (combine freely):
  --mole split|bmm  split: one F.linear per molecule per MOLE layer (the production patch, KNOW-222).
                    bmm: pad every molecule's rows to the longest and run one batched matmul per layer.
  --no-sync         drop the per-step torch.isfinite(loss) and float(loss) GPU->CPU syncs.
  --fused-adam      AdamW(fused=True) instead of the default foreach implementation.
  --compile         torch.compile(model, dynamic=True).
  --check-grads     compare split and bmm predictions and gradients on one batch at fp32 "highest".

Example:
  python scripts/profile_uma_training.py --mole bmm --check-grads --out runs/profile_uma/bmm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_uma_qm9gwbse import (  # noqa: E402
    PackedMolecules, UMAExcitedState, backbone_param_groups, collate, load_backbone, loss_kwargs,
    lr_lambda_factory, patch_mole_split_forward, qm9gwbse_loss,
)

_FORWARDS: dict = {}
_PAD_CACHE: dict = {}


def _pad_index(sizes: list[int], longest: int, device: torch.device) -> torch.Tensor:
    """Row positions of each molecule's rows inside a (molecules x longest) padded block."""
    key = (tuple(sizes), str(device))
    if key not in _PAD_CACHE:
        _PAD_CACHE.clear()
        starts = np.arange(len(sizes)) * longest
        index = np.concatenate([start + np.arange(n) for start, n in zip(starts, sizes)])
        _PAD_CACHE[key] = torch.from_numpy(index).pin_memory().to(device, non_blocking=True)
    return _PAD_CACHE[key]


def mole_forwards() -> dict:
    """{"stock", "split", "bmm"} implementations of fairchem's MOLE.forward."""
    if _FORWARDS:
        return _FORWARDS
    from fairchem.core.models.uma.nn import mole as mole_mod

    stock = mole_mod.MOLE.forward
    if getattr(stock, "_gnn_excited_split", False):
        raise RuntimeError("MOLE.forward is already patched; call mole_forwards() before patching")
    patch_mole_split_forward()
    split = mole_mod.MOLE.forward

    def bmm_forward(self, x):
        g = self.global_mole_tensors
        sizes = g.mole_sizes.tolist()  # mole_sizes is a CPU tensor (escn_moe moves it once per forward)
        if g.ac_start_idx != 0 or x.shape[0] != sum(sizes):
            return stock(self, x)
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum("eoi, be->bio", self.weights, g.expert_mixing_coefficients)
        n_sys, longest = len(sizes), max(sizes)
        index = _pad_index(sizes, longest, x.device)
        flat = x.reshape(x.shape[0], -1, x.shape[-1])  # rows, inner, in_features
        inner = flat.shape[1]
        padded = flat.new_zeros(n_sys * longest, inner, flat.shape[-1]).index_copy(0, index, flat)
        out = torch.bmm(padded.view(n_sys, longest * inner, -1), weights)  # molecules, longest*inner, out
        out = out.view(n_sys * longest, inner, -1).index_select(0, index)
        out = out.reshape(*x.shape[:-1], out.shape[-1])
        return out + self.bias if self.bias is not None else out

    def labelled(name, fn):
        def forward(self, x):
            with torch.profiler.record_function(f"mole_{name}"):
                return fn(self, x)
        return forward

    _FORWARDS.update(stock=stock, split=labelled("split", split), bmm=labelled("bmm", bmm_forward))
    return _FORWARDS


def set_mole(kind: str) -> None:
    from fairchem.core.models.uma.nn import mole as mole_mod

    mole_mod.MOLE.forward = mole_forwards()[kind]


def grad_check(model, batch, target, target_columns, loss_kw) -> dict:
    """Predictions and gradients of split vs bmm on one batch, at full fp32 matmul precision."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    results = {}
    for kind in ("split", "bmm"):
        set_mole(kind)
        model.zero_grad(set_to_none=True)
        pred = model(batch)
        qm9gwbse_loss(pred.float(), target, target_columns, **loss_kw).backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        results[kind] = (pred.detach().clone(), grads.clone())
    model.zero_grad(set_to_none=True)
    torch.set_float32_matmul_precision(previous)
    (p_split, g_split), (p_bmm, g_bmm) = results["split"], results["bmm"]
    return {
        "pred_max_abs_diff": float((p_bmm - p_split).abs().max()),
        "grad_rel_l2_diff": float((g_bmm - g_split).norm() / g_split.norm()),
        "grad_numel": int(g_split.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/uma_qm9gwbse_richind_s17.yaml")
    parser.add_argument("--mole", choices=["split", "bmm"], default="split")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--check-grads", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--out", required=True, help="output prefix")
    args = parser.parse_args()
    started_at = time.time()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg, data_cfg, train_cfg = config["model"], config["dataset"], config["training"]
    loss_kw = loss_kwargs(config.get("loss") or {})
    target_columns = tuple(config["targets"]["columns"])
    torch.manual_seed(17)
    np.random.seed(17)
    torch.set_float32_matmul_precision(str(train_cfg.get("matmul_precision", "high")))
    device = torch.device("cuda")

    arrays = dict(np.load(data_cfg["packed_path"], allow_pickle=False))
    available = [str(c) for c in arrays["target_columns"]]
    target_index = np.asarray([available.index(c) for c in target_columns])
    train_idx = np.flatnonzero(arrays["split"].astype(str) == "train")
    train_targets = arrays["targets"][train_idx][:, target_index]
    target_mean, target_std = train_targets.mean(axis=0), np.maximum(train_targets.std(axis=0), 1e-6)
    workers = int(train_cfg.get("num_workers", 8)) if args.num_workers is None else args.num_workers
    batch_size = int(train_cfg["batch_size"])
    loader = DataLoader(PackedMolecules(arrays, train_idx, target_index), batch_size=batch_size, shuffle=True,
                        drop_last=True, generator=torch.Generator().manual_seed(17), num_workers=workers,
                        collate_fn=collate, pin_memory=False, persistent_workers=workers > 0)

    mole_forwards()
    set_mole(args.mole)
    backbone = load_backbone(model_cfg["checkpoint_path"])
    model = UMAExcitedState(
        backbone, target_columns, target_mean, target_std,
        embed_dim=int(getattr(backbone, "sphere_channels", model_cfg.get("embed_dim", 128))),
        head_hidden=int(model_cfg.get("head_hidden", 256)), readout=str(model_cfg.get("readout", "l0_mean")),
        readout_layers=int(model_cfg.get("readout_layers", 0)), lmax=int(model_cfg.get("lmax", 2)),
    ).to(device)
    model.train()

    batches = iter(loader)

    def next_batch():
        nonlocal batches
        try:
            return next(batches)
        except StopIteration:
            batches = iter(loader)
            return next(batches)

    result: dict = {"args": vars(args), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "batch_size": batch_size, "num_workers": workers}
    if args.check_grads:
        batch, target, _ = next_batch()
        result["grad_check"] = grad_check(model, batch.to(device), target.to(device), target_columns, loss_kw)
        set_mole(args.mole)
        print("GRAD_CHECK " + json.dumps(result["grad_check"]), flush=True)

    head_params = [p for head in model.heads() for p in head.parameters()]
    groups = backbone_param_groups(model.backbone, float(train_cfg["backbone_lr"]),
                                   float(train_cfg.get("backbone_weight_decay", 0.0)), train_cfg.get("layer_lr_decay"))
    optimizer = torch.optim.AdamW(
        groups + [{"params": head_params, "lr": float(train_cfg["head_lr"]),
                   "weight_decay": float(train_cfg.get("head_weight_decay", 1e-3)), "name": "heads"}],
        fused=True if args.fused_adam else None,
    )
    steps_per_epoch = len(loader)
    sched_cfg = train_cfg.get("scheduler") or {}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(
        int(train_cfg["epochs"]) * steps_per_epoch, int(float(sched_cfg.get("warmup_epochs", 1)) * steps_per_epoch),
        float(sched_cfg.get("start_factor", 0.01)), float(sched_cfg.get("min_factor", 0.01))))
    ema = None
    if train_cfg.get("ema_decay"):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(train_cfg["ema_decay"])), use_buffers=True)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    step_model = torch.compile(model, dynamic=True) if args.compile else model

    data_wait = 0.0
    loss_sum = torch.zeros((), device=device)

    def step() -> None:
        nonlocal data_wait, loss_sum
        tick = time.perf_counter()
        batch, target, _ = next_batch()
        data_wait += time.perf_counter() - tick
        batch, target = batch.to(device), target.to(device)
        pred = step_model(batch)
        loss = qm9gwbse_loss(pred.float(), target, target_columns, **loss_kw)
        if not args.no_sync and not torch.isfinite(loss):  # the production per-step check (a sync)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            return
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update_parameters(model)
        if args.no_sync:
            loss_sum += loss.detach()
        else:
            loss_sum += float(loss)  # the production running-loss read (a second sync)

    warm_start = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    result["warmup_seconds"] = time.perf_counter() - warm_start
    torch.cuda.reset_peak_memory_stats()
    data_wait = 0.0
    result["timed_start_epoch"] = time.time()
    began = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - began
    result["timed_end_epoch"] = time.time()
    result.update({
        "steps": args.steps,
        "ms_per_step": 1000 * elapsed / args.steps,
        "molecules_per_s": args.steps * batch_size / elapsed,
        "data_wait_fraction": data_wait / elapsed,
        "peak_allocated_GB": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_GB": torch.cuda.max_memory_reserved() / 1e9,
        "mean_loss": float(loss_sum) / max(1, args.warmup + args.steps),
    })
    print("TIMING " + json.dumps({k: result[k] for k in ("ms_per_step", "molecules_per_s", "data_wait_fraction",
                                                          "peak_allocated_GB")}), flush=True)

    if args.profile_steps:
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            began = time.perf_counter()
            for _ in range(args.profile_steps):
                step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - began
        averages = prof.key_averages()
        result["profile"] = {"steps": args.profile_steps, "ms_per_step_under_profiler": 1000 * wall / args.profile_steps}
        try:  # event attribute names drift between torch releases; keep the timing if they do
            kernels = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
            kernel_us = sum(e.time_range.elapsed_us() for e in kernels)
            result["profile"].update({
                "kernels_per_step": len(kernels) / args.profile_steps,
                "kernel_busy_fraction": kernel_us / (wall * 1e6),
                "mean_kernel_us": kernel_us / max(1, len(kernels)),
            })
        except Exception as exc:
            result["profile"]["kernel_stats_error"] = f"{type(exc).__name__}: {exc}"
        Path(f"{args.out}.profile.txt").write_text(
            "== by self device time\n" + averages.table(sort_by="self_device_time_total", row_limit=30)
            + "\n\n== by self CPU time\n" + averages.table(sort_by="self_cpu_time_total", row_limit=30) + "\n",
            encoding="utf-8")
        print("PROFILE " + json.dumps(result["profile"]), flush=True)

    result["wall_seconds"] = time.time() - started_at
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
