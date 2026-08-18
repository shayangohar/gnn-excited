"""Generate compact per-state sTDA-xTB descriptors for QM9GWBSE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from gnn_excited.models.visnet import OSCILLATOR_STRENGTH_FACTOR


SCHEMA_VERSION = "qm9gwbse-electronic-descriptors-v1"
FEATURE_NAMES = (
    "stda_energy_eV",
    "log1p_stda_f",
    "stda_dipole_magnitude_D",
    "dominant_amplitude_abs",
    "dominant_occupied_offset_from_homo",
    "dominant_virtual_offset_from_lumo",
    "top3_character_weight",
    "xtb_homo_lumo_gap_eV",
    "xtb_koopmans_ip_eV",
)
_ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
_STATE_RE = re.compile(
    r"^\s*(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+(.*)$"
)
_AMPLITUDE_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\(\s*(\d+)\s*->\s*(\d+)\s*\)"
)


def parse_stda_descriptors(ground_output: str, excited_output: str, num_states: int = 5) -> np.ndarray:
    """Parse the stable text tables emitted by xtb4stda 1.0/stda 1.6.1."""
    gap_match = re.search(r"gap \(eV\)\s*:\s*([+-]?\d+(?:\.\d+)?)", ground_output)
    ip_match = re.search(r"Koopmans IP \(eV\)\s*:\s*([+-]?\d+(?:\.\d+)?)", ground_output)
    electrons_match = re.search(r"# electrons in TDA:\s*([+-]?\d+(?:\.\d+)?)", excited_output)
    if not (gap_match and ip_match and electrons_match):
        raise ValueError("sTDA-xTB output is missing gap, Koopmans IP, or electron count")
    homo = int(round(float(electrons_match.group(1)) / 2.0))
    gap = float(gap_match.group(1))
    ip = float(ip_match.group(1))
    in_state_table = False
    rows: list[list[float]] = []
    for line in excited_output.splitlines():
        if "excitation energies, transition moments and TDA amplitudes" in line:
            in_state_table = True
            continue
        if not in_state_table:
            continue
        match = _STATE_RE.match(line)
        if match is None:
            if rows and line.strip() and not line.lstrip().startswith("state"):
                break
            continue
        state = int(match.group(1))
        if state != len(rows) + 1:
            raise ValueError(f"Unexpected sTDA state ordering: got {state} after {len(rows)} states")
        energy = float(match.group(2))
        oscillator = float(match.group(4))
        amplitudes = [
            (float(value), int(occupied), int(virtual))
            for value, occupied, virtual in _AMPLITUDE_RE.findall(match.group(6))
        ]
        if not amplitudes:
            raise ValueError(f"sTDA state {state} has no transition amplitudes")
        dominant, occupied, virtual = max(amplitudes, key=lambda item: abs(item[0]))
        dipole = math.sqrt(max(oscillator, 0.0) / max(OSCILLATOR_STRENGTH_FACTOR * energy, 1e-12))
        rows.append(
            [
                energy,
                math.log1p(max(oscillator, 0.0)),
                dipole,
                abs(dominant),
                occupied - homo,
                virtual - (homo + 1),
                sum(value * value for value, _, _ in amplitudes[:3]),
                gap,
                ip,
            ]
        )
        if len(rows) == int(num_states):
            break
    if len(rows) != int(num_states):
        raise ValueError(f"Expected {num_states} sTDA roots, found {len(rows)}")
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("sTDA descriptors contain non-finite values")
    return result


def _xyz_text(labels: np.ndarray, coords: np.ndarray, molecule_key: str) -> str:
    try:
        atoms = [_ELEMENTS[int(label)] for label in labels]
    except KeyError as exc:
        raise ValueError(f"Unsupported atomic number {exc.args[0]} in molecule {molecule_key}") from exc
    lines = [str(len(atoms)), f"QM9GWBSE {molecule_key}"]
    lines.extend(
        f"{atom} {x:.10f} {y:.10f} {z:.10f}"
        for atom, (x, y, z) in zip(atoms, coords)
    )
    return "\n".join(lines) + "\n"


def _run_command(command: list[str], workdir: Path, timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} exited {completed.returncode}: {completed.stdout[-1000:]}")
    return completed.stdout


def _calculate_one(payload: tuple[Any, ...]) -> tuple[str, np.ndarray | None, str | None, float]:
    molecule_key, labels, coords, xtb4stda, stda, parameter_dir, scratch_root, timeout = payload
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"stda-{molecule_key}-", dir=scratch_root) as directory:
            workdir = Path(directory)
            xyz_path = workdir / "molecule.xyz"
            xyz_path.write_text(_xyz_text(labels, coords, molecule_key), encoding="utf-8")
            environment_value = os.environ.get("XTB4STDAHOME")
            os.environ["XTB4STDAHOME"] = str(parameter_dir)
            try:
                ground = _run_command([str(xtb4stda), xyz_path.name], workdir, timeout)
                if not (workdir / "wfn.xtb").exists() or "SCC done" not in ground:
                    raise RuntimeError("xtb4stda did not produce a converged wfn.xtb")
                last_error: Exception | None = None
                for energy_window in (12, 20):
                    excited = _run_command(
                        [str(stda), "-xtb", "wfn.xtb", "-e", str(energy_window)],
                        workdir,
                        timeout,
                    )
                    try:
                        descriptors = parse_stda_descriptors(ground, excited)
                    except ValueError as exc:
                        last_error = exc
                        continue
                    return molecule_key, descriptors, None, time.perf_counter() - started
                raise last_error or RuntimeError("stda produced fewer than five roots")
            finally:
                if environment_value is None:
                    os.environ.pop("XTB4STDAHOME", None)
                else:
                    os.environ["XTB4STDAHOME"] = environment_value
    except Exception as exc:  # keep the batch alive and persist exact failures
        return molecule_key, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - started


def _selected_rows(manifest_path: Path, limit: int, seed: int) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status") == "ok"]
    rows.sort(
        key=lambda row: (
            row.get("random_split", ""),
            hashlib.sha256(f"{seed}:{row['molecule_key']}".encode()).digest(),
        )
    )
    by_split: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_split.setdefault(row.get("random_split", "unknown"), []).append(row)
    selected: list[dict[str, str]] = []
    total = len(rows)
    for split_rows in by_split.values():
        count = min(len(split_rows), int(round(limit * len(split_rows) / total)))
        selected.extend(split_rows[:count])
    if len(selected) < min(limit, total):
        chosen = {row["molecule_key"] for row in selected}
        remaining = [row for row in rows if row["molecule_key"] not in chosen]
        selected.extend(remaining[: min(limit, total) - len(selected)])
    return sorted(selected[:limit], key=lambda row: int(row["molecule_key"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_descriptors(
    hdf5_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    xtb4stda: Path,
    stda: Path,
    parameter_dir: Path,
    limit: int,
    workers: int,
    seed: int,
    scratch_root: Path,
    timeout: int = 300,
) -> dict[str, Any]:
    """Calculate and persist a split-stratified descriptor pilot with train-only scaling."""
    for executable in (xtb4stda, stda):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"Executable is missing or not executable: {executable}")
    for name in (".param_stda1.xtb", ".param_stda2.xtb", ".xtb4stdarc"):
        if not (parameter_dir / name).is_file():
            raise FileNotFoundError(f"Required xtb4stda parameter file is missing: {parameter_dir / name}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows(manifest_path, int(limit), int(seed))
    selected_by_key = {row["molecule_key"]: row for row in rows}
    payloads = []
    with h5py.File(hdf5_path, "r") as source:
        for row in rows:
            molecule = source[row["molecule_key"]]["ground_state"]
            payloads.append(
                (
                    row["molecule_key"],
                    np.asarray(molecule["labels"][()], dtype=np.int16),
                    np.asarray(molecule["coords"][()], dtype=np.float32),
                    xtb4stda,
                    stda,
                    parameter_dir,
                    scratch_root,
                    int(timeout),
                )
            )
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    completed: list[tuple[str, np.ndarray, float]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        for index, (key, descriptors, error, seconds) in enumerate(pool.map(_calculate_one, payloads), start=1):
            if descriptors is None:
                failures.append({"molecule_key": key, "error": error, "seconds": seconds})
            else:
                completed.append((key, descriptors, seconds))
            if index % 100 == 0 or index == len(payloads):
                print(f"completed={index}/{len(payloads)} ok={len(completed)} failed={len(failures)}", flush=True)
    train_arrays = [
        array
        for key, array, _ in completed
        if selected_by_key[key].get("random_split") == "train"
    ]
    if not train_arrays:
        raise RuntimeError("No successful training descriptors were generated")
    train_values = np.concatenate(train_arrays, axis=0).astype(np.float64)
    train_mean = train_values.mean(axis=0)
    train_std = train_values.std(axis=0)
    train_std[train_std < 1e-8] = 1.0
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(temporary_path, "w") as target:
        target.attrs["schema_version"] = SCHEMA_VERSION
        target.attrs["method"] = "sTDA-xTB"
        target.attrs["feature_names_json"] = json.dumps(FEATURE_NAMES)
        target.attrs["train_mean"] = train_mean
        target.attrs["train_std"] = train_std
        target.attrs["selection_seed"] = int(seed)
        target.attrs["requested_count"] = int(limit)
        target.attrs["xtb4stda_sha256"] = _sha256(xtb4stda)
        target.attrs["stda_sha256"] = _sha256(stda)
        for key, descriptors, seconds in completed:
            group = target.create_group(key)
            group.create_dataset("state_features", data=descriptors, compression="gzip", shuffle=True)
            group.attrs["split"] = selected_by_key[key].get("random_split", "")
            group.attrs["seconds"] = float(seconds)
    temporary_path.replace(output_path)
    failure_path = output_path.with_suffix(".failures.csv")
    with failure_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("molecule_key", "error", "seconds"))
        writer.writeheader()
        writer.writerows(failures)
    near_degenerate = sum(
        int(np.any(np.diff(array[:, 0]) < 0.1)) for _, array, _ in completed
    )
    raw_baseline: dict[str, dict[str, list[float]]] = {}
    for key, array, _ in completed:
        row = selected_by_key[key]
        split = row.get("random_split", "unknown")
        for group in ("all", split):
            errors = raw_baseline.setdefault(
                group, {"energy": [], "log_oscillator": []}
            )
            for state in range(5):
                errors["energy"].append(
                    abs(float(array[state, 0]) - float(row[f"S{state + 1}_eV"]))
                )
                errors["log_oscillator"].append(
                    abs(
                        float(array[state, 1])
                        - float(row[f"log1p_S{state + 1}_f"])
                    )
                )
    raw_baseline_metrics = {
        group: {
            "energy_mae_eV": float(np.mean(errors["energy"])),
            "log1p_oscillator_mae": float(np.mean(errors["log_oscillator"])),
            "molecules": len(errors["energy"]) // 5,
        }
        for group, errors in raw_baseline.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": "sTDA-xTB",
        "output_path": str(output_path),
        "failure_path": str(failure_path),
        "requested": len(rows),
        "ok": len(completed),
        "failed": len(failures),
        "success_rate": len(completed) / max(len(rows), 1),
        "wall_seconds": time.perf_counter() - started,
        "mean_calculation_seconds": float(np.mean([seconds for _, _, seconds in completed])) if completed else None,
        "near_degenerate_lt_0.1eV_count": near_degenerate,
        "near_degenerate_lt_0.1eV_fraction": near_degenerate / max(len(completed), 1),
        "raw_low_level_baseline": raw_baseline_metrics,
        "feature_names": list(FEATURE_NAMES),
        "selection": {"type": "split-stratified deterministic hash", "seed": int(seed)},
        "executables": {
            "xtb4stda": str(xtb4stda),
            "xtb4stda_sha256": _sha256(xtb4stda),
            "stda": str(stda),
            "stda_sha256": _sha256(stda),
            "parameter_dir": str(parameter_dir),
        },
    }
    summary_path = output_path.with_suffix(".provenance.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xtb4stda", type=Path, required=True)
    parser.add_argument("--stda", type=Path, required=True)
    parser.add_argument("--parameter-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scratch-root", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    summary = build_descriptors(
        args.hdf5,
        args.manifest,
        args.output,
        xtb4stda=args.xtb4stda,
        stda=args.stda,
        parameter_dir=args.parameter_dir,
        limit=args.limit,
        workers=args.workers,
        seed=args.seed,
        scratch_root=args.scratch_root,
        timeout=args.timeout,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
