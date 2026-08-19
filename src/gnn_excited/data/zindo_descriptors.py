"""Build and screen provenance-controlled ORCA ZINDO/S descriptors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from gnn_excited.data.electronic_descriptors import (
    _selected_rows,
    _sha256,
    _run_command,
)


SCHEMA_VERSION = "qm9gwbse-orca-zindo-v1"
FEATURE_NAMES = (
    "zindo_energy_eV",
    "log1p_zindo_f",
    "zindo_dipole_x_au",
    "zindo_dipole_y_au",
    "zindo_dipole_z_au",
    "zindo_dipole_magnitude_au",
    "dominant_amplitude_abs",
    "dominant_occupied_offset_from_homo",
    "dominant_virtual_offset_from_lumo",
    "top3_character_weight",
    "zindo_homo_lumo_gap_eV",
    "zindo_koopmans_ip_eV",
)
_ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
_ORBITAL_RE = re.compile(
    r"^\s*(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*$"
)
_STATE_RE = re.compile(
    r"^STATE\s+(\d+):\s+E=.*?\s+([+-]?\d+(?:\.\d+)?)\s+eV\b"
)
_TRANSITION_RE = re.compile(
    r"^\s*(\d+)[ab]\s*->\s*(\d+)[ab]\s*:\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*\(c=\s*([+-]?\d+(?:\.\d+)?)\)"
)
_SPECTRUM_RE = re.compile(
    r"^\s*0-\d+[A-Za-z]+\s*->\s*(\d+)-\d+[A-Za-z]+\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+\S+\s+\S+\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+\S+\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s*$"
)


def parse_orca_zindo(output: str, num_states: int = 5) -> dict[str, np.ndarray]:
    """Parse ORCA 6.1 ZINDO/S roots, characters, and length-gauge dipoles."""
    if "ORCA TERMINATED NORMALLY" not in output:
        raise ValueError("ORCA output did not terminate normally")
    orbitals: list[tuple[int, float, float]] = []
    in_orbitals = False
    for line in output.splitlines():
        if line.strip() == "ORBITAL ENERGIES":
            in_orbitals = True
            continue
        if in_orbitals and "MULLIKEN POPULATION ANALYSIS" in line:
            break
        if in_orbitals and (match := _ORBITAL_RE.match(line)):
            orbitals.append((int(match.group(1)), float(match.group(2)), float(match.group(4))))
    occupied = [row for row in orbitals if row[1] > 0]
    virtual = [row for row in orbitals if row[1] == 0]
    if not occupied or not virtual:
        raise ValueError("ORCA output is missing occupied or virtual orbital energies")
    homo_index, _, homo_energy = occupied[-1]
    lumo_index, _, lumo_energy = virtual[0]

    energies: dict[int, float] = {}
    transitions: dict[int, list[tuple[int, int, float, float]]] = {}
    current_state: int | None = None
    for line in output.splitlines():
        state_match = _STATE_RE.match(line)
        if state_match:
            current_state = int(state_match.group(1))
            energies[current_state] = float(state_match.group(2))
            transitions[current_state] = []
            continue
        if current_state is not None and (match := _TRANSITION_RE.match(line)):
            transitions[current_state].append(
                (int(match.group(1)), int(match.group(2)), float(match.group(3)), float(match.group(4)))
            )

    spectra: dict[int, tuple[float, float, float, float, float]] = {}
    in_length_spectrum = False
    for line in output.splitlines():
        if "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
            in_length_spectrum = True
            continue
        if in_length_spectrum and "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY" in line:
            break
        if in_length_spectrum and (match := _SPECTRUM_RE.match(line)):
            spectra[int(match.group(1))] = tuple(float(match.group(i)) for i in range(2, 7))

    available = sorted(set(energies) & set(spectra))
    if available[:num_states] != list(range(1, num_states + 1)):
        raise ValueError(f"Expected ordered ORCA roots 1-{num_states}, found {available}")
    root_count = max(available)
    root_energy = np.full(root_count, np.nan, dtype=np.float32)
    root_oscillator = np.full(root_count, np.nan, dtype=np.float32)
    root_dipoles = np.full((root_count, 3), np.nan, dtype=np.float32)
    state_features = np.empty((num_states, len(FEATURE_NAMES)), dtype=np.float32)
    transition_pairs = np.full((num_states, 3, 2), -1, dtype=np.int16)
    transition_coefficients = np.zeros((num_states, 3), dtype=np.float32)
    for state in available:
        spectrum_energy, oscillator, dx, dy, dz = spectra[state]
        root_energy[state - 1] = spectrum_energy
        root_oscillator[state - 1] = oscillator
        root_dipoles[state - 1] = (dx, dy, dz)
        if state > num_states:
            continue
        character = sorted(transitions.get(state, ()), key=lambda item: item[2], reverse=True)[:3]
        if not character:
            raise ValueError(f"ORCA state {state} has no printed transition character")
        dominant_occupied, dominant_virtual, _, dominant_coefficient = character[0]
        for index, (orbital_from, orbital_to, _, coefficient) in enumerate(character):
            transition_pairs[state - 1, index] = (orbital_from, orbital_to)
            transition_coefficients[state - 1, index] = coefficient
        dipole_magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
        state_features[state - 1] = (
            spectrum_energy,
            math.log1p(max(oscillator, 0.0)),
            dx,
            dy,
            dz,
            dipole_magnitude,
            abs(dominant_coefficient),
            dominant_occupied - homo_index,
            dominant_virtual - lumo_index,
            sum(item[2] for item in character),
            lumo_energy - homo_energy,
            -homo_energy,
        )
    if not np.isfinite(state_features).all():
        raise ValueError("ORCA ZINDO/S descriptors contain non-finite values")
    return {
        "state_features": state_features,
        "root_energies_eV": root_energy,
        "root_oscillator_strengths": root_oscillator,
        "root_transition_dipoles_au": root_dipoles,
        "transition_pairs": transition_pairs,
        "transition_coefficients": transition_coefficients,
    }


def orca_input(labels: np.ndarray, coords: np.ndarray, num_roots: int = 10) -> str:
    try:
        atoms = [_ELEMENTS[int(label)] for label in labels]
    except KeyError as exc:
        raise ValueError(f"Unsupported atomic number {exc.args[0]}") from exc
    lines = [
        "! ZINDO/S TightSCF NoAutoStart",
        "",
        "%cis",
        f"  NRoots {int(num_roots)}",
        "end",
        "",
        "* xyz 0 1",
    ]
    lines.extend(f"{atom} {x:.10f} {y:.10f} {z:.10f}" for atom, (x, y, z) in zip(atoms, coords))
    return "\n".join((*lines, "*", ""))


def _calculate_one(payload: tuple[Any, ...]) -> tuple[str, dict[str, np.ndarray] | None, str | None, float]:
    molecule_key, labels, coords, orca, scratch_root, timeout, num_roots = payload
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"zindo-{molecule_key}-", dir=scratch_root) as directory:
            workdir = Path(directory)
            input_path = workdir / "molecule.inp"
            input_path.write_text(orca_input(labels, coords, num_roots), encoding="utf-8")
            output = _run_command([str(orca), input_path.name], workdir, timeout)
            if "SCF CONVERGED AFTER" not in output or "ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR" not in output:
                raise RuntimeError("ORCA ZINDO/S did not converge both SCF and CIS")
            return molecule_key, parse_orca_zindo(output), None, time.perf_counter() - started
    except Exception as exc:  # keep the screening batch alive and persist exact failures
        return molecule_key, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - started


def build_descriptors(
    hdf5_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    orca: Path,
    limit: int,
    workers: int,
    seed: int,
    scratch_root: Path,
    timeout: int = 300,
    num_roots: int = 10,
    minimum_success_rate: float = 0.95,
) -> dict[str, Any]:
    if not orca.is_file() or not os.access(orca, os.X_OK):
        raise FileNotFoundError(f"ORCA executable is missing or not executable: {orca}")
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
                    orca,
                    scratch_root,
                    int(timeout),
                    int(num_roots),
                )
            )
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    completed: list[tuple[str, dict[str, np.ndarray], float]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        for index, (key, result, error, seconds) in enumerate(pool.map(_calculate_one, payloads), start=1):
            if result is None:
                failures.append({"molecule_key": key, "error": error, "seconds": seconds})
            else:
                completed.append((key, result, seconds))
            if index % 25 == 0 or index == len(payloads):
                print(f"completed={index}/{len(payloads)} ok={len(completed)} failed={len(failures)}", flush=True)
    train_arrays = [
        result["state_features"]
        for key, result, _ in completed
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
        target.attrs.update(
            schema_version=SCHEMA_VERSION,
            method="ORCA 6.1.1 ZINDO/S",
            feature_names_json=json.dumps(FEATURE_NAMES),
            train_mean=train_mean,
            train_std=train_std,
            selection_seed=int(seed),
            requested_count=int(limit),
            orca_sha256=_sha256(orca),
        )
        for key, result, seconds in completed:
            group = target.create_group(key)
            for name, value in result.items():
                group.create_dataset(name, data=value, compression="gzip", shuffle=True)
            group.attrs["split"] = selected_by_key[key].get("random_split", "")
            group.attrs["seconds"] = float(seconds)
    temporary_path.replace(output_path)
    failure_path = output_path.with_suffix(".failures.csv")
    with failure_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("molecule_key", "error", "seconds"))
        writer.writeheader()
        writer.writerows(failures)
    raw_errors: dict[str, list[float]] = {}
    near_degenerate = 0
    for key, result, _ in completed:
        features = result["state_features"]
        near_degenerate += int(np.any(np.diff(features[:, 0]) < 0.1))
        split = selected_by_key[key].get("random_split", "unknown")
        for group in ("all", split):
            raw_errors.setdefault(group, []).extend(
                abs(float(features[state, 0]) - float(selected_by_key[key][f"S{state + 1}_eV"]))
                for state in range(5)
            )
    success_rate = len(completed) / max(len(rows), 1)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": "ORCA 6.1.1 ZINDO/S",
        "output_path": str(output_path),
        "failure_path": str(failure_path),
        "requested": len(rows),
        "ok": len(completed),
        "failed": len(failures),
        "success_rate": success_rate,
        "wall_seconds": time.perf_counter() - started,
        "mean_calculation_seconds": float(np.mean([seconds for _, _, seconds in completed])) if completed else None,
        "near_degenerate_lt_0.1eV_count": near_degenerate,
        "near_degenerate_lt_0.1eV_fraction": near_degenerate / max(len(completed), 1),
        "raw_energy_mae_eV": {name: float(np.mean(values)) for name, values in raw_errors.items()},
        "feature_names": list(FEATURE_NAMES),
        "selection": {"type": "split-stratified deterministic hash", "seed": int(seed)},
        "executable": {"path": str(orca), "sha256": _sha256(orca)},
    }
    output_path.with_suffix(".provenance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if success_rate < float(minimum_success_rate):
        raise RuntimeError(
            f"ORCA success rate {success_rate:.3f} is below required {minimum_success_rate:.3f}"
        )
    return summary


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
    test = np.column_stack((np.ones(len(test_x)), (test_x - mean) / scale))
    penalty = np.eye(train.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    return test @ np.linalg.solve(train.T @ train + penalty, train.T @ train_y)


def screen_descriptors(
    descriptor_path: Path,
    manifest_path: Path,
    production_predictions_path: Path,
    output_path: Path,
    *,
    seed: int = 17,
    folds: int = 5,
) -> dict[str, Any]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        manifest = {row["molecule_key"]: row for row in csv.DictReader(stream) if row.get("status") == "ok"}
    with production_predictions_path.open(newline="", encoding="utf-8") as stream:
        production = {row["molecule_key"]: row for row in csv.DictReader(stream)}
    with h5py.File(descriptor_path, "r") as descriptors:
        keys = sorted((key for key in descriptors if key in manifest), key=int)
        x = np.stack([descriptors[key]["state_features"][()] for key in keys]).astype(np.float64)
    y = np.asarray(
        [[float(manifest[key][f"S{state}_eV"]) for state in range(1, 6)] for key in keys],
        dtype=np.float64,
    )
    splits = np.asarray([manifest[key].get("random_split", "") for key in keys])
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "val")
    test_indices = np.asarray([i for i, key in enumerate(keys) if splits[i] == "test" and key in production])
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise ValueError("ZINDO screening requires matched train, validation, and production-test rows")
    fold_ids = np.asarray(
        [int.from_bytes(hashlib.sha256(f"{seed}:{keys[i]}".encode()).digest()[:4]) % folds for i in train_indices]
    )
    feature_sets = {"energy_only": (0,), "all_descriptors": tuple(range(x.shape[-1]))}
    ridge_values = (1e-6, 1e-4, 1e-2, 1.0)
    models: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, feature_indices in feature_sets.items():
        state_predictions = np.empty((len(test_indices), 5), dtype=np.float64)
        state_oof_mae = []
        selected_ridges = []
        for state in range(5):
            state_x = x[:, state, feature_indices]
            state_x = state_x.reshape(len(keys), -1)
            residual = y[:, state] - x[:, state, 0]
            ridge_scores = []
            for ridge in ridge_values:
                errors = []
                for fold in range(folds):
                    fold_train = train_indices[fold_ids != fold]
                    fold_test = train_indices[fold_ids == fold]
                    if not len(fold_test):
                        continue
                    correction = _ridge_predict(state_x[fold_train], residual[fold_train], state_x[fold_test], ridge)
                    errors.extend(abs(x[fold_test, state, 0] + correction - y[fold_test, state]))
                ridge_scores.append((float(np.mean(errors)), ridge))
            oof_mae, selected_ridge = min(ridge_scores)
            fit_indices = np.concatenate((train_indices, validation_indices))
            correction = _ridge_predict(
                state_x[fit_indices], residual[fit_indices], state_x[test_indices], selected_ridge
            )
            state_predictions[:, state] = x[test_indices, state, 0] + correction
            state_oof_mae.append(oof_mae)
            selected_ridges.append(selected_ridge)
        predictions[name] = state_predictions
        models[name] = {
            "oof_energy_mae_eV": float(np.mean(state_oof_mae)),
            "oof_per_state_energy_mae_eV": state_oof_mae,
            "selected_ridge_per_state": selected_ridges,
        }
    test_y = y[test_indices]
    raw = x[test_indices, :, 0]
    production_values = np.asarray(
        [
            [float(production[keys[i]][f"prediction_S{state}_eV"]) for state in range(1, 6)]
            for i in test_indices
        ],
        dtype=np.float64,
    )
    metrics = {
        "raw_zindo_energy_mae_eV": float(np.mean(abs(raw - test_y))),
        "production_energy_mae_eV": float(np.mean(abs(production_values - test_y))),
    }
    for name, value in predictions.items():
        metrics[f"{name}_energy_mae_eV"] = float(np.mean(abs(value - test_y)))
        metrics[f"{name}_per_state_energy_mae_eV"] = np.mean(abs(value - test_y), axis=0).tolist()
    improvement = metrics["production_energy_mae_eV"] - metrics["all_descriptors_energy_mae_eV"]
    descriptor_gain = metrics["energy_only_energy_mae_eV"] - metrics["all_descriptors_energy_mae_eV"]
    summary = {
        "descriptor_path": str(descriptor_path),
        "production_predictions_path": str(production_predictions_path),
        "folds": int(folds),
        "seed": int(seed),
        "matched_test_molecules": int(len(test_indices)),
        "models": models,
        "metrics": metrics,
        "gate": {
            "production_improvement_eV": improvement,
            "additional_descriptor_gain_eV": descriptor_gain,
            "passes_0.001eV_production_gate": bool(improvement >= 0.001),
            "passes_energy_only_ablation": bool(descriptor_gain > 0),
            "recommend_20k": bool(improvement >= 0.001 and descriptor_gain > 0),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--hdf5", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--orca", type=Path, required=True)
    build.add_argument("--limit", type=int, required=True)
    build.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    build.add_argument("--seed", type=int, default=17)
    build.add_argument("--scratch-root", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    build.add_argument("--timeout", type=int, default=300)
    build.add_argument("--num-roots", type=int, default=10)
    build.add_argument("--minimum-success-rate", type=float, default=0.95)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--descriptors", type=Path, required=True)
    screen.add_argument("--manifest", type=Path, required=True)
    screen.add_argument("--production-predictions", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--seed", type=int, default=17)
    screen.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_descriptors(
            args.hdf5,
            args.manifest,
            args.output,
            orca=args.orca,
            limit=args.limit,
            workers=args.workers,
            seed=args.seed,
            scratch_root=args.scratch_root,
            timeout=args.timeout,
            num_roots=args.num_roots,
            minimum_success_rate=args.minimum_success_rate,
        )
    else:
        result = screen_descriptors(
            args.descriptors,
            args.manifest,
            args.production_predictions,
            args.output,
            seed=args.seed,
            folds=args.folds,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
