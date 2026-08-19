from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


def test_parse_orca_zindo_extracts_orbitals_character_and_dipoles() -> None:
    from gnn_excited.data.zindo_descriptors import parse_orca_zindo

    output = """
ORBITAL ENERGIES
  NO   OCC          E(Eh)            E(eV)
   4   2.0000      -0.500000       -13.6057
   5   2.0000      -0.400000       -10.8846
   6   0.0000       0.100000         2.7211
   7   0.0000       0.200000         5.4423
                    * MULLIKEN POPULATION ANALYSIS *
STATE  1:  E=   0.100000 au      2.000 eV    10000.0 cm**-1 <S**2> = 0 Mult 1
     5a ->   6a  :     0.810000 (c= -0.90000000)
     4a ->   6a  :     0.040000 (c=  0.20000000)
STATE  2:  E= 0 au 3.000 eV 0 cm**-1
     5a ->   7a  :     1.000000 (c=  1.00000000)
STATE  3:  E= 0 au 4.000 eV 0 cm**-1
     5a ->   7a  :     1.000000 (c=  1.00000000)
STATE  4:  E= 0 au 5.000 eV 0 cm**-1
     5a ->   7a  :     1.000000 (c=  1.00000000)
STATE  5:  E= 0 au 6.000 eV 0 cm**-1
     5a ->   7a  :     1.000000 (c=  1.00000000)
ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
  0-1A -> 1-1A 2.000 10000.0 500.0 0.100000 0.25 0.3 0.4 0.0
  0-1A -> 2-1A 3.000 10000.0 400.0 0.200000 0.25 0.0 0.0 0.5
  0-1A -> 3-1A 4.000 10000.0 300.0 0.300000 0.25 0.0 0.0 0.5
  0-1A -> 4-1A 5.000 10000.0 250.0 0.400000 0.25 0.0 0.0 0.5
  0-1A -> 5-1A 6.000 10000.0 200.0 0.500000 0.25 0.0 0.0 0.5
ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS
****ORCA TERMINATED NORMALLY****
"""
    parsed = parse_orca_zindo(output)
    features = parsed["state_features"]
    assert features.shape == (5, 12)
    assert features[0, :2].tolist() == pytest.approx([2.0, np.log1p(0.1)])
    assert features[0, 2:6].tolist() == pytest.approx([0.3, 0.4, 0.0, 0.5])
    assert features[0, 6:10].tolist() == pytest.approx([0.9, 0.0, 0.0, 0.85])
    assert features[0, 10:].tolist() == pytest.approx([13.6057, 10.8846])
    assert parsed["transition_pairs"][0, :2].tolist() == [[5, 6], [4, 6]]


def test_screen_zindo_uses_matched_test_rows_and_five_fold_controls(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    from gnn_excited.data.zindo_descriptors import screen_descriptors

    descriptor_path = tmp_path / "zindo.h5"
    manifest_path = tmp_path / "manifest.csv"
    predictions_path = tmp_path / "predictions.csv"
    result_path = tmp_path / "screen.json"
    manifest_fields = ["molecule_key", "status", "random_split", *(f"S{s}_eV" for s in range(1, 6))]
    prediction_fields = ["molecule_key", *(f"prediction_S{s}_eV" for s in range(1, 6))]
    with (
        h5py.File(descriptor_path, "w") as descriptors,
        manifest_path.open("w", newline="", encoding="utf-8") as manifest_stream,
        predictions_path.open("w", newline="", encoding="utf-8") as prediction_stream,
    ):
        manifest_writer = csv.DictWriter(manifest_stream, fieldnames=manifest_fields)
        prediction_writer = csv.DictWriter(prediction_stream, fieldnames=prediction_fields)
        manifest_writer.writeheader()
        prediction_writer.writeheader()
        for index in range(30):
            split = "train" if index < 20 else "val" if index < 25 else "test"
            targets = np.arange(1, 6, dtype=np.float32) + index / 100
            features = np.zeros((5, 12), dtype=np.float32)
            features[:, 0] = targets + 0.1
            features[:, 1] = index
            descriptors.create_group(str(index)).create_dataset("state_features", data=features)
            manifest_writer.writerow(
                {
                    "molecule_key": index,
                    "status": "ok",
                    "random_split": split,
                    **{f"S{state}_eV": targets[state - 1] for state in range(1, 6)},
                }
            )
            if split == "test":
                prediction_writer.writerow(
                    {
                        "molecule_key": index,
                        **{
                            f"prediction_S{state}_eV": targets[state - 1] + 0.05
                            for state in range(1, 6)
                        },
                    }
                )
    result = screen_descriptors(descriptor_path, manifest_path, predictions_path, result_path)
    assert result["matched_test_molecules"] == 5
    assert result["metrics"]["raw_zindo_energy_mae_eV"] == pytest.approx(0.1)
    assert result["metrics"]["production_energy_mae_eV"] == pytest.approx(0.05, abs=1e-6)
    assert result["models"]["energy_only"]["selected_ridge_per_state"]
    assert result_path.is_file()
