from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest


def test_parse_stda_descriptors_extracts_root_character_and_physics() -> None:
    from gnn_excited.data.electronic_descriptors import parse_stda_descriptors
    from gnn_excited.models.visnet import OSCILLATOR_STRENGTH_FACTOR

    ground = "gap (eV) : 5.000\nKoopmans IP (eV) : 10.000\n"
    excited = """
# electrons in TDA:  10.000
excitation energies, transition moments and TDA amplitudes
state    eV      nm       fL        Rv(corr)
  1  2.000  620.0  0.1000  0.0  -0.80(  5->  6) 0.20(  4->  6)
  2  3.000  413.3  0.2000  0.0   0.70(  4->  6) 0.30(  5->  7)
  3  4.000  310.0  0.3000  0.0   0.60(  5->  8)
  4  5.000  248.0  0.4000  0.0   0.50(  3->  6)
  5  6.000  206.7  0.5000  0.0   0.40(  5->  7)
"""
    descriptors = parse_stda_descriptors(ground, excited)
    assert descriptors.shape == (5, 9)
    assert descriptors[0, 0] == pytest.approx(2.0)
    assert descriptors[0, 3] == pytest.approx(0.8)
    assert descriptors[0, 4:6].tolist() == pytest.approx([0.0, 0.0])
    assert descriptors[0, 6] == pytest.approx(0.8**2 + 0.2**2)
    assert descriptors[0, 2] == pytest.approx(
        np.sqrt(0.1 / (OSCILLATOR_STRENGTH_FACTOR * 2.0))
    )


def test_electronic_delta_starts_at_frozen_energy_and_backpropagates() -> None:
    pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    from gnn_excited.models.visnet import (
        OSCILLATOR_STRENGTH_FACTOR,
        build_visnet,
        load_transfer_checkpoint,
    )

    columns = tuple(
        column
        for state in range(1, 6)
        for column in (f"S{state}_eV", f"log1p_S{state}_f")
    )
    kwargs = {
        "target_columns": columns,
        "hidden_channels": 32,
        "num_layers": 1,
        "num_rbf": 8,
        "cutoff": 3.0,
        "max_num_neighbors": 8,
    }
    baseline = build_visnet(**kwargs)
    delta = build_visnet(
        **kwargs,
        electronic_descriptor_delta=True,
        descriptor_dim=9,
        num_states=5,
    )
    load_transfer_checkpoint(
        delta,
        {"model_state_dict": baseline.state_dict()},
        mode="frozen_sidecar",
    )
    z = torch.tensor([6, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]
    )
    batch = torch.zeros(3, dtype=torch.long)
    descriptors = torch.randn(1, 5, 9)
    baseline_prediction = baseline(z, pos, batch)
    prediction, dipoles = delta(
        z, pos, batch, electronic_descriptors=descriptors
    )
    assert torch.equal(prediction[:, 0::2], baseline_prediction[:, 0::2])
    assert torch.allclose(
        prediction[:, 1::2],
        torch.log1p(
            OSCILLATOR_STRENGTH_FACTOR
            * prediction[:, 0::2].clamp_min(0)
            * dipoles.pow(2).sum(dim=-1)
        ),
    )
    prediction.sum().backward()
    assert delta.energy_delta_head[-1].weight.grad is not None
    assert all(
        parameter.grad is None
        for module in (delta.encoder, delta.energy_head, delta.oscillator_head)
        for parameter in module.parameters()
    )


def test_qm9gwbse_dataset_loads_train_standardized_descriptors(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("torch_geometric")
    from gnn_excited.data.qm9gwbse import QM9GWBSEDataset

    molecular_path = tmp_path / "molecules.h5"
    with h5py.File(molecular_path, "w") as handle:
        molecule = handle.create_group("1")
        ground = molecule.create_group("ground_state")
        ground.create_dataset("labels", data=np.asarray([6, 1], dtype=np.int16))
        ground.create_dataset(
            "coords", data=np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.float32)
        )
        excited = molecule.create_group("excited_state")
        excited.create_dataset(
            "trans_dip_mom_D", data=np.zeros((5, 3), dtype=np.float32)
        )
    descriptor_path = tmp_path / "descriptors.h5"
    raw = np.arange(45, dtype=np.float32).reshape(5, 9)
    with h5py.File(descriptor_path, "w") as handle:
        handle.attrs["feature_names_json"] = '["energy", "oscillator", "dipole", "amplitude", "occupied", "virtual", "character", "gap", "ip"]'
        handle.attrs["train_mean"] = np.ones(9, dtype=np.float32)
        handle.attrs["train_std"] = np.full(9, 2.0, dtype=np.float32)
        handle.create_group("1").create_dataset("state_features", data=raw)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "molecule_key,qm9_id,status,S1_eV,log1p_S1_f\n"
        "1,1,ok,2.0,0.1\n",
        encoding="utf-8",
    )
    dataset = QM9GWBSEDataset(
        molecular_path,
        manifest_path,
        target_columns=("S1_eV", "log1p_S1_f"),
        electronic_descriptors_path=descriptor_path,
    )
    assert dataset[0].electronic_descriptors.shape == (5, 9)
    assert np.allclose(
        dataset[0].electronic_descriptors.numpy(), (raw - 1.0) / 2.0
    )
    selected = QM9GWBSEDataset(
        molecular_path,
        manifest_path,
        target_columns=("S1_eV", "log1p_S1_f"),
        electronic_descriptors_path=descriptor_path,
        electronic_descriptor_features=("energy", "gap"),
    )
    assert selected[0].electronic_descriptors.shape == (5, 2)
    assert np.allclose(
        selected[0].electronic_descriptors.numpy(), (raw[:, [0, 7]] - 1.0) / 2.0
    )
    with pytest.raises(ValueError, match="Unknown electronic descriptor features"):
        QM9GWBSEDataset(
            molecular_path,
            manifest_path,
            electronic_descriptors_path=descriptor_path,
            electronic_descriptor_features=("missing",),
        )
