from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


def test_smiles_invalid_error_when_rdkit_available() -> None:
    if importlib.util.find_spec('rdkit') is None:
        pytest.skip('RDKit is not installed')
    from gnn_excited.inference.smiles import smiles_to_geometry

    with pytest.raises(ValueError, match='Invalid SMILES'):
        smiles_to_geometry('not-a-smiles')


def test_pyg_dataset_one_sample_when_pyg_available(tmp_path: Path) -> None:
    if importlib.util.find_spec('torch_geometric') is None:
        pytest.skip('PyTorch Geometric is not installed')
    from tests.test_qcdge import _write_test_hdf5
    from gnn_excited.data.qcdge import build_manifest
    from gnn_excited.data.pyg_dataset import QCDGES1Dataset

    hdf5_path = tmp_path / 'sample.hdf5'
    manifest_path = tmp_path / 'manifest.csv'
    _write_test_hdf5(hdf5_path)
    build_manifest(hdf5_path, manifest_path, max_count=1)

    sample = QCDGES1Dataset(hdf5_path, manifest_path)[0]
    assert sample.z.shape[0] == 4
    assert sample.pos.shape == (4, 3)
    assert sample.y.shape == (1, 2)
    assert not hasattr(sample, 'excited_state')


def test_pyg_dataset_supports_multistate_targets_when_pyg_available(tmp_path: Path) -> None:
    if importlib.util.find_spec('torch_geometric') is None:
        pytest.skip('PyTorch Geometric is not installed')
    from tests.test_qcdge import _write_test_hdf5
    from gnn_excited.data.qcdge import build_manifest
    from gnn_excited.data.pyg_dataset import QCDGES1Dataset

    hdf5_path = tmp_path / 'sample.hdf5'
    manifest_path = tmp_path / 'manifest.csv'
    _write_test_hdf5(hdf5_path)
    build_manifest(hdf5_path, manifest_path, max_count=1, singlet_count=2)

    target_columns = ('S1_eV', 'log1p_S1_f', 'S2_eV', 'log1p_S2_f')
    sample = QCDGES1Dataset(hdf5_path, manifest_path, target_columns=target_columns)[0]
    assert sample.y.shape == (1, 4)
    assert sample.y[0, 0].item() == pytest.approx(3.7)
    assert sample.y[0, 1].item() == pytest.approx(math.log1p(0.02))
    assert sample.y[0, 2].item() == pytest.approx(4.1)
    assert sample.y[0, 3].item() == pytest.approx(math.log1p(0.12))


def test_split_head_target_indices_classify_energy_and_oscillator_targets() -> None:
    from gnn_excited.models.dimenetpp import _energy_oscillator_indices

    energy_indices, oscillator_indices = _energy_oscillator_indices(
        ('S1_eV', 'log1p_S1_f', 'S2_eV', 'log1p_S2_f'),
        out_channels=4,
    )

    assert energy_indices == [0, 2]
    assert oscillator_indices == [1, 3]


def test_shared_backbone_split_head_builds_split_output_blocks_when_pyg_available() -> None:
    pytest.importorskip('torch_geometric')
    torch = pytest.importorskip('torch')
    from gnn_excited.models.dimenetpp import (
        SharedBackboneSplitEnergyOscillatorDimeNetPlusPlus,
        SplitOutputPPBlock,
        build_dimenetpp,
    )

    model = build_dimenetpp(
        cutoff=3.0,
        num_blocks=1,
        hidden_channels=8,
        out_emb_channels=8,
        int_emb_size=4,
        basis_emb_size=4,
        num_radial=2,
        num_spherical=2,
        max_num_neighbors=8,
        out_channels=4,
        head_type='shared_split_energy_oscillator',
        target_columns=('S1_eV', 'log1p_S1_f', 'S2_eV', 'log1p_S2_f'),
    )

    assert isinstance(model, SharedBackboneSplitEnergyOscillatorDimeNetPlusPlus)
    assert not hasattr(model, 'energy_model')
    assert len(model.output_blocks) == 2
    assert all(isinstance(block, SplitOutputPPBlock) for block in model.output_blocks)
    assert model.energy_indices.tolist() == [0, 2]
    assert model.oscillator_indices.tolist() == [1, 3]

    x = torch.ones(3, 8)
    rbf = torch.ones(3, 2)
    i = torch.tensor([0, 0, 1])
    output = model.output_blocks[0](x, rbf, i, num_nodes=2)

    assert output.shape == (2, 4)
