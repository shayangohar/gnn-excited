"""Tests for the heavy-element multi-fidelity pieces (TASK-287) and the bet C helpers (TASK-276)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_element_counts_and_matched_sample():
    pytest.importorskip("h5py")
    from build_pubchemqc_aux import element_counts, matched_sample

    numbers = np.array([6, 1, 16, 6, 6, 1, 8, 1, 6, 6, 6, 1])
    offsets = np.array([0, 3, 7, 12])
    counts = element_counts(numbers, offsets, (1, 6, 16))
    assert counts[16].tolist() == [1, 0, 0] and counts[6].tolist() == [1, 2, 3]
    heavy_atoms = np.array([2, 2, 3, 2, 3, 3, 4])
    picked = matched_sample(heavy_atoms, np.array([0, 1, 2]), np.array([3, 4, 5, 6]), np.random.default_rng(0))
    assert sorted(heavy_atoms[picked].tolist()) == [2, 3]  # only one size-2 candidate exists
    assert set(picked.tolist()) <= {3, 4, 5}


def test_frontier_reads_homo_lumo_from_ascending_levels():
    from label_noise_diagnostic import frontier

    methane = np.array([-297.69, -23.52, -14.64, -14.64, -14.64, 2.30, 3.89, 3.89, 3.89, 6.82])
    f = frontier(methane, n_occ=5)
    assert f["homo"] == -14.64 and f["lumo"] == 2.30 and f["lumo3"] == 3.89 and f["lumo4"] == 6.82 and f["homo1"] == -14.64
    assert f["n_bound_virtual"] == 0


class _FakeBackbone:
    """Stands in for the UMA backbone: fixed node embeddings [N, 9, C] and the node->molecule map."""

    def __init__(self, torch, channels: int):
        self.torch, self.channels = torch, channels

    def __call__(self, batch):
        n = int(batch.natoms.sum())
        node_batch = self.torch.repeat_interleave(self.torch.arange(batch.natoms.numel()), batch.natoms)
        gen = self.torch.Generator().manual_seed(0)
        return {"node_embedding": self.torch.randn(n, 9, self.channels, generator=gen), "batch": node_batch}


def _model(torch, mode, aux_columns=("S1_eV", "S2_eV")):
    from train_uma_qm9gwbse import UMAExcitedState

    backbone = torch.nn.Module()
    fake = _FakeBackbone(torch, channels=4)
    backbone.forward = fake.__call__
    return UMAExcitedState(backbone, ("S1_eV", "log1p_S1_f", "S2_eV", "log1p_S2_f"), [5.0, 0.1, 6.0, 0.2], [1.0, 0.1, 1.0, 0.1],
                           embed_dim=4, head_hidden=8, readout="rich", readout_layers=0, lmax=2,
                           aux_columns=aux_columns, aux_mean=[3.0, 4.0], aux_std=[2.0, 2.0], aux_mode=mode)


def test_delta_mode_gw_energies_follow_the_aux_head():
    torch = pytest.importorskip("torch")
    batch = SimpleNamespace(natoms=torch.tensor([2, 3]))
    model = _model(torch, "delta")
    with torch.no_grad():
        gw, aux = model(batch), model(batch, fidelity="aux")
        assert torch.allclose(gw[:, [0, 2]], torch.tensor([[5.0, 6.0]] * 2))  # zero-init heads start at the GW mean
        assert torch.allclose(aux, torch.tensor([[3.0, 4.0]] * 2))
        model.aux_head.out[-1].bias.copy_(torch.tensor([0.5, -0.25]))
        shifted = model(batch)
    assert torch.allclose(shifted[:, [0, 2]] - gw[:, [0, 2]], torch.tensor([[1.0, -0.5]] * 2))
    assert torch.allclose(shifted[:, [1, 3]], gw[:, [1, 3]])  # oscillators do not depend on the aux head


def test_heads_mode_keeps_gw_independent_of_the_aux_head():
    torch = pytest.importorskip("torch")
    batch = SimpleNamespace(natoms=torch.tensor([2, 3]))
    model = _model(torch, "heads")
    with torch.no_grad():
        gw = model(batch)
        model.aux_head.out[-1].bias.copy_(torch.tensor([0.5, -0.25]))
        assert torch.allclose(model(batch), gw)
    with pytest.raises(ValueError):
        _model(torch, "delta", aux_columns=("S1_eV", "S3_eV"))
