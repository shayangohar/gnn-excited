"""UMA finetune pieces that run without fairchem (scripts/train_uma_qm9gwbse.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_uma_qm9gwbse import UMAExcitedState, lr_lambda_factory  # noqa: E402


class FakeBackbone(torch.nn.Module):
    """Mimics eSCNMDMoeBackbone output: node_embedding [N, 9, C] and node->molecule batch."""

    def __init__(self, channels: int = 8):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.channels = channels

    def forward(self, batch):
        n = batch.atomic_numbers.numel()
        scalars = batch.atomic_numbers.float().unsqueeze(1).repeat(1, self.channels) * self.scale
        embedding = torch.zeros(n, 9, self.channels)
        embedding[:, 0, :] = scalars
        return {"node_embedding": embedding, "batch": batch.batch}


def _batch():
    return SimpleNamespace(
        atomic_numbers=torch.tensor([6, 1, 1, 8, 1]),
        batch=torch.tensor([0, 0, 0, 1, 1]),
        natoms=torch.tensor([3, 2]),
    )


def test_model_starts_at_target_mean_with_paired_layout():
    columns = ("S1_eV", "log1p_S1_f", "S2_eV", "log1p_S2_f")
    mean, std = [5.0, 0.1, 6.0, 0.2], [1.0, 0.05, 1.5, 0.1]
    model = UMAExcitedState(FakeBackbone(), columns, mean, std, embed_dim=8, head_hidden=16)
    out = model(_batch())
    assert out.shape == (2, 4)
    assert torch.allclose(out, torch.tensor([mean, mean]))
    out.sum().backward()
    assert model.energy_head[-1].weight.grad is not None
    assert model.oscillator_head[-1].weight.grad is not None


def test_pooling_is_per_molecule_mean_and_columns_land_in_place():
    columns = ("S1_eV", "log1p_S1_f")
    model = UMAExcitedState(FakeBackbone(), columns, [0.0, 0.0], [1.0, 1.0], embed_dim=8, head_hidden=16)
    with torch.no_grad():
        for head, value in ((model.energy_head, 1.0), (model.oscillator_head, -1.0)):
            head[-1].bias.fill_(value)
    out = model(_batch())
    assert torch.allclose(out, torch.tensor([[1.0, -1.0], [1.0, -1.0]]))


def test_energy_only_layout_has_no_oscillator_head():
    model = UMAExcitedState(FakeBackbone(), ("S1_eV", "S2_eV"), [1.0, 2.0], [1.0, 1.0], embed_dim=8, head_hidden=16)
    assert model.oscillator_head is None
    assert model(_batch()).shape == (2, 2)


def test_lr_schedule_warms_up_then_decays_to_min():
    factor = lr_lambda_factory(total_steps=100, warmup_steps=10, start_factor=0.01, min_factor=0.05)
    assert factor(0) == pytest.approx(0.01)
    assert factor(10) == pytest.approx(1.0)
    assert factor(55) == pytest.approx(0.05 + 0.95 * 0.5)
    assert factor(100) == pytest.approx(0.05)
    assert factor(500) == pytest.approx(0.05)
