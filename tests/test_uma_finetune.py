"""UMA finetune pieces that run without fairchem (scripts/train_uma_qm9gwbse.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_uma_qm9gwbse import (  # noqa: E402
    UMAExcitedState,
    average_state_dicts,
    backbone_param_groups,
    lr_lambda_factory,
    spherical_invariants,
)


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


class FakeBlockBackbone(torch.nn.Module):
    """Two 'blocks' producing [N, 9, C] spherical features; the second feeds node_embedding."""

    def __init__(self, channels: int = 8):
        super().__init__()
        self.embed = torch.nn.Linear(1, channels)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(channels, channels), torch.nn.Linear(channels, channels)])
        self.norm = torch.nn.LayerNorm(channels)

    def forward(self, batch):
        x = self.embed(batch.atomic_numbers.float().unsqueeze(1)).unsqueeze(1).repeat(1, 9, 1)
        for block in self.blocks:
            x = torch.tanh(block(x))
        return {"node_embedding": self.norm(x), "batch": batch.batch}


def test_spherical_invariants_take_l0_and_per_l_norms():
    x = torch.zeros(2, 9, 3)
    x[:, 0, :] = 1.0
    x[:, 1:4, :] = 2.0
    x[:, 4:9, :] = 1.0
    out = spherical_invariants(x)
    assert out.shape == (2, 9)
    assert torch.allclose(out[0], torch.tensor([1.0] * 3 + [12 ** 0.5] * 3 + [5 ** 0.5] * 3))
    with pytest.raises(ValueError):
        spherical_invariants(torch.zeros(1, 5, 3))


def test_rich_readout_starts_at_mean_and_is_atom_order_invariant():
    columns = ("S1_eV", "log1p_S1_f", "S2_eV", "log1p_S2_f")
    mean, std = [5.0, 0.1, 6.0, 0.2], [1.0, 0.05, 1.5, 0.1]
    model = UMAExcitedState(FakeBlockBackbone(), columns, mean, std, embed_dim=8, head_hidden=16,
                            readout="rich", readout_layers=1)
    out = model(_batch())
    assert torch.allclose(out, torch.tensor([mean, mean]))
    with torch.no_grad():
        for p in model.energy_head.out[-1].parameters():
            p.normal_()
    perm = SimpleNamespace(atomic_numbers=torch.tensor([1, 6, 1, 1, 8]), batch=torch.tensor([0, 0, 0, 1, 1]),
                           natoms=torch.tensor([3, 2]))
    assert torch.allclose(model(_batch()), model(perm), atol=1e-6)


def test_rich_readout_hooks_survive_ema_deepcopy():
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    model = UMAExcitedState(FakeBlockBackbone(), ("S1_eV",), [1.0], [1.0], embed_dim=8, head_hidden=16,
                            readout="rich", readout_layers=1)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    ema.update_parameters(model)
    assert ema.module(_batch()).shape == (2, 1)
    assert model(_batch()).shape == (2, 1)


def test_layer_lr_decay_orders_groups_by_depth():
    backbone = FakeBlockBackbone()
    groups = {g["name"]: g["lr"] for g in backbone_param_groups(backbone, 1.0, 0.0, 0.5)}
    # embed -> depth 0, blocks.0 -> 1, blocks.1 -> 2, norm -> top (3)
    assert groups == {"backbone_depth0": 0.125, "backbone_depth1": 0.25, "backbone_depth2": 0.5, "backbone_depth3": 1.0}
    single = backbone_param_groups(backbone, 2e-4, 0.0, None)
    assert len(single) == 1 and single[0]["lr"] == 2e-4


def test_average_state_dicts_weights_floats_and_keeps_others():
    a = {"w": torch.tensor([0.0, 2.0]), "n": torch.tensor(3)}
    b = {"w": torch.tensor([2.0, 4.0]), "n": torch.tensor(5)}
    soup = average_state_dicts([a, b])
    assert torch.allclose(soup["w"], torch.tensor([1.0, 3.0])) and int(soup["n"]) == 3
    assert torch.allclose(average_state_dicts([a, b], [3, 1])["w"], torch.tensor([0.5, 2.5]))
