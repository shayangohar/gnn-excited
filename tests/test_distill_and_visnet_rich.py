"""Tests for the distillation helpers and the ViSNet rich residual readout (TASK-257)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_distill_dataset import teacher_labels  # noqa: E402
from build_distill_pool import select_pool  # noqa: E402


def test_select_pool_filters_elements_size_and_exclusions():
    # mol0 CH4 ok; mol1 has S (16) -> out; mol2 three heavy atoms -> out at max_heavy=2; mol3 excluded
    numbers = np.array([6, 1, 1, 1, 1, 16, 1, 1, 6, 6, 8, 7, 1])
    offsets = np.array([0, 5, 8, 11, 13])
    keys = np.array(["a", "b", "c", "d"])
    keep, heavy = select_pool(numbers, offsets, keys, {1, 6, 7, 8, 9}, max_heavy=2, excluded={"d"})
    assert keep.tolist() == [True, False, False, False]
    assert heavy.tolist() == [1, 1, 3, 1]


def test_teacher_labels_weight_families_sort_and_spread():
    uma = [np.array([[3.0, 2.0]]), np.array([[3.2, 2.2]])]
    vis = [np.array([[2.6, 2.4]])]
    label, sigma = teacher_labels([uma, vis], [0.5, 0.5])
    assert label[0].tolist() == sorted(label[0].tolist())
    assert label[0, 1] == pytest.approx(0.5 * 3.1 + 0.5 * 2.6)
    assert sigma[0, 0] == pytest.approx(np.std([3.0, 3.2, 2.6]))


def test_visnet_rich_readout_starts_unchanged_and_loads_energy_transfer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gnn_excited.models.visnet import build_visnet, load_transfer_checkpoint

    common = dict(target_columns=("S1_eV", "log1p_S1_f", "S2_eV", "log1p_S2_f"), hidden_channels=16, num_layers=1,
                  num_rbf=8, cutoff=5.0, max_num_neighbors=16, reduce_op="mean")
    torch.manual_seed(0)
    base = build_visnet(**common)
    rich = build_visnet(**common, rich_readout=True)
    assert hasattr(rich, "rich_energy_head") and not hasattr(base, "rich_energy_head")
    load_transfer_checkpoint(rich, {"model_state_dict": base.state_dict()}, mode="encoder_energy_finetune")
    rich.oscillator_head.load_state_dict(base.oscillator_head.state_dict())
    z = torch.tensor([6, 1, 1, 1, 8, 1])
    pos = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    with torch.no_grad():
        assert torch.allclose(rich(z, pos, batch), base(z, pos, batch), atol=1e-6)
