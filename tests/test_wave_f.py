"""Tests for the wave F pieces (TASK-310): production spec, xTB packing, few-shot folds, bet A decoders."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def test_production_spec_loads_and_weights_sum_to_one():
    pytest.importorskip("yaml")
    from head_to_head_report import load_production

    singlets = load_production(ROOT / "configs/production.yaml", "singlets")
    triplets = load_production(ROOT / "configs/production.yaml", "triplets")
    assert [f["family"] for f in singlets] == ["he_heads", "st_he_heads", "distill", "he_delta"]
    assert [f["weight"] for f in singlets] == [0.40, 0.10, 0.35, 0.15]
    assert {f["family"] for f in triplets} == {"uma_richind_st", "st_he_heads"}
    assert all("{seed}" in f["test"] for f in singlets + triplets)
    assert all("{seed}" in f["external"] for f in singlets)


def test_load_production_rejects_bad_weights(tmp_path):
    pytest.importorskip("yaml")
    from head_to_head_report import load_production

    spec = tmp_path / "p.yaml"
    spec.write_text("singlets:\n- {family: a, weight: 0.5, seeds: [1], test: x}\n- {family: b, weight: 0.4, seeds: [1], test: y}\n")
    with pytest.raises(ValueError, match="sum"):
        load_production(spec)


def test_family_predictions_fills_seed(tmp_path):
    from head_to_head_report import family_predictions

    for s in (1, 2):
        (tmp_path / f"m_s{s}.csv").write_text(f"molecule_key,prediction_S1_eV\nA,{s}.5\n")
    preds = family_predictions({"seeds": [1, 2], "test": str(tmp_path / "m_s{seed}.csv")}, "test", ["prediction_S1_eV"])
    assert [p["A"][0] for p in preds] == [1.5, 2.5]


def test_xtb_assemble_keeps_order_and_replaces_geometry():
    from build_xtb_packed import assemble, per_molecule

    arrays = {"numbers": np.array([6, 1, 8, 7, 1]), "positions": np.zeros((5, 3)),
              "atom_offsets": np.array([0, 2, 3, 5]), "molecule_key": np.array(["a", "b", "c"]),
              "split": np.array(["train", "val", "test"]), "targets": np.arange(9.0).reshape(3, 3),
              "target_columns": np.array(["S1_eV", "S2_eV", "S3_eV"])}
    built = {2: (np.array([7, 1]), np.ones((2, 3))), 0: (np.array([1, 6]), np.full((2, 3), 2.0))}
    out = assemble(arrays, built)
    assert out["molecule_key"].tolist() == ["a", "c"] and out["split"].tolist() == ["train", "test"]
    assert out["source_row"].tolist() == [0, 2] and out["targets"][:, 0].tolist() == [0.0, 6.0]
    assert out["numbers"].tolist() == [1, 6, 7, 1] and out["atom_offsets"].tolist() == [0, 2, 4]
    assert out["target_columns"].tolist() == ["S1_eV", "S2_eV", "S3_eV"]  # never subset, even if 3 molecules
    assert not per_molecule("target_columns", arrays["target_columns"], 3)


def test_fewshot_folds_are_disjoint_and_shots_nested():
    from build_oe62h_shots import fold_assignment, nested_shots, run_config

    keys = [f"m{i:03d}" for i in range(103)]
    folds = fold_assignment(keys, 5, seed=0)
    assert sorted(k for f in folds for k in f) == keys and all(20 <= len(f) <= 21 for f in folds)
    pool = sorted(set(keys) - set(folds[0]))
    shots = nested_shots(pool, [5, 10, 40], seed=1)
    assert shots[10][:5] == shots[5] and shots[40][:10] == shots[10]
    assert not set(shots[40]) & set(folds[0])
    with pytest.raises(ValueError):
        nested_shots(pool, [len(pool) + 1], seed=1)
    template = {"training": {}, "wandb": {"name": "x"}, "post_inference": {"out_csv": "x"}, "shots": {"packed_path": "S"}}
    control = run_config(template, "k0", None, "runs/f")
    assert "shots" not in control and control["post_inference"]["out_csv"] == "runs/f/ext_k0.csv"
    run = run_config(template, "f1_k25", "d/fold1_k25.npz", "runs/f")
    assert run["shots"]["packed_path"] == "d/fold1_k25.npz" and template["shots"]["packed_path"] == "S"


def test_symmetric_matrices_and_decoder_initialisation():
    torch = pytest.importorskip("torch")
    from gnn_excited.models.spectral_decoders import SpectralDecoder, symmetric_from_parameters

    diag = torch.tensor([[1.0, 2.0, 3.0]])
    tri = symmetric_from_parameters(diag, torch.tensor([[0.1, 0.2]]), "tridiagonal")
    assert torch.equal(tri, tri.transpose(1, 2)) and tri[0, 0, 2] == 0 and tri[0, 1, 2] == 0.2
    full = symmetric_from_parameters(diag, torch.tensor([[0.1, 0.2, 0.3]]), "full")
    assert full[0, 0, 2] == 0.2 and full[0, 2, 1] == 0.3
    means, stds = [3.0, 4.0, 5.0, 6.0, 7.0], [0.5] * 5
    atoms, node_batch = torch.randn(7, 12), torch.tensor([0, 0, 0, 1, 1, 2, 2])
    for kind in ("mlp", "r0", "r1"):
        model = SpectralDecoder(12, means, stds, [0.1] * 5, [0.05] * 5, kind=kind, hidden=16, configs=9)
        energies, osc = model(atoms, node_batch, 3)
        assert energies.shape == (3, 5) and osc.shape == (3, 5)
        assert torch.allclose(energies, torch.tensor(means).expand(3, 5), atol=1e-5), kind
        assert torch.allclose(osc, torch.full((3, 5), 0.1))
        energies.sum().backward()  # gradients reach the energy head through eigvalsh
        assert model.energy_head[-1].weight.grad is not None
    with pytest.raises(ValueError):
        SpectralDecoder(12, means, stds, [0.1] * 5, [0.05] * 5, kind="r1", configs=3)


def test_gather_atom_indices():
    torch = pytest.importorskip("torch")
    from train_frozen_decoder import energy_metrics, gather

    atoms, node_batch = gather(torch.tensor([10, 3]), torch.tensor([2, 3]))
    assert atoms.tolist() == [10, 11, 3, 4, 5] and node_batch.tolist() == [0, 0, 1, 1, 1]
    target = np.array([[3.0, 3.05, 4.0, 5.0, 6.0]])
    m = energy_metrics(np.array([[3.1, 3.0, 4.0, 5.0, 6.0]]), target)
    assert m["near_degenerate_pairs"] == 1 and m["adjacent_inversion_rate"] == 0.25
    assert np.isclose(m["S_agg_sorted"], (0.0 + 0.05) / 5)
