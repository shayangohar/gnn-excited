"""Tests for scripts/pack_pubchemqc.py and the packed PubChemQC dataset (TASK-217)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pack_pubchemqc import (  # noqa: E402
    ENERGY_COLUMNS,
    connectivity_block,
    exclusion_rows,
    read_chunk,
    write_packed,
)

HEADER = ["molecule_key", "inchikey", "smiles", "natoms", *ENERGY_COLUMNS, "split", "status"]


def _write_chunk(directory: Path, name: str, molecules: list[tuple[str, list[int], float]]) -> tuple[str, str]:
    """Write a chunk in the extract_pubchemqc.py layout; S_k = base + 0.1 k."""
    h5_path = directory / f"train_st_{name}.h5"
    manifest_path = directory / f"train_st_{name}_manifest.csv"
    with h5py.File(h5_path, "w") as h5, manifest_path.open("w", newline="", encoding="utf-8") as stream:
        root = h5.create_group("m")
        root.attrs["schema"] = "pubchemqc-st-v1"
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        for key, numbers, base in molecules:
            energies = base + 0.1 * np.arange(10)
            positions = np.arange(3 * len(numbers), dtype=np.float64).reshape(-1, 3) * 0.5
            group = root.create_group(key)
            group.create_dataset("numbers", data=np.asarray(numbers, dtype=np.int32))
            group.create_dataset("positions", data=positions)
            group.create_dataset("S_eV", data=energies)
            writer.writerow([key, f"IK{key.upper()}-AAAA-N", "C", len(numbers), *[f"{v:.6f}" for v in energies], "train", "ok"])
    return str(h5_path), str(manifest_path)


def _pack(tmp_path: Path) -> tuple[Path, list[dict]]:
    chunks = [
        _write_chunk(tmp_path, "c0", [("pqc_1", [6, 1, 1, 1, 1], 3.0), ("pqc_2", [8, 1, 1], 6.0)]),
        _write_chunk(tmp_path, "c1", [("pqc_3", [7, 1, 1, 1], 5.0)]),
    ]
    parts = [read_chunk(h5_path, manifest) for h5_path, manifest in chunks]
    out = tmp_path / "packed.h5"
    write_packed(out, parts, ["c0", "c1"])
    return out, parts


def test_pack_preserves_molecules_and_offsets(tmp_path):
    out, parts = _pack(tmp_path)
    with h5py.File(out, "r") as h5:
        assert h5.attrs["schema"] == "pubchemqc-st-packed-v1"
        offsets = h5["atom_offsets"][()]
        assert offsets.tolist() == [0, 5, 8, 12]
        assert h5["numbers"][5:8].tolist() == [8, 1, 1]
        assert h5["positions"].dtype == np.float32
        np.testing.assert_allclose(h5["positions"][8:12], np.arange(12).reshape(4, 3) * 0.5)
        np.testing.assert_allclose(h5["S_eV"][2], 5.0 + 0.1 * np.arange(10), atol=1e-6)
        assert [k.decode() for k in h5["molecule_key"][()]] == ["pqc_1", "pqc_2", "pqc_3"]


def test_read_chunk_rejects_manifest_drift(tmp_path):
    h5_path, manifest = _write_chunk(tmp_path, "c0", [("pqc_1", [6, 1], 3.0)])
    with h5py.File(h5_path, "r+") as h5:
        h5["m/pqc_1/S_eV"][0] = 9.0
    with pytest.raises(ValueError, match="drift"):
        read_chunk(h5_path, manifest)


def test_exclusions_match_on_connectivity_block_only():
    blocks = {"VNWKTOKETHGBQD": [("1", "test")]}
    rows = exclusion_rows(
        ["pqc_1", "pqc_2", "pqc_3"],
        ["VNWKTOKETHGBQD-UHFFFAOYSA-N", "VNWKTOKETHGBQD-UHFFFAOYSA-O", "XLYOFNOQVPJJNP-UHFFFAOYSA-N"],
        blocks,
    )
    assert [row["molecule_key"] for row in rows] == ["pqc_1", "pqc_2"]
    assert rows[0]["high_level_splits"] == "test"
    assert connectivity_block("  ") == ""


def test_holdout_blocks_use_rdkit_inchikeys(tmp_path):
    pytest.importorskip("rdkit")
    from pack_pubchemqc import qm9gwbse_holdout_blocks

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "molecule_key,qm9_id,canonical_smiles,status\n1,1,C,ok\n2,2,O,ok\n3,3,not_smiles,ok\n",
        encoding="utf-8",
    )
    splits = tmp_path / "splits.csv"
    splits.write_text("molecule_key,random_split\n1,test\n2,train\n3,val\n", encoding="utf-8")
    blocks, stats = qm9gwbse_holdout_blocks(manifest, splits)
    assert set(blocks) == {"VNWKTOKETHGBQD"}  # methane; water is train-only
    assert stats == {"holdout_molecules": 2, "holdout_unparsed": 1, "holdout_blocks": 1}


def test_packed_dataset_returns_requested_states(tmp_path):
    pytest.importorskip("torch_geometric")
    from gnn_excited.data.pubchemqc import PubChemQCPackedDataset, state_column_indices

    out, _ = _pack(tmp_path)
    dataset = PubChemQCPackedDataset(out, tmp_path / "unused.csv", ["pqc_3", "pqc_1"], ("S1_eV", "S5_eV"))
    assert len(dataset) == 2
    first = dataset[0]
    assert first.molecule_key == "pqc_3"
    assert first.z.tolist() == [7, 1, 1, 1]
    assert first.pos.shape == (4, 3)
    assert first.y.shape == (1, 2)
    assert first.y[0].tolist() == pytest.approx([5.0, 5.4], abs=1e-6)
    with pytest.raises(ValueError, match="singlets|S1_eV"):
        state_column_indices(("T1_eV",))
    with pytest.raises(ValueError, match="missing"):
        PubChemQCPackedDataset(out, tmp_path / "unused.csv", ["pqc_404"], ("S1_eV",))


def test_encoder_energy_finetune_keeps_energy_head_and_resets_oscillator():
    pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    from gnn_excited.models.visnet import build_visnet, load_transfer_checkpoint

    common = {"hidden_channels": 32, "num_layers": 1, "num_rbf": 8, "cutoff": 3.0, "max_num_neighbors": 8}
    energy_columns = tuple(f"S{k}_eV" for k in range(1, 6))
    paired_columns = tuple(c for k in range(1, 6) for c in (f"S{k}_eV", f"log1p_S{k}_f"))
    torch.manual_seed(0)
    donor = build_visnet(target_columns=energy_columns, **common)
    torch.manual_seed(1)
    model = build_visnet(target_columns=paired_columns, **common)
    fresh_oscillator = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("oscillator_head.")}
    load_transfer_checkpoint(model, {"model_state_dict": donor.state_dict()}, mode="encoder_energy_finetune")
    state = model.state_dict()
    for key, value in donor.state_dict().items():
        if key.startswith(("encoder.", "energy_head.")):
            assert torch.equal(state[key], value), key
    for key, value in fresh_oscillator.items():
        assert torch.equal(state[key], value), key
    assert all(p.requires_grad for p in model.parameters())

    mismatched = build_visnet(target_columns=("S1_eV", "S2_eV", "log1p_S1_f", "log1p_S2_f"), **common)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_transfer_checkpoint(mismatched, {"model_state_dict": donor.state_dict()}, mode="encoder_energy_finetune")
