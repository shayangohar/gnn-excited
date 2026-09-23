"""Parser tests for scripts/convert_qmsymex.py (pure python, no torch needed)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from convert_qmsymex import parse_qmsymex_file


def _example_lines():
    return [
        "2",
        "C2h|header|ignored",
        "C 0.0 0.0 0.0 0.1",
        "H 1.0 0.0 0.0 -0.1",
        "HOMO 5",
        "1|AU 5.0000 240.00 0.0100 0.000|4 6 0.70000|AU 3.0000 300.00 0.0000 2.000|4 6 0.60000",
        "2|BU 5.5000 230.00 0.0200 0.000|5 6 0.70000|BU 3.5000 290.00 0.0000 2.000|5 6 0.60000",
        "3|AU 6.0000 220.00 0.0300 0.000|4 7 0.70000|AU 4.0000 280.00 0.0000 2.000|4 7 0.60000",
        "4|BU 6.5000 210.00 0.0400 0.000|5 7 0.70000|BU 4.5000 270.00 0.0000 2.000|5 7 0.60000",
        "5|AU 7.0000 200.00 0.0500 0.000|4 8 0.70000|AU 5.0000 260.00 0.0000 2.000|4 8 0.60000",
        "6|BU 7.5000 190.00 0.0600 0.000|5 8 0.70000|BU 5.5000 250.00 0.0000 2.000|5 8 0.60000",
        "7|AU 8.0000 180.00 0.0700 0.000|4 9 0.70000|AU 6.0000 240.00 0.0000 2.000|4 9 0.60000",
        "8|BU 8.5000 170.00 0.0800 0.000|5 9 0.70000|BU 6.5000 230.00 0.0000 2.000|5 9 0.60000",
        "9|AU 9.0000 160.00 0.0900 0.000|4 10 0.70000|AU 7.0000 220.00 0.0000 2.000|4 10 0.60000",
        "10|BU 9.5000 150.00 0.1000 0.000|5 10 0.70000|BU 7.5000 210.00 0.0000 2.000|5 10 0.60000",
    ]


def test_parse_qmsymex_states(tmp_path):
    path = tmp_path / "QM_symex_000001.xyz"
    path.write_text("\n".join(_example_lines()) + "\n", encoding="utf-8")
    numbers, positions, S_eV, T_eV, S_f = parse_qmsymex_file(path)
    assert numbers.tolist() == [6, 1]
    assert positions.shape == (2, 3)
    assert [round(v, 4) for v in S_eV] == [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
    assert [round(v, 4) for v in T_eV] == [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    assert [round(v, 4) for v in S_f] == [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]


def test_parse_qmsymex_rejects_bad_file(tmp_path):
    path = tmp_path / "QM_symex_000002.xyz"
    path.write_text("not-an-xyz\n", encoding="utf-8")
    assert parse_qmsymex_file(path) is None


def test_qmsymex_dataset_reads_h5_rows(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    from gnn_excited.data.qmsymex import QMSymexSTDataset

    xyz = tmp_path / "QM_symex_000001.xyz"
    xyz.write_text("\n".join(_example_lines()) + "\n", encoding="utf-8")
    numbers, positions, S_eV, T_eV, S_f = parse_qmsymex_file(xyz)
    h5_path = tmp_path / "train.h5"
    with h5py.File(h5_path, "w") as h5:
        root = h5.create_group("m")
        grp = root.create_group("QM_symex_000001")
        grp.create_dataset("numbers", data=numbers)
        grp.create_dataset("positions", data=positions)
        grp.create_dataset("S_eV", data=S_eV)
        grp.create_dataset("T_eV", data=T_eV)
        grp.create_dataset("S_f", data=S_f)
    manifest = tmp_path / "manifest.csv"
    cols = (["molecule_key", "natoms"]
            + [f"S{s}_eV" for s in range(1, 11)]
            + [f"T{s}_eV" for s in range(1, 11)]
            + [f"S{s}_f" for s in range(1, 11)]
            + ["split", "status"])
    manifest.write_text(
        ",".join(cols) + "\n" + ",".join(
            ["QM_symex_000001", "2"]
            + [f"{v:.6f}" for v in S_eV]
            + [f"{v:.6f}" for v in T_eV]
            + [f"{v:.6f}" for v in S_f]
            + ["train", "ok"]) + "\n",
        encoding="utf-8",
    )
    ds = QMSymexSTDataset(h5_path, manifest)
    assert ds.len() == 1
    data = ds.get(0)
    assert data.z.tolist() == [6, 1]
    assert data.y.shape == (1, 10)
    assert float(data.y[0, 0]) == pytest.approx(5.0)
    assert float(data.y[0, 5]) == pytest.approx(3.0)
