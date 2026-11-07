"""Geometries from SMILES and their comparison with reference geometries (TASK-280 / TASK-291 E3).

embed: RDKit ETKDGv3 conformer, then an MMFF94 or UFF minimisation (the rCEG recompute protocol is
ETKDGv3 + UFF). xtb_optimize: GFN2-xTB optimisation through the xtb binary. heavy_atom_rmsd: best-fit
heavy-atom RMSD between two geometries of the same molecule, matched on the distance-based
connectivity graph so bond-order perception cannot break the match.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def charge_and_multiplicity(smiles: str) -> tuple[int, int] | None:
    """Formal charge and spin multiplicity (radical electrons + 1) from a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.GetFormalCharge(mol), Descriptors.NumRadicalElectrons(mol) + 1


def embed(smiles: str, seed: int = 0, forcefield: str = "MMFF", max_iters: int = 500) -> tuple[np.ndarray, np.ndarray] | None:
    """(atomic numbers, positions in Angstrom) of one ETKDGv3 conformer after force-field minimisation."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        if forcefield.upper() == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception:
        pass  # keep the ETKDG geometry if the force field cannot be set up
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.uint8)
    return numbers, np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)


def xtb_optimize(numbers: np.ndarray, positions: np.ndarray, xtb: str, charge: int = 0, uhf: int = 0,
                 level: str = "normal", timeout: int = 600) -> np.ndarray | None:
    """GFN2-xTB optimised positions, or None if xtb fails or does not converge."""
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    workdir = tempfile.mkdtemp(prefix="xtb_", dir=os.environ.get("TMPDIR"))
    try:
        lines = [str(len(numbers)), ""] + [f"{table.GetElementSymbol(int(z))} {x:.8f} {y:.8f} {w:.8f}"
                                           for z, (x, y, w) in zip(numbers, positions)]
        Path(workdir, "in.xyz").write_text("\n".join(lines) + "\n")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"}
        result = subprocess.run([xtb, "in.xyz", "--gfn", "2", "--opt", level, "--chrg", str(charge), "--uhf", str(uhf)],
                                cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
        out = Path(workdir, "xtbopt.xyz")
        if result.returncode != 0 or not out.exists() or "GEOMETRY OPTIMIZATION CONVERGED" not in result.stdout:
            return None
        rows = [line.split() for line in out.read_text().splitlines()[2:2 + len(numbers)]]
        return np.array([[float(v) for v in r[1:4]] for r in rows], dtype=np.float64)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _connectivity_mol(numbers: np.ndarray, positions: np.ndarray):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    table = Chem.GetPeriodicTable()
    block = f"{len(numbers)}\n\n" + "\n".join(f"{table.GetElementSymbol(int(z))} {x:.6f} {y:.6f} {w:.6f}"
                                              for z, (x, y, w) in zip(numbers, positions))
    mol = Chem.MolFromXYZBlock(block)
    rdDetermineBonds.DetermineConnectivity(mol)
    mol.UpdatePropertyCache(strict=False)
    return Chem.RemoveHs(mol, sanitize=False)


def heavy_atom_rmsd(numbers_a, positions_a, numbers_b, positions_b) -> float | None:
    """Symmetry-aware best-fit heavy-atom RMSD (Angstrom); None if the two connectivity graphs differ."""
    from rdkit.Chem import rdMolAlign

    try:
        return float(rdMolAlign.GetBestRMS(_connectivity_mol(numbers_a, positions_a), _connectivity_mol(numbers_b, positions_b)))
    except Exception:
        return None


def pack(records: list[tuple[str, np.ndarray, np.ndarray]], path: str | Path, charge=None, spin=None) -> None:
    """Write (key, numbers, positions) records as a pool .npz (scripts/train_uma_qm9gwbse.py, predict_packed_visnet.py)."""
    counts = np.array([r[1].size for r in records], dtype=np.int64)
    arrays = {"numbers": np.concatenate([r[1] for r in records]).astype(np.uint8),
              "positions": np.concatenate([r[2] for r in records]).astype(np.float32),
              "atom_offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
              "molecule_key": np.array([r[0] for r in records])}
    if charge is not None:
        arrays["charge"], arrays["spin"] = np.asarray(charge, dtype=np.int64), np.asarray(spin, dtype=np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
