from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch_geometric.data import Batch, Data
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    Batch = None
    Data = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ModuleNotFoundError as exc:  # pragma: no cover
    Chem = None
    AllChem = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None

from gnn_excited.models.dimenetpp import build_dimenetpp


@dataclass(frozen=True)
class MoleculeGeometry:
    smiles: str
    z: list[int]
    pos: list[list[float]]
    geometry_source: str


def smiles_to_geometry(smiles: str, seed: int = 17) -> MoleculeGeometry:
    """Generate one RDKit 3D conformer for a SMILES string."""
    if _RDKIT_IMPORT_ERROR is not None:
        raise ModuleNotFoundError("SMILES inference requires RDKit. Install the chem dependencies first.") from _RDKIT_IMPORT_ERROR
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    embed_code = AllChem.EmbedMolecule(mol, params)
    if embed_code != 0:
        raise RuntimeError(f"RDKit could not embed a conformer for SMILES {smiles!r}")

    geometry_source = "rdkit_etkdg"
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        geometry_source = "rdkit_etkdg_mmff"
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        geometry_source = "rdkit_etkdg_uff"

    conformer = mol.GetConformer()
    z = []
    pos = []
    for atom in mol.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        z.append(atom.GetAtomicNum())
        pos.append([float(point.x), float(point.y), float(point.z)])
    return MoleculeGeometry(canonical, z, pos, geometry_source)


def load_model(checkpoint_path: str | Path, device: str = "cpu"):
    if _TORCH_IMPORT_ERROR is not None:
        raise ModuleNotFoundError("Inference requires torch and torch_geometric.") from _TORCH_IMPORT_ERROR
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_dimenetpp(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_smiles(smiles: str, checkpoint_path: str | Path, device: str = "cpu", seed: int = 17) -> dict[str, Any]:
    """Predict S1 energy and oscillator strength from a SMILES string."""
    if _TORCH_IMPORT_ERROR is not None:
        raise ModuleNotFoundError("Inference requires torch and torch_geometric.") from _TORCH_IMPORT_ERROR
    geometry = smiles_to_geometry(smiles, seed=seed)
    model = load_model(checkpoint_path, device=device)
    data = Data(
        z=torch.tensor(geometry.z, dtype=torch.long),
        pos=torch.tensor(geometry.pos, dtype=torch.float32),
    )
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        pred = model(batch.z, batch.pos, batch.batch).view(-1, 2)[0]
    return {
        "smiles": geometry.smiles,
        "s1_energy_ev": float(pred[0].cpu()),
        "s1_oscillator_strength": float(torch.expm1(pred[1]).clamp_min(0).cpu()),
        "geometry_source": geometry.geometry_source,
    }
