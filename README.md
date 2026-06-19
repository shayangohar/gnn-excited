# gnn-excited

`gnn-excited` is a research codebase for predicting excited-state properties of small organic molecules with 3D graph neural networks. The initial model target is the lowest singlet excitation, using QCDGE molecular geometries and a DimeNet++ architecture.

The long-term goal is to explore whether learned molecular models can provide fast approximations to TD-DFT-style excited-state calculations for early-stage screening and analysis.

## Scope

This repository currently focuses on the QCDGE `A_9` subset, which contains QM9-like molecules with fewer than 10 heavy atoms. The first prediction targets are:

- `S1_eV`: lowest singlet excitation energy in electronvolts
- `S1_f`: oscillator strength for the same lowest singlet transition

Training uses the ground-state molecular geometry from QCDGE:

- atomic numbers from `ground_state/labels`
- 3D coordinates from `ground_state/coords`

Excited-state fields are used only as supervised learning targets, not as model inputs.

## Model Interface

The public-facing inference interface is a SMILES string. Since DimeNet++ is a 3D molecular graph model, the SMILES string is converted into an approximate 3D conformer before prediction:

1. canonicalize the SMILES string with RDKit
2. add hydrogens
3. generate a 3D conformer with ETKDG
4. optimize with MMFF when available, falling back to UFF
5. pass atomic numbers and coordinates to DimeNet++

This introduces an important modeling caveat: QCDGE training geometries are quantum-chemistry-derived ground-state geometries, while SMILES inference uses RDKit-generated conformers. This geometry mismatch should be evaluated carefully before interpreting predictions quantitatively.

## Repository Structure

```text
gnn-excited/
├── data/               # local datasets and processed manifests; gitignored
├── notebooks/          # exploratory notebooks
├── src/gnn_excited/    # parser, dataset, model, training, and inference code
├── scripts/            # command-line entrypoints
├── configs/            # training configuration files
├── tests/              # parser, dataset, and inference tests
├── environment.yml
├── pyproject.toml
└── README.md
```

Dataset files and generated caches are intentionally excluded from version control.

## Installation

Create the project environment:

```powershell
conda env create -f environment.yml
conda activate gnn-excited
pip install -e .
```

The environment file provides a CPU-compatible development setup. For GPU training, install CUDA-compatible PyTorch and PyTorch Geometric packages appropriate for the target system.

## Data Preparation

Place the QCDGE HDF5 file under `data/`:

```text
data/A_9.hdf5
```

Then build a processed manifest:

```powershell
python scripts/build_manifest.py --hdf5 data/A_9.hdf5 --out data/processed/a9_manifest_1000.csv --max-count 1000
```

The manifest records molecule key, atom count, target values, parse status, and any parse errors. It is used to validate data extraction before training.

Inspect target distributions:

```powershell
python scripts/inspect_targets.py data/processed/a9_manifest_1000.csv
```

## Training

Run a small CPU training job:

```powershell
python scripts/train_dimenet.py --config configs/small_cpu.yaml
```

Run a tiny overfit check:

```powershell
python scripts/train_dimenet.py --config configs/overfit_16_cpu.yaml
```

The overfit check is intended to verify the data parser, PyTorch Geometric batching, DimeNet++ forward/backward pass, optimizer, loss calculation, and checkpoint writing.

## SMILES Prediction

After training creates a checkpoint:

```powershell
python scripts/predict_smiles.py "CCO" --checkpoint checkpoints/dimenetpp_a9_small.pt
```

Example output:

```json
{
  "smiles": "CCO",
  "s1_energy_ev": 0.0,
  "s1_oscillator_strength": 0.0,
  "geometry_source": "rdkit_etkdg_mmff"
}
```

## Development Status

The current implementation includes:

- lazy parsing of QCDGE HDF5 molecules
- extraction of lowest singlet excitation energy and oscillator strength
- processed manifest generation
- a PyTorch Geometric dataset layer
- a DimeNet++ training script
- RDKit-based SMILES-to-geometry inference
- parser, dataset, and inference smoke tests

The project is in an early validation phase. Current results should be treated as software and workflow validation, not as a benchmarked scientific model.

## Research Directions

Planned extensions include:

- training and evaluating larger DimeNet++ models on expanded QCDGE subsets
- quantifying the effect of RDKit-generated geometries versus quantum-chemistry geometries
- adding dataset SMILES alignment when corresponding QCDGE metadata is available
- comparing direct prediction against delta machine learning approaches
- evaluating uncertainty, calibration, and chemical-domain generalization

## License

License information has not yet been specified.
