# gnn-excited

DimeNet++ models for predicting TD/DFT-style excited-state properties from QCDGE molecular geometries, with a SMILES-facing inference path.

## Current Scope

This project starts with the QCDGE `A_9.hdf5` subset: QM9 molecules with fewer than 10 heavy atoms. Dataset files are intentionally kept under `data/` and ignored by git.

The v1 model predicts:

- `S1_eV` - lowest singlet excitation energy in eV
- `S1_f` - oscillator strength for that same lowest singlet

Training uses QCDGE ground-state atom labels and coordinates as DimeNet++ inputs. Inference accepts a SMILES string, generates one RDKit 3D conformer, and feeds atom types plus coordinates to the trained model.

## Current Status

The CPU validation pipeline is working end-to-end:

- `A_9.hdf5` is installed locally under `data/` and ignored by git.
- A 1,000-molecule manifest was generated with 1,000 successful parses and 0 errors.
- The test suite passes: 7 tests passing.
- A tiny DimeNet++ CPU overfit run on 16 molecules reduced training loss from 13.109 to 0.001576.
- SMILES inference works after training from an RDKit-generated conformer.

The current environment uses CPU PyTorch. GPU training should wait for the RTX 5080/CUDA environment.

## Repository Structure

- `data/` - local datasets and generated manifests. Contents are gitignored.
- `notebooks/` - Jupyter exploration notebooks.
- `src/gnn_excited/` - parser, dataset, model, training, and inference code.
- `scripts/` - command-line entrypoints for manifest building, training, and prediction.
- `configs/` - small-run training configs.
- `tests/` - parser, dataset, and inference smoke tests.

## Environment

Current CPU-first setup:

```powershell
conda env create -f environment.yml
conda activate gnn-excited
pip install -e .
```

When moving to the RTX 5080, replace the CPU PyTorch install with the matching CUDA-enabled PyTorch and PyTorch Geometric packages.

## Build a Small Manifest

Start with 100 molecules:

```powershell
python scripts/build_manifest.py --hdf5 data/A_9.hdf5 --out data/processed/a9_manifest_100.csv --max-count 100
```

The manifest contains molecule key, atom count, `S1_eV`, `S1_f`, `log1p_S1_f`, parse status, and any parse error.

For the current validation target, build 1,000 molecules:

```powershell
python scripts/build_manifest.py --hdf5 data/A_9.hdf5 --out data/processed/a9_manifest_1000.csv --max-count 1000
```

Inspect target distributions:

```powershell
python scripts/inspect_targets.py data/processed/a9_manifest_1000.csv
```

## Train a Tiny DimeNet++ Run

```powershell
python scripts/train_dimenet.py --config configs/small_cpu.yaml
```

This is meant as a correctness smoke test on CPU, not a production-quality training run.

To verify the model can memorize a tiny subset:

```powershell
python scripts/train_dimenet.py --config configs/overfit_16_cpu.yaml
```

The overfit run is the main pre-GPU wiring check: parser, PyG batching, DimeNet++ forward/backward pass, loss, optimizer, and checkpointing.

## Predict From SMILES

After training creates a checkpoint:

```powershell
python scripts/predict_smiles.py "CCO" --checkpoint checkpoints/dimenetpp_a9_small.pt
```

Output shape:

```json
{
  "smiles": "CCO",
  "s1_energy_ev": 0.0,
  "s1_oscillator_strength": 0.0,
  "geometry_source": "rdkit_etkdg_mmff"
}
```
