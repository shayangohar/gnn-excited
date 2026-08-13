# gnn-excited

`gnn-excited` is a research codebase for predicting excited-state properties of small organic molecules with 3D graph neural networks. The current clean QCDGE baseline predicts paired `S1`-`S5` energy/oscillator targets; the QM9GWBSE qsGW-BSE data provides a transfer-learning target using a one-pass ViSNet encoder.

The long-term goal is to explore whether learned molecular models can provide fast approximations to quantum-chemistry excited-state calculations for early-stage screening and analysis.

## Scope

The primary supervised baseline is the cleaned, deduplicated QCDGE `S1`-`S5` manifest. QM9GWBSE stores five qsGW-BSE singlet/triplet excitation arrays, oscillator strengths, and transition dipoles in a compact HDF5 for leakage-safe transfer experiments. Targets include `S1_eV`-`S5_eV` and paired `log1p_S1_f`-`log1p_S5_f`.

Training uses ground-state molecular geometries:

- atomic numbers from `ground_state/labels`
- 3D coordinates from `ground_state/coords`

Excited-state fields are used only as supervised learning targets, not as model inputs. Raw and processed datasets remain local and gitignored.

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

The default manifest contains the current baseline targets, `S1_eV` and `S1_f`. Fixed-count multi-state manifests can also be generated for later experiments:

```powershell
python scripts/build_manifest.py --hdf5 data/A_9.hdf5 --out data/processed/a9_manifest_s5.csv --max-count 10000 --singlets 5 --triplets 0
```

Inspect target distributions:

```powershell
python scripts/inspect_targets.py data/processed/a9_manifest_1000.csv
```

Build the compact QM9GWBSE HDF5, manifest, splits, checksums, and explicit QCDGE identity audit (the identity CSVs must be supplied; keys are never inferred as identities):

```powershell
python scripts/build_qm9gwbse.py `
  --raw-dir data/raw/qm9gwbse `
  --out-dir data/processed/qm9gwbse `
  --identity-csv data/raw/qm9gwbse/qm9_identities.csv `
  --qcdge-identity-csv data/raw/qm9gwbse/qcdge_final_all.csv
```

### Full-dataset integrity audit

Before training on `final_all`, reconcile the published molecule index with the HDF5 keys and target
manifest:

```powershell
python scripts/audit_qcdge.py `
  --csv data/final_all.csv `
  --hdf5 data/final_all.hdf5 `
  --manifest data/processed/final_all_manifest_s5.csv `
  --dedup-manifest data/processed/final_all_manifest_s5_dedup.csv `
  --out-dir data/processed/qcdge_audit `
  --checksum-file data/SHA512SUM `
  --compressed-hdf5 data/final_all.hdf5.gz
```

The audit canonicalizes RDKit SMILES and validates InChI identities, matches the published CSV index to
HDF5 and manifest records, verifies SHA-512 hashes, and writes deterministic random, Murcko-scaffold, and
generic-core split assignments. Scaffold/core groups are kept wholly within one split; acyclic molecules
use a generic full-molecule topology key instead of being collapsed into one empty-scaffold group.

Validate that the persisted assignments cover every usable row in the deduplicated manifest before
starting a training run:

```powershell
python scripts/validate_split.py `
  --manifest data/processed/final_all_manifest_s5_dedup.csv `
  --splits data/processed/qcdge_audit/splits.csv `
  --columns random_split scaffold_split core_split
```

Training configs can select one of these persisted assignments with `dataset.split_path` and
`dataset.split_column`. This avoids regenerating a row-level random split and keeps duplicate molecular
identities confined to a single partition.

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

Run the deterministic QM9GWBSE smoke check, then the clean QCDGE ViSNet pretrain and readout-only transfer configs:

```powershell
python scripts/train_dimenet.py --config configs/qm9gwbse_smoke_cpu.yaml
python scripts/train_dimenet.py --config configs/final_all_s5_visnet_pretrain.yaml
python scripts/train_dimenet.py --config configs/qm9gwbse_visnet_readout_transfer.yaml
```

Compare completed training runs:

```powershell
python scripts/compare_runs.py runs/dimenetpp_a9_10k_gpu.summary.json runs/dimenetpp_a9_10k_gpu_long.summary.json
```

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

- clean QCDGE `S1`-`S5` manifests and deterministic leakage-safe splits
- streaming QM9GWBSE ZIP-to-HDF5 data engineering with checksum and identity audits
- PyTorch Geometric datasets and shared direct/gap/order losses
- one-pass ViSNet paired energy/oscillator readouts with readout-only or full transfer loading
- DimeNet++ training and RDKit-based SMILES-to-geometry inference
- parser, dataset, model, loss, and training smoke tests

The project remains in early validation. Full QCDGE pretraining and QM9GWBSE transfer runs are not yet benchmarked; current results are software and workflow validation.

## Research Directions

Planned extensions include:

- benchmarking clean QCDGE `S1`-`S5` pretraining and QM9GWBSE qsGW-BSE transfer
- quantifying quantum-chemistry versus RDKit-generated geometry effects
- extending explicit identity alignment where source metadata permits it
- comparing direct prediction against delta machine learning approaches
- evaluating uncertainty, calibration, and chemical-domain generalization

## License

License information has not yet been specified.
