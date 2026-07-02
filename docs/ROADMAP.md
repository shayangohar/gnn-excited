# QCDGE Excited-State Modeling Roadmap

## Current Baseline

The active baseline predicts the lowest singlet excitation energy and oscillator strength:

- `S1_eV`
- `S1_f`, trained as `log1p(S1_f)`

Training inputs are QCDGE ground-state geometries only:

- atomic numbers from `ground_state/labels`
- Cartesian coordinates from `ground_state/coords`

Excited-state fields are supervised targets, not model inputs.

## Stage 1: Finish The 10k Baseline

Use the long 10k run to decide whether the current small DimeNet++ configuration is stable enough to scale.

Compare against the previous 25-epoch 10k run:

```bash
python scripts/compare_runs.py \
  runs/dimenetpp_a9_10k_gpu.summary.json \
  runs/dimenetpp_a9_10k_gpu_long.summary.json
```

Primary comparison metrics:

- best validation `S1_eV` MAE
- test `S1_eV` MAE
- test `S1_f` MAE
- early-stopping status
- learning-rate schedule behavior

Decision rule:

- If the long 10k run improves or stabilizes, scale to a 50k A_9 run.
- If the long 10k run plateaus near the earlier result, tune model capacity before scaling.

## Stage 2: Multi-State Targets

The manifest builder supports fixed-count excited-state manifests:

```bash
python scripts/build_manifest.py \
  --hdf5 data/A_9.hdf5 \
  --out data/processed/a9_manifest_s5.csv \
  --max-count 10000 \
  --singlets 5 \
  --triplets 0
```

Initial multi-state target:

- `S1-S5` excitation energies
- matching `S1-S5` oscillator strengths

Later optional target:

- `T1-T5` excitation energies

Molecules with malformed excited-state JSON or insufficient requested states are recorded as manifest errors.

## Stage 3: SMILES-To-Geometry Covariate Shift

The model is trained on QCDGE optimized geometries, while SMILES inference currently uses RDKit conformers. This mismatch must be quantified before interpreting SMILES predictions quantitatively.

Planned geometry sources:

- QCDGE reference geometry: in-distribution upper-bound input
- RDKit ETKDG + MMFF/UFF: cheap SMILES baseline
- xTB-optimized geometry: more expensive semiempirical geometry

Evaluation method:

1. Select molecules with known QCDGE geometry and aligned SMILES.
2. Predict with QCDGE, RDKit, and xTB geometries using the same checkpoint.
3. Report degradation relative to QCDGE geometry.

Exact SMILES alignment requires `final_all.csv`.

## Stage 4: Delta-ML

Delta-ML remains a fallback if direct DimeNet++ prediction plateaus.

Candidate target:

```text
TDDFT_S1_eV - cheap_method_S1_eV
```

Use delta-ML only if a lower-cost method provides useful correlated excited-state estimates and the added compute cost is justified by accuracy gains.
