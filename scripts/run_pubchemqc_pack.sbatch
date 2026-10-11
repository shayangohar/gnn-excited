#!/bin/bash
#SBATCH --job-name=pqc_pack
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-217: repack the 8 extraction chunks into one packed HDF5 (replaces the
# per-group merge in run_pubchemqc_merge.sbatch) and write QM9GWBSE val/test
# identity exclusions for leakage-safe pretraining.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source envs/gnn-excited-delta/bin/activate
LOCAL_DIR=/tmp/${USER}/pqc_pack_${SLURM_JOB_ID}
mkdir -p "$LOCAL_DIR"
trap 'rm -rf "$LOCAL_DIR"' EXIT
python scripts/pack_pubchemqc.py \
  --chunk-dir data/processed/pubchemqc \
  --out-h5 data/processed/pubchemqc/pubchemqc_s10_packed.h5 \
  --manifest data/processed/pubchemqc/pubchemqc_s10_packed_manifest.csv \
  --local-dir "$LOCAL_DIR" --workers 8 \
  --qm9gwbse-manifest data/processed/qm9gwbse/manifest.csv \
  --qm9gwbse-splits data/processed/qm9gwbse/splits.csv \
  --exclusions-out data/processed/pubchemqc/qm9gwbse_holdout_exclusions.csv
echo PACK_DONE
