#!/bin/bash
#SBATCH --job-name=qm9_zindo_screen
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER/gnn-excited-$SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /etc/profile.d/modules.sh
module purge
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
python scripts/train_dimenet.py --config configs/qm9gwbse_visnet_full_finetune_eval_predictions.yaml
python scripts/zindo_descriptors.py screen \
  --descriptors data/processed/qm9gwbse/electronic_descriptors_zindo_2000.h5 \
  --manifest data/processed/qm9gwbse/manifest.csv \
  --production-predictions runs/visnet_qm9gwbse_full_finetune.predictions.csv \
  --output runs/zindo_2000_screen.json \
  --seed 17 \
  --folds 5
