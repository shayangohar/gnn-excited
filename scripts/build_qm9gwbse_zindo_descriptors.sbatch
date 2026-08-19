#!/bin/bash
#SBATCH --job-name=qm9_zindo_desc
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=12:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p data/processed/qm9gwbse runs
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER/gnn-excited-$SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /etc/profile.d/modules.sh
module purge
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
ORCA=/projects/bhzu/software/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca
LIMIT=${LIMIT:-100}
OUTPUT=${OUTPUT:-data/processed/qm9gwbse/electronic_descriptors_zindo_${LIMIT}.h5}
python scripts/zindo_descriptors.py build \
  --hdf5 data/processed/qm9gwbse/qm9gwbse.h5 \
  --manifest data/processed/qm9gwbse/manifest.csv \
  --output "$OUTPUT" \
  --orca "$ORCA" \
  --limit "$LIMIT" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --seed 17 \
  --num-roots 10 \
  --minimum-success-rate 0.95 \
  --scratch-root "$TMPDIR"
