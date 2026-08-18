#!/bin/bash
#SBATCH --job-name=qm9_stda_desc
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
export XTB4STDAHOME=/projects/bhzu/software/xtb4stda-v1.0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /etc/profile.d/modules.sh
module purge
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
python scripts/build_electronic_descriptors.py \
  --hdf5 data/processed/qm9gwbse/qm9gwbse.h5 \
  --manifest data/processed/qm9gwbse/manifest.csv \
  --output data/processed/qm9gwbse/electronic_descriptors_stda_20k.h5 \
  --xtb4stda "$XTB4STDAHOME/xtb4stda" \
  --stda "$XTB4STDAHOME/stda_v1.6.1" \
  --parameter-dir "$XTB4STDAHOME" \
  --limit 20000 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --seed 17 \
  --scratch-root "$TMPDIR"
