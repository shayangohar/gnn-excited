#!/bin/bash
#SBATCH --job-name=gnn_final_all_s5_bench
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=csn100
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /home/mgohar/gnn-excited
mkdir -p runs checkpoints
SCRATCH_CANDIDATE=/scratch/$USER/gnn-excited-$SLURM_JOB_ID
if mkdir -p $SCRATCH_CANDIDATE 2>/dev/null; then
  export TMPDIR=$SCRATCH_CANDIDATE
else
  export TMPDIR=/tmp/$USER/gnn-excited-$SLURM_JOB_ID
  mkdir -p $TMPDIR
fi
trap 'rm -rf $TMPDIR' EXIT
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh
conda activate /expanse/lustre/projects/csn100/mgohar/gnn-excited/envs/gnn-excited-gpu
python scripts/train_dimenet.py --config configs/final_all_s5_wide_clip_benchmark.yaml
