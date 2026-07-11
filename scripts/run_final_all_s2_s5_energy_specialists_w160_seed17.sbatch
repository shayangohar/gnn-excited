#!/bin/bash
#SBATCH --job-name=gnn_s2_s5_specs
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=377300M
#SBATCH --account=csn100
#SBATCH --no-requeue
#SBATCH --time=48:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd /home/mgohar/gnn-excited
mkdir -p runs checkpoints

export SPECIALIST_SCRATCH_ROOT=${SLURM_TMPDIR:-/tmp/${USER}/gnn-excited-${SLURM_JOB_ID}}
target_dir="${SPECIALIST_SCRATCH_ROOT}/gnn_excited_data"
mkdir -p "${target_dir}"
trap 'rm -rf "${SPECIALIST_SCRATCH_ROOT}"' EXIT

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh
conda activate /expanse/lustre/projects/csn100/mgohar/gnn-excited/envs/gnn-excited-gpu

cp -p data/final_all.hdf5 "${target_dir}/final_all.hdf5.tmp"
mv "${target_dir}/final_all.hdf5.tmp" "${target_dir}/final_all.hdf5"

pids=()
for state in 2 3 4 5; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem=90G \
    --output="runs/s${state}_energy_specialist.%j.%N.out" \
    --error="runs/s${state}_energy_specialist.%j.%N.err" \
    python scripts/train_dimenet.py \
    --config "configs/final_all_s${state}_energy_specialist_w160_seed17.yaml" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
