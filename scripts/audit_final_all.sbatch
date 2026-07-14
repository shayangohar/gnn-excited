#!/bin/bash
#SBATCH --job-name=qcdge_audit
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=csn100
#SBATCH --no-requeue
#SBATCH --time=12:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /home/mgohar/gnn-excited
mkdir -p data/processed/qcdge_audit runs
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
source /cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh
conda activate /expanse/lustre/projects/csn100/mgohar/gnn-excited/envs/gnn-excited-gpu

RAW_DATA=/expanse/lustre/projects/csn100/mgohar/gnn-excited/data
python scripts/audit_qcdge.py \
  --csv "$RAW_DATA/final_all.csv" \
  --hdf5 "$RAW_DATA/final_all.hdf5" \
  --manifest data/processed/final_all_manifest_s5.csv \
  --dedup-manifest data/processed/final_all_manifest_s5_dedup.csv \
  --out-dir data/processed/qcdge_audit \
  --checksum-file "$RAW_DATA/SHA512SUM" \
  --compressed-hdf5 "$RAW_DATA/final_all.hdf5.gz" \
  --seed 17 \
  --train-fraction 0.8 \
  --val-fraction 0.1
