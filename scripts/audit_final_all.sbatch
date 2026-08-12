#!/bin/bash
#SBATCH --job-name=qcdge_audit
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=12:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p data/processed/qcdge_audit runs
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
source /etc/profile.d/modules.sh
module purge
module load cudatoolkit/25.3_12.8
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate

RAW_DATA=/work/hdd/bhzu/gnn-excited/data
python -m pytest -q tests/test_data_audit.py
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
