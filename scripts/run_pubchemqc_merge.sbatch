#!/bin/bash
#SBATCH --job-name=pqc_merge
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-217: merge the 8 extraction chunks into the final training set.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source envs/gnn-excited-delta/bin/activate
python scripts/merge_pubchemqc.py \
  --chunk-dir data/processed/pubchemqc \
  --out-h5 data/processed/pubchemqc/train_st.h5 \
  --manifest data/processed/pubchemqc/train_st_manifest.csv
echo MERGE_DONE
