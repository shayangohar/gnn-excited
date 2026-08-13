#!/bin/bash
#SBATCH --job-name=qm9gwbse_build
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
#SBATCH --time=08:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p data/processed/qm9gwbse runs
source /etc/profile.d/modules.sh
module purge
module load cudatoolkit/25.3_12.8
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
python scripts/build_qm9gwbse.py \
  --raw-dir data/raw/qm9gwbse \
  --out-dir data/processed/qm9gwbse \
  --identity-csv data/raw/qm9gwbse/qm9_identities.csv \
  --qcdge-identity-csv data/raw/qm9gwbse/qcdge_final_all.csv
