#!/bin/bash
#SBATCH --job-name=pqc_screen
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-244 / KNOW-242: flag disconnected, multi-fragment, short-contact and S1-outlier
# PubChemQC geometries, write the combined exclusion CSV the e20 pretrain reads, and
# probe the 10-epoch checkpoint on CPU to check that the flags catch the blow-ups.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
export OMP_NUM_THREADS=16
OUT=data/processed/pubchemqc
python scripts/screen_pubchemqc.py \
  --packed-h5 $OUT/pubchemqc_s10_packed.h5 \
  --base-exclusions $OUT/qm9gwbse_holdout_exclusions.csv \
  --exclusions-out $OUT/pretrain_exclusions_screened.csv \
  --flagged-out $OUT/screen_flagged.csv \
  --report-out runs/pubchemqc_screen.report.json \
  --workers 16 \
  --checkpoint checkpoints/pubchemqc_s5_pretrain.pt
echo SCREEN_DONE
