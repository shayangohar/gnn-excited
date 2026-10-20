#!/bin/bash
#SBATCH --job-name=calibrate_uq
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-250 (DOC-248 L6): split-conformal intervals for the production blend (KNOW-247):
# UMA blr2e-4 seeds 17-19 at 0.55 and PubChemQC-e20 ViSNet seeds 17-19 at 0.45.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
P=runs/ensemble_preds
python scripts/calibrate_uncertainty.py \
  --family 0.55 $P/uma_blr2e-4_s17 runs/uma_qm9gwbse_finetune_blr2e-4_s18 runs/uma_qm9gwbse_finetune_blr2e-4_s19 \
  --family 0.45 $P/pqc_e20_s17 $P/pqc_e20_s18 $P/pqc_e20_s19 \
  --out runs/production_blend_conformal.summary.json
echo CALIBRATE_DONE
