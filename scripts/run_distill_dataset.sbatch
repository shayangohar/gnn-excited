#!/bin/bash
#SBATCH --job-name=distill_dataset
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-257 B2: production-blend teacher labels (0.55 UMA rich x3 + 0.45 PQC-e20 x3, KNOW-256) -> packed pseudo-label set.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
D=runs/distill
python scripts/build_distill_dataset.py --pool data/processed/distill/pool_pubchemqc.npz \
  --family 0.55 $D/teacher_uma_rich_s17.csv $D/teacher_uma_rich_s18.csv $D/teacher_uma_rich_s19.csv \
  --family 0.45 $D/teacher_pqc_e20_s17.csv $D/teacher_pqc_e20_s18.csv $D/teacher_pqc_e20_s19.csv \
  --out-h5 data/processed/distill/teacher_pseudolabels.h5 --out-manifest data/processed/distill/teacher_pseudolabels_manifest.csv
echo DATASET_DONE
