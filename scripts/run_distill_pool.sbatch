#!/bin/bash
#SBATCH --job-name=distill_pool
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

# TASK-257 B2: CHONF <=12-heavy PubChemQC pool (max 1M) for teacher pseudo-labels (scripts/build_distill_pool.py).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs data/processed/distill
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python scripts/build_distill_pool.py --out data/processed/distill/pool_pubchemqc.npz
echo POOL_DONE
