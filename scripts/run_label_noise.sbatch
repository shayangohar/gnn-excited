#!/bin/bash
#SBATCH --job-name=label_noise
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-276 (bet C): qsGW/DFT orbital-energy features vs production errors
# (scripts/label_noise_diagnostic.py) -> runs/label_noise_diagnostic.json.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
export OMP_NUM_THREADS=8
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python scripts/label_noise_diagnostic.py
echo LABEL_NOISE_DONE
