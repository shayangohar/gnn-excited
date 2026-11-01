#!/bin/bash
#SBATCH --job-name=joint_distill_data
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-262 C2: merge teacher pseudo-labels with QM9GWBSE true labels (scripts/build_joint_distill_dataset.py).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python scripts/build_joint_distill_dataset.py
echo JOINT_DATA_DONE
