#!/bin/bash
#SBATCH --job-name=qm9_export
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-211: QM9GWBSE -> packed .npz for the UMA env (no h5py there).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source envs/gnn-excited-delta/bin/activate
python scripts/export_qm9gwbse_packed.py
echo EXPORT_DONE
