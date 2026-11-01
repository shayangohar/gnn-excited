#!/bin/bash
#SBATCH --job-name=orca_prep
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-262: pick and write QM9GWBSE train molecules for the ORCA STEOM-DLPNO-CCSD pilot.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs runs/orca_pilot
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python scripts/make_orca_pilot.py
echo ORCA_PREP_DONE
