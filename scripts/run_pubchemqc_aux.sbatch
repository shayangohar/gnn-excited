#!/bin/bash
#SBATCH --job-name=pqc_aux
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-287: unit tests, then the heavy-element and matched-CHONF PubChemQC aux sets
# (scripts/build_pubchemqc_aux.py) that the multi-fidelity UMA arms train on.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python -m pytest tests/test_multi_fidelity.py tests/test_head_to_head.py tests/test_wave_c_helpers.py -q -p no:cacheprovider
python scripts/build_pubchemqc_aux.py
echo PQC_AUX_DONE
