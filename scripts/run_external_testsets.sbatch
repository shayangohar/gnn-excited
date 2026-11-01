#!/bin/bash
#SBATCH --job-name=external_testsets
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-271: external head-to-head pool + labels + QM9 overlap keys (scripts/build_external_testsets.py),
# after the head-to-head unit tests.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python -m pytest tests/test_head_to_head.py -q -p no:cacheprovider
python scripts/build_external_testsets.py
echo EXTERNAL_TESTSETS_DONE
