#!/bin/bash
#SBATCH --job-name=wave_e_aux
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-291 E2 (bet B): nested heavy-element PubChemQC aux subsets of 25k/50k/100k/400k molecules
# (same seed, so each is a prefix of the next) after the unit tests.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python -m pytest tests/test_multi_fidelity.py tests/test_head_to_head.py -q -p no:cacheprovider
for n in 25 50 100 400; do
  python scripts/build_pubchemqc_aux.py --n-heavy ${n}000 --skip-chonf --out-dir data/processed/pubchemqc_aux_n${n}k
done
echo WAVE_E_AUX_DONE
