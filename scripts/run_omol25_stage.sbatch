#!/bin/bash
#SBATCH --job-name=omol25_stage
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
export HF_TOKEN="${HF_TOKEN:-$(cat /u/mgohar/.omol25_hf_token 2>/dev/null || true)}"
if [ -z "${HF_TOKEN}" ]; then
  echo "HF token missing; see scripts/stage_omol25_subset.py docstring" >&2
  exit 1
fi
python scripts/stage_omol25_subset.py ${STAGE_ARGS:---list-only}
