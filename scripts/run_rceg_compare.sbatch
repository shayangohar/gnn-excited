#!/bin/bash
#SBATCH --job-name=rceg_compare
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# rCEG comparison (KNOW-293). PART=prep: unit tests, then scripts/build_rceg_comparison.py.
# PART=report: scripts/rceg_comparison_report.py over the UMA gap runs that finished.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs/rceg
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
if [ "${PART:?set PART=prep or report}" = prep ]; then
  python -m pytest tests/test_cheap_geometry.py tests/test_multi_fidelity.py -q -p no:cacheprovider
  python scripts/build_rceg_comparison.py
else
  args=()
  for arm in ref qm9; do
    s=runs/uma_rceg_gap_${arm}_s17.summary.json
    if grep -q '"status": "completed"' $s 2>/dev/null; then args+=(--run $arm=$s); else echo "arm $arm did not finish; skipped"; fi
  done
  python scripts/rceg_comparison_report.py "${args[@]}"
fi
echo RCEG_${PART^^}_DONE
