#!/bin/bash
#SBATCH --job-name=heavy_element_report
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-287: per-set / per-element S1 error of the multi-fidelity UMA arms against UMA richind s17
# (same recipe, GW only). Arms that did not finish are skipped.
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
args=(--pred richind=runs/head_to_head/ext_uma_richind_s17.csv --reference richind
      --summary richind=runs/uma_qm9gwbse_richind_s17.summary.json)
for arm in he_delta he_heads chonf_delta; do
  if [ -f runs/head_to_head/ext_uma_richind_${arm}_s17.csv ]; then
    args+=(--pred $arm=runs/head_to_head/ext_uma_richind_${arm}_s17.csv
           --summary $arm=runs/uma_qm9gwbse_richind_${arm}_s17.summary.json)
  else
    echo "arm $arm has no pool predictions; skipped"
  fi
done
python scripts/element_bias_report.py "${args[@]}"
echo HEAVY_ELEMENT_REPORT_DONE
