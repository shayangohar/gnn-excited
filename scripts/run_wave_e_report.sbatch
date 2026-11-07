#!/bin/bash
#SBATCH --job-name=wave_e_report
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-291. PART=e1: val-weighted blends of the heavy-aux members vs production (KNOW-288) on
# QM9GWBSE, then the same weights on the external pool, per set and per element.
# PART=e2: the exchange-rate curve, OE62H error vs number of heavy-element TD-DFT labels.
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
H=runs/head_to_head
if [ "${PART:?set PART=e1 or e2}" = e1 ]; then
  python scripts/blend_predictions.py --spec configs/ensemble_blend_wave_e.yaml --out runs/ensemble_blend_wave_e.summary.json
  python scripts/blend_pool_predictions.py --spec configs/ensemble_blend_wave_e.yaml --report runs/ensemble_blend_wave_e.summary.json
  args=(--pred "production=$H/ext_blend_uma_richind+uma_richind_st+distill.csv" --reference production
        --report runs/wave_e_production_report.json)
  for b in he_heads+st_he_heads+distill he_heads+st_he_heads+distill+he_delta he_heads+distill; do
    [ -f "$H/ext_blend_$b.csv" ] && args+=(--pred "$b=$H/ext_blend_$b.csv")
  done
else
  args=(--pred richind=$H/ext_uma_richind_s17.csv --reference richind --report runs/exchange_rate_report.json
        --summary richind=runs/uma_qm9gwbse_richind_s17.summary.json)
  for arm in he_heads_n25k he_heads_n50k he_heads_n100k he_heads he_heads_n400k; do
    if [ -f $H/ext_uma_richind_${arm}_s17.csv ]; then
      args+=(--pred $arm=$H/ext_uma_richind_${arm}_s17.csv --summary $arm=runs/uma_qm9gwbse_richind_${arm}_s17.summary.json)
    else
      echo "arm $arm has no pool predictions; skipped"
    fi
  done
fi
python scripts/element_bias_report.py "${args[@]}"
echo WAVE_E_REPORT_DONE
