#!/bin/bash
#SBATCH --job-name=wave_f_report
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

# TASK-310 wave F reports, one per PART (each submitted afterany its training jobs):
#   delta    val-weight blends with he_delta x3     -> runs/ensemble_blend_production_f.summary.json
#   knee     OE62H error vs heavy aux labels 0-400k -> runs/exchange_rate_report_v2.json
#   xtb      xTB-trained vs B3LYP-trained on the E3 geometry pools -> runs/geometry_training_report.json
#   fewshot  few-shot OE62H GW labels               -> runs/fewshot_oe62h_report.json
#   bet_a    R0/R1 vs the mlp control              -> runs/bet_a_report.json
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
H=runs/head_to_head
case "${PART:?set PART}" in
  delta)
    python scripts/blend_predictions.py --spec configs/ensemble_blend_production.yaml --out runs/ensemble_blend_production_f.summary.json ;;
  knee)
    args=(--pred richind=$H/ext_uma_richind_s17.csv --reference richind --report runs/exchange_rate_report_v2.json
          --summary richind=runs/uma_qm9gwbse_richind_s17.summary.json)
    for arm in he_heads_n2p5k he_heads_n5k he_heads_n10k he_heads_n25k he_heads_n50k he_heads_n100k he_heads he_heads_n400k; do
      if [ -f $H/ext_uma_richind_${arm}_s17.csv ]; then
        args+=(--pred $arm=$H/ext_uma_richind_${arm}_s17.csv --summary $arm=runs/uma_qm9gwbse_richind_${arm}_s17.summary.json)
      else
        echo "arm $arm has no pool predictions; skipped"
      fi
    done
    python scripts/element_bias_report.py "${args[@]}" ;;
  xtb)
    python scripts/geometry_training_report.py --reference b3lyp_trained \
      --model "b3lyp_trained=runs/cheap_geometry/{source}.uma_richind_s17.csv" \
      --model "xtb_trained=runs/cheap_geometry/{source}.uma_richind_xtb_s17.csv" \
      --summary b3lyp_trained=runs/uma_qm9gwbse_richind_s17.summary.json \
      --summary xtb_trained=runs/uma_qm9gwbse_richind_xtb_s17.summary.json ;;
  fewshot)
    python scripts/fewshot_report.py ;;
  bet_a)
    python scripts/bet_a_report.py ;;
  *) echo "unknown PART $PART"; exit 1 ;;
esac
echo WAVE_F_REPORT_${PART}_DONE
