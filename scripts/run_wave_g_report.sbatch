#!/bin/bash
#SBATCH --job-name=wave_g_report
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

# TASK-320 wave G reports, one per PART (each submitted afterany its training jobs):
#   fewshot   L->H and S-free few-shot cross-tests -> runs/fewshot_cross_report.json
#   geometry  xTB 3-seed ensemble vs B3LYP 3-seed ensemble, and singles incl. the mixed-geometry model,
#             on the four geometry pools -> runs/geometry_training_report_g_{ens,single}.json
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
C=runs/cheap_geometry
case "${PART:?set PART}" in
  fewshot)
    python scripts/fewshot_cross_report.py ;;
  geometry)
    for src in b3lyp etkdg xtb uma; do
      python scripts/average_pool_predictions.py --inputs $C/$src.uma_richind_s{17,18,19}.csv --out $C/$src.uma_richind_ens3.csv
      python scripts/average_pool_predictions.py --inputs $C/$src.uma_richind_xtb_s{17,18,19}.csv --out $C/$src.uma_richind_xtb_ens3.csv
    done
    python scripts/geometry_training_report.py --reference b3lyp_ens3 --report runs/geometry_training_report_g_ens.json \
      --model "b3lyp_ens3=$C/{source}.uma_richind_ens3.csv" --model "xtb_ens3=$C/{source}.uma_richind_xtb_ens3.csv"
    python scripts/geometry_training_report.py --reference b3lyp_trained --report runs/geometry_training_report_g_single.json \
      --model "b3lyp_trained=$C/{source}.uma_richind_s17.csv" --model "xtb_trained=$C/{source}.uma_richind_xtb_s17.csv" \
      --model "mixed=$C/{source}.uma_richind_mixed_s17.csv" \
      --summary b3lyp_trained=runs/uma_qm9gwbse_richind_s17.summary.json \
      --summary xtb_trained=runs/uma_qm9gwbse_richind_xtb_s17.summary.json \
      --summary mixed=runs/uma_qm9gwbse_richind_mixed_s17.summary.json ;;
  *) echo "unknown PART $PART"; exit 1 ;;
esac
echo WAVE_G_REPORT_${PART}_DONE
