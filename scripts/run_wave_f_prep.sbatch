#!/bin/bash
#SBATCH --job-name=wave_f_prep
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

# TASK-310 wave F (CPU prep): unit tests; the he4 promotion record (val-weight blend, production
# metrics with triplets, head-to-head rebuilt from configs/production.yaml); the few-shot OE62H folds
# and configs; the 2.5k/5k/10k nested heavy aux sets for the knee of the exchange-rate curve.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python -m pytest tests/test_wave_f.py tests/test_multi_fidelity.py tests/test_head_to_head.py -q -p no:cacheprovider
python scripts/blend_predictions.py --spec configs/ensemble_blend_production.yaml --out runs/ensemble_blend_production.summary.json
python scripts/production_metrics.py --report runs/production_metrics.json
[ -f runs/head_to_head_report.json ] && cp -n runs/head_to_head_report.json runs/head_to_head_report.v2_three_family.json
python scripts/head_to_head_report.py
python scripts/build_oe62h_shots.py
for n in 2500:2p5k 5000:5k 10000:10k; do
  python scripts/build_pubchemqc_aux.py --n-heavy ${n%%:*} --skip-chonf --out-dir data/processed/pubchemqc_aux_n${n##*:}
done
echo WAVE_F_PREP_DONE
