#!/bin/bash
#SBATCH --job-name=wave_g_prep
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-320 wave G (CPU prep): tests; the he5 promotion record (production metrics and head-to-head
# rebuilt from configs/production.yaml, he4 copies kept as *.v3_he4.json); the few-shot cross-test shot
# sets and configs (OE62L shots; S-free OE62H shots); the mixed-geometry training file; the conformer pool.
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python -m pytest tests/test_wave_g.py tests/test_wave_f.py tests/test_head_to_head.py -q -p no:cacheprovider
for f in production_metrics head_to_head_report; do
  if [ -f runs/$f.json ] && [ ! -f runs/$f.v3_he4.json ]; then cp runs/$f.json runs/$f.v3_he4.json; fi
done
python scripts/production_metrics.py --report runs/production_metrics.json
python scripts/head_to_head_report.py
G=runs/fewshot_wave_g
python scripts/build_oe62h_shots.py --set OE62L --ks 25 200 --tag oe62l_ --no-control --wandb-group uma-wave-g-fewshot \
  --out-dir data/processed/oe62l_fewshot --config-dir $G/configs_oe62l --run-dir $G
python scripts/build_oe62h_shots.py --set OE62H --ks 100 --exclude-element S --draws 3 --tag nos_ --no-control \
  --wandb-group uma-wave-g-fewshot --out-dir data/processed/oe62h_nos_fewshot --config-dir $G/configs_nos --run-dir $G
cat $G/configs_oe62l/configs.txt $G/configs_nos/configs.txt > $G/configs.txt
echo "few-shot configs: $(wc -l < $G/configs.txt)"
python scripts/build_mixed_packed.py
export TMPDIR=/tmp/$USER/confpool_$SLURM_JOB_ID && mkdir -p $TMPDIR
python scripts/build_conformer_pool.py --xtb $HOME/software/xtb/bin/xtb --workers 64 --n-conf 10
rm -rf $TMPDIR
echo WAVE_G_PREP_DONE
