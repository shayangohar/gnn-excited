#!/bin/bash
#SBATCH --job-name=audit_followups
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-328 (DOC-327 audit follow-ups, CPU only):
#   1. lock dev/test halves of the external sets -> configs/splits/external_dev_test.csv (+ sha256)
#   2. frozen pooled he_heads_s17 features of the external pool (CPU) -> data/processed/uma_features/pool_he_heads_s17
#   3. few-shot recalibration baselines on the wave F folds -> runs/fewshot_baselines_report.json
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python -m pytest tests/test_audit_followups.py -q -p no:cacheprovider
python scripts/lock_external_split.py
deactivate
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
F=data/processed/uma_features/pool_he_heads_s17
if [ ! -f $F.meta.npz ]; then
  /projects/bhzu/gnn-excited/envs/uma-fairchem/bin/python scripts/extract_uma_features.py --device cpu --num-workers 4 --batch-size 32 \
    --packed data/processed/external_testsets/pool.npz --checkpoint checkpoints/uma_qm9gwbse_richind_he_heads_s17.pt \
    --config configs/uma_qm9gwbse_richind_he_heads_s17.yaml --out $F
fi
conda deactivate
module purge
source envs/gnn-excited-delta/bin/activate
python scripts/fewshot_baselines.py
echo AUDIT_FOLLOWUPS_DONE
