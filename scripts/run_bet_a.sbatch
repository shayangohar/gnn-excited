#!/bin/bash
#SBATCH --job-name=bet_a
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=06:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-310 wave F, bet A R0/R1 (TASK-277). FEATURES=ft: per-atom invariants of the fine-tuned
# uma_qm9gwbse_richind_s17 backbone; FEATURES=pt: of the pretrained UMA-S backbone. Extracts the
# features once, then trains the mlp control, r0 (5x5 tridiagonal) and r1 (16 configurations) decoders
# at seeds 17-19 on them (scripts/train_frozen_decoder.py).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs/bet_a data/processed/uma_features
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
F=${FEATURE_DIR:-data/processed/uma_features}/${FEATURES:?set FEATURES=ft or pt}
mkdir -p $(dirname $F) ${RUN_DIR:-runs/bet_a}
EPOCHS=${EPOCHS:-150}
if [ ! -f $F.meta.npz ]; then
  if [ "$FEATURES" = ft ]; then
    python scripts/extract_uma_features.py --checkpoint checkpoints/uma_qm9gwbse_richind_s17.pt \
      --config configs/uma_qm9gwbse_richind_s17.yaml --out $F ${LIMIT:+--limit $LIMIT}
  else
    python scripts/extract_uma_features.py --pretrained --checkpoint checkpoints/uma/checkpoints/uma-s-1p2p1.pt \
      --out $F ${LIMIT:+--limit $LIMIT}
  fi
fi
for kind in ${KINDS:-mlp r0 r1}; do
  for seed in ${SEEDS:-17 18 19}; do
    python scripts/train_frozen_decoder.py --features $F --kind $kind --seed $seed --epochs $EPOCHS \
      --out ${RUN_DIR:-runs/bet_a}/${FEATURES}_${kind}_s${seed}
  done
done
echo BET_A_DONE
