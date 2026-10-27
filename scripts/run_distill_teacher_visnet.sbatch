#!/bin/bash
#SBATCH --job-name=distill_teacher_visnet
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=02:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-257 B2: the production PubChemQC-e20 ViSNet members (KNOW-256) predict the distillation pool.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs runs/distill
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /etc/profile.d/modules.sh
module purge
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
args=()
for s in 17 18 19; do
  args+=(--checkpoint checkpoints/pubchemqc_e20_enc_qm9gwbse_full_finetune_s$s.pt --out runs/distill/teacher_pqc_e20_s$s.csv)
done
python scripts/predict_packed_visnet.py --pool data/processed/distill/pool_pubchemqc.npz "${args[@]}"
echo TEACHER_VISNET_DONE
