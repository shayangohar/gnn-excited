#!/bin/bash
#SBATCH --job-name=head_to_head
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=01:30:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-271: production members (UMA richind x3, distilled ViSNet x3) and the Baum et al.
# ViSNetGWBSE checkpoints predict the external pool; theirs also predict all of QM9GWBSE; then
# scripts/head_to_head_report.py. Run after scripts/run_external_testsets.sbatch.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs/head_to_head
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
ORIG_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
POOL=data/processed/external_testsets/pool.npz
source /etc/profile.d/modules.sh

echo "== UMA richind x3 (fairchem env; KNOW-219: no module purge first)"
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python scripts/train_uma_qm9gwbse.py --config configs/uma_external_eval_inference.yaml
conda deactivate
export LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH

echo "== distilled ViSNet x3 and ViSNetGWBSE (project venv)"
module purge
source envs/gnn-excited-delta/bin/activate
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
args=()
for s in 17 18 19; do
  args+=(--checkpoint checkpoints/distill_enc_qm9gwbse_finetune_s$s.pt --out runs/head_to_head/ext_distill_s$s.csv)
done
python scripts/predict_packed_visnet.py --pool $POOL "${args[@]}"
python scripts/eval_visnetgwbse.py --pool $POOL --pool data/processed/qm9gwbse/qm9gwbse_packed.npz

echo "== report"
python scripts/head_to_head_report.py
echo HEAD_TO_HEAD_DONE
