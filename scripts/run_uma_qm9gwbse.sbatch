#!/bin/bash
#SBATCH --job-name=uma_qm9
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=24:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-211: UMA-S finetune on QM9GWBSE in the fairchem env (KNOW-219 for the
# LD_LIBRARY_PATH fix; no module purge, it can leave conda undefined).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs checkpoints
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python -c 'import torch, fairchem.core; assert torch.cuda.is_available(); print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")'
CONFIG_PATH=${CONFIG_PATH:-configs/uma_qm9gwbse_finetune.yaml}
python scripts/train_uma_qm9gwbse.py --config "$CONFIG_PATH"
echo UMA_DONE
