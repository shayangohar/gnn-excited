#!/bin/bash
#SBATCH --job-name=production_external
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
#SBATCH --time=01:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# Production promoted 2026-11-03: the UMA richind S+T members predict the external pool, then the
# head-to-head report is rebuilt for the three-family production blend.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs/head_to_head
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
ORIG_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python scripts/train_uma_qm9gwbse.py --config configs/uma_st_external_eval_inference.yaml
conda deactivate
export LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH
module purge
source envs/gnn-excited-delta/bin/activate
python scripts/head_to_head_report.py
echo PRODUCTION_EXTERNAL_DONE
