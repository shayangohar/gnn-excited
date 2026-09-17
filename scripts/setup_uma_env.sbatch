#!/bin/bash
#SBATCH --job-name=uma_env_setup
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-211 step 1: fairchem env + UMA-S weights on Delta.
# Weights: huggingface.co/facebook/UMA, checkpoints/uma-s-1p2p1.pt (small, latest).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs checkpoints/uma
source /etc/profile.d/modules.sh
module purge
module load miniforge3-python

if [ ! -d envs/uma-fairchem ]; then
  conda create --yes --prefix envs/uma-fairchem python=3.11
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./envs/uma-fairchem
pip install --upgrade pip
pip install fairchem-core huggingface_hub
python -c 'import fairchem, torch; print("fairchem ok, torch", torch.__version__)'

# UMA-S checkpoint + config (UMA-S ~150M params / 6M active MoLE).
python - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='facebook/UMA',
    allow_patterns=['checkpoints/uma-s-1p2p1.pt', 'config.json', 'LICENSE', 'USE_POLICY.md'],
    local_dir='checkpoints/uma',
)
print('snapshot:', path)
EOF
ls -lh checkpoints/uma checkpoints/uma/checkpoints
