#!/bin/bash
#SBATCH --job-name=uma_weights
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-211 step 2: download UMA-S weights (gated repo; HF token lives in
# ~/.config/hf/token on Delta, mode 600 — NEVER commit it).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs checkpoints/uma
source /etc/profile.d/modules.sh
module purge
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./envs/uma-fairchem
export HF_TOKEN="$(cat ~/.config/hf/token)"
python - <<'EOF'
import os
from huggingface_hub import snapshot_download, whoami
print('hf user:', whoami()['name'])
path = snapshot_download(
    repo_id='facebook/UMA',
    allow_patterns=['checkpoints/uma-s-1p2p1.pt', 'config.json', 'LICENSE', 'USE_POLICY.md'],
    local_dir='checkpoints/uma',
)
print('snapshot:', path)
EOF
ls -lh checkpoints/uma checkpoints/uma/checkpoints
python - <<'EOF'
import torch
try:
    sd = torch.load('checkpoints/uma/checkpoints/uma-s-1p2p1.pt', map_location='cpu', weights_only=True)
    print('state_dict tensors:', len(sd) if isinstance(sd, dict) else type(sd))
    print('total params:', sum(v.numel() for v in sd.values()) if isinstance(sd, dict) else 'n/a')
except Exception as exc:
    print('inspection skipped:', type(exc).__name__, str(exc)[:200])
EOF
