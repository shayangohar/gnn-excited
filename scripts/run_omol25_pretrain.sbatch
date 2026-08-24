#!/bin/bash
#SBATCH --job-name=omol_pre
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuH200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=bhzu-delta-gpu

set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
CONFIG_PATH=${CONFIG_PATH:-configs/omol25_gap_pretrain.yaml}
python scripts/train_dimenet.py --config "$CONFIG_PATH"
echo PRETRAIN_DONE
