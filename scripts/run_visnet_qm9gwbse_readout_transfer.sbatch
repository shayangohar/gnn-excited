#!/bin/bash
#SBATCH --job-name=visnet_qm9_transfer
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=48:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs checkpoints
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER/gnn-excited-$SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /etc/profile.d/modules.sh
module purge
module load cudatoolkit/25.3_12.8
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
python scripts/train_dimenet.py --config configs/qm9gwbse_visnet_readout_transfer.yaml
