#!/bin/bash
#SBATCH --job-name=visnet_eval_batch
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=01:30:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-244: run several eval-only ViSNet configs (epochs 0 + evaluation.checkpoint_path)
# in one GPU allocation; each writes val and test prediction CSVs for the ensemble blend.
# CONFIG_PATHS is a space-separated list. A failing config does not stop the others.
set -uo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs runs/ensemble_preds
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER/gnn-excited-$SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /etc/profile.d/modules.sh
module purge
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")'
failed=0
for config in ${CONFIG_PATHS:?set CONFIG_PATHS}; do
  echo "== $config"
  python scripts/train_dimenet.py --config "$config" || { echo "FAILED $config"; failed=$((failed + 1)); }
done
echo "EVAL_BATCH_DONE failed=$failed"
exit $((failed > 0))
