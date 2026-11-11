#!/bin/bash
#SBATCH --job-name=fewshot_oe62h
#SBATCH --output=runs/fewshot_oe62h/%x.%A_%a.%N.out
#SBATCH --error=runs/fewshot_oe62h/%x.%A_%a.%N.err
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=02:30:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=FAIL

# TASK-310 wave F (bet B follow-up, KNOW-307): one few-shot OE62H fine-tune per array task. Task i
# runs line i+1 of runs/fewshot_oe62h/configs/configs.txt (scripts/build_oe62h_shots.py; line 1 is the
# k=0 control). The 1.1 GB checkpoint is deleted after a successful run: only the predictions are kept.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs/fewshot_oe62h checkpoints/fewshot_oe62h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${LIST:-runs/fewshot_oe62h/configs/configs.txt})
[ -n "$CONFIG" ] || { echo "no config for task $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "config $CONFIG"
python scripts/train_uma_qm9gwbse.py --config "$CONFIG"
CKPT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['training']['checkpoint_path'])" "$CONFIG")
rm -f "$CKPT"
echo FEWSHOT_DONE
