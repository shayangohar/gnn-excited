#!/bin/bash
#SBATCH --job-name=xmace_qm9_20k
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
#SBATCH --time=24:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs/xmace_qm9gwbse_20k checkpoints/xmace_qm9gwbse_20k data/processed/qm9gwbse/xmace_20k
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER/gnn-excited-$SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source /etc/profile.d/modules.sh
module purge
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
test "$(git -C /projects/bhzu/software/X-MACE rev-parse HEAD)" = a92a3e9fa1639f3a94cadc00d3b16542c2b760b5
git -C /projects/bhzu/software/X-MACE apply --reverse --check --unidiff-zero /projects/bhzu/gnn-excited/patches/xmace_energy_only.patch
python -c 'import mace, torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(mace.__version__)'
python scripts/export_qm9gwbse_xmace.py \
  --hdf5 data/processed/qm9gwbse/qm9gwbse.h5 \
  --manifest data/processed/qm9gwbse/manifest.csv \
  --output-dir data/processed/qm9gwbse/xmace_20k \
  --limit 20000 \
  --seed 17
python /projects/bhzu/software/X-MACE/scripts/run_train.py \
  --name=xmace_qm9gwbse_20k \
  --train_file=data/processed/qm9gwbse/xmace_20k/train.xyz \
  --valid_file=data/processed/qm9gwbse/xmace_20k/val.xyz \
  --test_file=data/processed/qm9gwbse/xmace_20k/test.xyz \
  --energy_key=REF_energy \
  --E0s='{1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}' \
  --model=AutoencoderExcitedMACE \
  --n_energies=5 \
  --num_permutational_invariant=16 \
  --r_max=5.0 \
  --batch_size=32 \
  --correlation=3 \
  --max_num_epochs=50 \
  --patience=10 \
  --ema \
  --lr=0.0001 \
  --ema_decay=0.99 \
  --default_dtype=float32 \
  --device=cuda \
  --hidden_irreps='128x0e + 128x1o' \
  --MLP_irreps=128x0e \
  --num_radial_basis=8 \
  --num_interactions=2 \
  --scaling=no_scaling \
  --energy_weight=1.0 \
  --forces_weight=0.0 \
  --dipoles_weight=0.0 \
  --nacs_weight=0.0 \
  --socs_weight=0.0 \
  --work_dir=runs/xmace_qm9gwbse_20k \
  --checkpoints_dir=checkpoints/xmace_qm9gwbse_20k \
  --results_dir=runs/xmace_qm9gwbse_20k/results \
  --log_dir=runs/xmace_qm9gwbse_20k/logs \
  --model_dir=runs/xmace_qm9gwbse_20k/models \
  --error_table=EnergyNacsDipoleMAE \
  --wandb \
  --wandb_project=gnn-excited \
  --wandb_entity=shayangohar2007-fresno-state \
  --wandb_name=xmace_qm9gwbse_20k
