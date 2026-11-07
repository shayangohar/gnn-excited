#!/bin/bash
#SBATCH --job-name=cheap_geometry_gpu
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=06:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-291 E3 (GPU part): UMA-relax the ETKDG geometries, run the nine production members on the
# b3lyp / etkdg / xtb / uma pools, then scripts/cheap_geometry_report.py.
set -euo pipefail
cd /projects/bhzu/gnn-excited
G=data/processed/cheap_geometry
P=runs/cheap_geometry
mkdir -p $P/configs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
ORIG_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python scripts/relax_uma_pool.py --pool $G/pool_etkdg.npz --out $G/pool_uma.npz --steps 200
for src in b3lyp etkdg xtb uma; do
  for family in uma_external_eval_inference uma_st_external_eval_inference; do
    cfg=$P/configs/$src.$family.yaml
    sed -e "s#data/processed/external_testsets/pool.npz#$G/pool_$src.npz#" -e "s#runs/head_to_head/ext_#$P/$src.#" configs/$family.yaml > $cfg
    python scripts/train_uma_qm9gwbse.py --config $cfg
  done
done
conda deactivate
export LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH
module purge
source envs/gnn-excited-delta/bin/activate
for src in b3lyp etkdg xtb uma; do
  args=()
  for s in 17 18 19; do args+=(--checkpoint checkpoints/distill_enc_qm9gwbse_finetune_s$s.pt --out $P/$src.distill_s$s.csv); done
  python scripts/predict_packed_visnet.py --pool $G/pool_$src.npz "${args[@]}"
done
python scripts/cheap_geometry_report.py
echo CHEAP_GEOMETRY_GPU_DONE
