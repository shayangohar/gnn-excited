#!/bin/bash
#SBATCH --job-name=qcdge_audit_env
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=csn100
#SBATCH --no-requeue
#SBATCH --time=00:30:00

set -euo pipefail
cd /home/mgohar/gnn-excited
mkdir -p runs
source /cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh
conda activate /expanse/lustre/projects/csn100/mgohar/gnn-excited/envs/gnn-excited-gpu
python -m pip install --no-deps "rdkit>=2023.9"
python -c "import rdkit; print(rdkit.__version__)"
