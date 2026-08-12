#!/bin/bash
#SBATCH --job-name=qcdge_audit_env
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
source /etc/profile.d/modules.sh
module purge
module load cudatoolkit/25.3_12.8
source /projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/activate
python -m pip install --no-deps "rdkit>=2023.9"
python -c "import rdkit; print(rdkit.__version__)"
