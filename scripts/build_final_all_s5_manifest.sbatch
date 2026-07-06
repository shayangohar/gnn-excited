#!/bin/bash
#SBATCH --job-name=qcdge_s5_manifest
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --account=csn100
#SBATCH --no-requeue
#SBATCH --time=08:00:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /home/mgohar/gnn-excited
mkdir -p data/processed runs
source /cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh
conda activate /expanse/lustre/projects/csn100/mgohar/gnn-excited/envs/gnn-excited-gpu
python scripts/build_manifest.py --hdf5 data/final_all.hdf5 --out data/processed/final_all_manifest_s5.csv --singlets 5 --triplets 0 --progress-every 5000
