#!/bin/bash
#SBATCH --job-name=cheap_geometry
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-291 E3 (CPU part): 2,000 QM9GWBSE test molecules from SMILES: ETKDGv3+MMFF and GFN2-xTB
# geometries (scripts/build_cheap_geometries.py). The GPU part is scripts/run_cheap_geometry_gpu.sbatch.
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
export TMPDIR=/tmp/$USER/cheapgeom_$SLURM_JOB_ID && mkdir -p $TMPDIR
python scripts/build_cheap_geometries.py --n 2000 --xtb $HOME/software/xtb/bin/xtb --workers 32
rm -rf $TMPDIR
echo CHEAP_GEOMETRY_CPU_DONE
