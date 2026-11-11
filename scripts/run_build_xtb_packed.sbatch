#!/bin/bash
#SBATCH --job-name=build_xtb_packed
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-310 wave F (cheap-geometry fix, KNOW-308): every QM9GWBSE molecule rebuilt from its SMILES
# (ETKDGv3 + MMFF94, then GFN2-xTB), same keys/labels/split -> data/processed/qm9gwbse_xtb/.
# The E3 run did 2,000 molecules in 73 s on 32 cores, so ~134k should take about 1.5 h on 64.
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
export TMPDIR=/tmp/$USER/xtbpack_$SLURM_JOB_ID && mkdir -p $TMPDIR
python scripts/build_xtb_packed.py --xtb $HOME/software/xtb/bin/xtb --workers 64 ${LIMIT:+--limit $LIMIT} \
  --out ${OUT:-data/processed/qm9gwbse_xtb/qm9gwbse_xtb_packed.npz}
rm -rf $TMPDIR
echo BUILD_XTB_PACKED_DONE
