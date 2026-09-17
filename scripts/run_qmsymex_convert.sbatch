#!/bin/bash
#SBATCH --job-name=qmsymex_conv
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-210: QM-symex xyz -> project HDF5 + manifest (schema qmsymex-st-v1).
set -euo pipefail
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
STAGE=${STAGE:-probe}
if [ "$STAGE" = "probe" ]; then
  python scripts/convert_qmsymex.py \
    --xyz-dir data/raw/qmsymex/QM_symex \
    --out-h5 data/processed/qmsymex/probe_st.h5 \
    --manifest data/processed/qmsymex/probe_st_manifest.csv \
    --limit 500
else
  python scripts/convert_qmsymex.py \
    --xyz-dir data/raw/qmsymex/QM_symex \
    --out-h5 data/processed/qmsymex/train_st.h5 \
    --manifest data/processed/qmsymex/train_st_manifest.csv
fi
echo CONVERT_STAGE_DONE
