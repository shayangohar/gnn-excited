#!/bin/bash
#SBATCH --job-name=omol25_conv
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
DEST=/work/hdd/bhzu/gnn-excited/data/omol25
cd /projects/bhzu/gnn-excited
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
STAGE=${STAGE:-probe}
if [ "$STAGE" = "probe" ]; then
  if [ ! -d "$DEST/neutral_val/neutral_val" ]; then tar -xzf "$DEST/neutral_val.tar.gz" -C "$DEST/neutral_val"; fi
  python scripts/convert_omol25_lmdb.py \
    --lmdb-dir "$DEST/neutral_val" \
    --out-h5 data/processed/omol25/val_gap.h5 \
    --manifest data/processed/omol25/val_gap_manifest.csv \
    --max-atoms 32
else
  if [ ! -d "$DEST/train_4M/train_4M" ]; then
    echo "[extract] train_4M"; mkdir -p "$DEST/train_4M"
    tar -xzf "$DEST/train_4M.tar.gz" -C "$DEST/train_4M"
  fi
  python scripts/convert_omol25_lmdb.py \
    --lmdb-dir "$DEST/train_4M" \
    --out-h5 data/processed/omol25/train4m_gap.h5 \
    --manifest data/processed/omol25/train4m_gap_manifest.csv \
    --max-atoms 32
fi
echo CONVERT_STAGE_DONE
