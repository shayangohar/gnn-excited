#!/bin/bash
#SBATCH --job-name=omol25_fetch
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
DEST=/work/hdd/bhzu/gnn-excited/data/omol25
mkdir -p "$DEST"
cd "$DEST"
BASE=https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514
for F in train_4M.tar.gz val_neutral.tar.gz; do
  if [ ! -s "$F" ]; then
    echo "[fetch] $F"
    curl -fL --retry 5 --retry-delay 10 -o "$F.part" "$BASE/$F"
    mv "$F.part" "$F"
  fi
  echo "[size] $(du -h "$F" | cut -f1) $F"
done
echo "[smoke] tarball listings:"
tar -tzf train_4M.tar.gz | head -5
tar -tzf val_neutral.tar.gz | head -5
echo FETCH_DONE
