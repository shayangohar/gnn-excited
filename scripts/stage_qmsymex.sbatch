#!/bin/bash
#SBATCH --job-name=qmsymex_stage
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-210 (pivoted from PubChemQC SharePoint: guest links return an HTML
# interstitial to headless wget/curl, so the 24GB bundle is unfetchable from
# Delta without interactive OAuth): stage QM-symex instead.
# Source: Liang et al. Sci Data 7, 400 (2020), Figshare doi 10.6084/m9.figshare.12815276.
# 173k symmetric organics, first TEN singlet AND triplet transitions
# (energy eV, wavelength, oscillator strength, spin) per molecule in xyz files.
# Triplets included -> serves joint S+T directly. NOTE: curl needs -L
# (ndownloader 302-redirects; plain HEAD is 403-rejected).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p data/raw/qmsymex runs
OUT=data/raw/qmsymex/QM_symex.tar

if [ ! -f "$OUT" ]; then
  curl -L --retry 5 --retry-all-errors --continue-at - \
    'https://ndownloader.figshare.com/files/24310253' --output "$OUT"
fi
ls -lh "$OUT"

# Integrity probe, then extract.
tar -tf "$OUT" > data/raw/qmsymex/contents.txt
wc -l data/raw/qmsymex/contents.txt
grep -c '\.xyz$' data/raw/qmsymex/contents.txt || true
tar -xf "$OUT" -C data/raw/qmsymex
ls data/raw/qmsymex | head
XYZ=$(find data/raw/qmsymex -name 'QM_symex_*.xyz' | head -n 1)
echo "example: $XYZ"
head -n 12 "$XYZ"
find data/raw/qmsymex -name 'QM_symex_*.xyz' | wc -l
