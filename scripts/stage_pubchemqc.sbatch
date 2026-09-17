#!/bin/bash
#SBATCH --job-name=pubchemqc_stage
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

# TASK-210: stage the PubChemQC JCIM-2017 excited-state JSON bundle on Delta.
# Canonical source: https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_2017.html
# 24GB tarball, TD-DFT B3LYP/6-31+G* S1-S10 for 2M+ molecules.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p data/raw/pubchemqc runs
OUT=data/raw/pubchemqc/pubchemqc_jcim2017_jsons.10150017a15274edd1e5ed06ad5831de.tar.gz
URL='https://chibakoudai.sharepoint.com/:u:/s/stair02/Ed9Z16k0ctJKk9nQLMYFHYUBp_E9zerPApRaWTrOIYN-Eg?download=1'

if [ ! -f "$OUT" ]; then
  wget --tries=5 --timeout=120 --continue "$URL" -O "$OUT"
fi
ls -lh "$OUT"

echo "$OUT"
echo '10150017a15274edd1e5ed06ad5831de  pubchemqc_jcim2017_jsons.10150017a15274edd1e5ed06ad5831de.tar.gz' > data/raw/pubchemqc/MD5SUM
echo '0eef77c2818c190ef8183cd3e92110f021c7c217fe69686af7b3bfe0a760dddd  pubchemqc_jcim2017_jsons.10150017a15274edd1e5ed06ad5831de.tar.gz' > data/raw/pubchemqc/SHA256SUM
(cd data/raw/pubchemqc && md5sum -c MD5SUM && sha256sum -c SHA256SUM)

# Probe contents without full extraction (full extract deferred to converter step).
tar -tzf "$OUT" | head -n 5
tar -tzf "$OUT" | wc -l
