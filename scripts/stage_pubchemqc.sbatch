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

# TASK-210: stage PubChemQC JCIM-2017 on Delta.
# NOTE: the 24GB SharePoint JSON bundle is unfetchable headless (HTML
# interstitial, Graph API refuses cross-tenant guest link even with user
# OAuth). The RIKEN download page lists an Internet Archive mirror of the
# PostgreSQL docker database holding ALL the JSON files (23.3GB) — plain
# https, no auth. Source: https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_2017.html
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p data/raw/pubchemqc runs
OUT=data/raw/pubchemqc/postgrest-docker-compose-b3lyp_gin_and_ginp_index-20200725-230753.4168985229d8981ed5ed6c48ea92071c.tar.xz
URL='https://archive.org/download/pubchemqc-jcim2017-postgresql-docker/postgrest-docker-compose-b3lyp_gin_and_ginp_index-20200725-230753.4168985229d8981ed5ed6c48ea92071c.tar.xz'

if [ ! -f "$OUT" ]; then
  wget --tries=5 --timeout=120 --continue "$URL" -O "$OUT"
fi
ls -lh "$OUT"

echo '4168985229d8981ed5ed6c48ea92071c  postgrest-docker-compose-b3lyp_gin_and_ginp_index-20200725-230753.4168985229d8981ed5ed6c48ea92071c.tar.xz' > data/raw/pubchemqc/MD5SUM
echo 'aee32e5d4915e4ae47e1a972d367460fe0dc1e61c49a5c693f3d63bbe8b002d5  postgrest-docker-compose-b3lyp_gin_and_ginp_index-20200725-230753.4168985229d8981ed5ed6c48ea92071c.tar.xz' > data/raw/pubchemqc/SHA256SUM
(cd data/raw/pubchemqc && md5sum -c MD5SUM && sha256sum -c SHA256SUM)

# Probe structure without full extraction (no `| head` under pipefail: SIGPIPE=141).
tar -tf "$OUT" > data/raw/pubchemqc/contents.txt || true
wc -l data/raw/pubchemqc/contents.txt
head -n 20 data/raw/pubchemqc/contents.txt || true
