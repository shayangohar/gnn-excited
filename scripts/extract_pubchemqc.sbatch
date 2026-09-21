#!/bin/bash
#SBATCH --job-name=pubchemqc_extract
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

# TASK-210: extract the 24GB PubChemQC postgres-docker tarball (checksummed
# in the staging job) and report its layout + PG version for DB revival.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p data/raw/pubchemqc/pgdata runs
TARBALL=data/raw/pubchemqc/postgrest-docker-compose-b3lyp_gin_and_ginp_index-20200725-230753.4168985229d8981ed5ed6c48ea92071c.tar.xz
tar -xJf "$TARBALL" -C data/raw/pubchemqc/pgdata
du -sh data/raw/pubchemqc/pgdata
find data/raw/pubchemqc/pgdata -maxdepth 4 | head -n 30
cat data/raw/pubchemqc/pgdata/postgrest-docker-compose/data/postgresql/lib/PG_VERSION 2>/dev/null || true
echo EXTRACT_DONE
