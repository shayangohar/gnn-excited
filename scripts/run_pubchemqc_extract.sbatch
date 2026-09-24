#!/bin/bash
#SBATCH --job-name=pqc_extract
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

# TASK-217: stream public.b3lyp (3.98M rows) from the revived pg12 server
# straight into training HDF5 (schema pubchemqc-st-v1). LIMIT=0 means all.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs data/processed/pubchemqc
export PGPORT=55433
export PGHOST=/tmp/pqc_sock
mkdir -p "$PGHOST"
DATADIR=/projects/bhzu/gnn-excited/data/raw/pubchemqc/pgdata/postgrest-docker-compose/data/postgresql/lib
PGBIN=/projects/bhzu/gnn-excited/tmp/pg12_sandbox/usr/lib/postgresql/12/bin
PGLIB=/projects/bhzu/gnn-excited/tmp/pg12_sandbox/usr/lib/postgresql/12/lib:/projects/bhzu/gnn-excited/tmp/pg12_sandbox/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="$PGLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PGBIN/pg_ctl" -D "$DATADIR" -o "-c listen_addresses='' -c unix_socket_directories='$PGHOST' -p $PGPORT" -l /tmp/pg.log start
trap '"$PGBIN/pg_ctl" -D "$DATADIR" stop || true' EXIT
sleep 5
source envs/gnn-excited-delta/bin/activate
LIMIT=${LIMIT:-0}
python scripts/extract_pubchemqc.py \
  --socket-dir "$PGHOST" --port "$PGPORT" --password '3oAXwRkZ5r809Tj1FVbnF)JrQ' \
  --out-h5 data/processed/pubchemqc/train_st.h5 \
  --manifest data/processed/pubchemqc/train_st_manifest.csv \
  --limit "$LIMIT"
echo EXTRACT_DONE
