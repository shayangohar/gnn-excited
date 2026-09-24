#!/bin/bash
#SBATCH --job-name=pqc_sample
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-210 follow-up: dump 3 sample JSON documents + key inventory, so the
# full COPY-out can be designed (target: cid, geometry, S1-S10 energies/f).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs
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
export PGPASSWORD='3oAXwRkZ5r809Tj1FVbnF)JrQ'
PSQL="$PGBIN/psql -h $PGHOST -p $PGPORT -U pgrest -d db -At"
$PSQL -c "SELECT jsonb_object_keys(data) FROM public.b3lyp LIMIT 1;"
$PSQL -c "COPY (SELECT data FROM public.b3lyp LIMIT 3) TO STDOUT;" > runs/pqc_sample.jsonl
wc -c runs/pqc_sample.jsonl
head -c 1500 runs/pqc_sample.jsonl
echo
echo SAMPLE_DONE
