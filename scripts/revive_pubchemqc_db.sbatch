#!/bin/bash
#SBATCH --job-name=pqc_db_revive
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

# TASK-210: revive the PubChemQC postgres:12.3 data dir under apptainer and
# report schema + row counts (extraction of the JSON records comes next).
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs tmp/apptainer_cache tmp/apptainer_tmp
export APPTAINER_CACHEDIR=/projects/bhzu/gnn-excited/tmp/apptainer_cache
# mksquashfs segfaults (139) with node-local /tmp during OCI->SIF build;
# point all scratch space at the roomy project filesystem.
export APPTAINER_TMPDIR=/projects/bhzu/gnn-excited/tmp/apptainer_tmp
export TMPDIR=/projects/bhzu/gnn-excited/tmp/apptainer_tmp
export SINGULARITY_TMPDIR=/projects/bhzu/gnn-excited/tmp/apptainer_tmp
export PGPORT=55433
export PGHOST=/tmp/pqc_sock
mkdir -p "$PGHOST"
DATADIR=/projects/bhzu/gnn-excited/data/raw/pubchemqc/pgdata/postgrest-docker-compose/data/postgresql/lib
IMG=tmp/postgres_12.3.sif
if [ ! -f "$IMG" ]; then
  apptainer pull "$IMG" docker://postgres:12.3
fi
# Server in background; stop it on exit.
apptainer exec --bind "$DATADIR:/var/lib/postgresql/data,$PGHOST:/var/run/postgresql" "$IMG" \
  pg_ctl -D /var/lib/postgresql/data -o "-c listen_addresses='' -c unix_socket_directories=/var/run/postgresql -p $PGPORT" -l /tmp/pg.log start
trap 'apptainer exec "$IMG" pg_ctl -D /var/lib/postgresql/data stop || true' EXIT
sleep 5
export PGPASSWORD='3oAXwRkZ5r809Tj1FVbnF)JrQ'
PSQL="apptainer exec $IMG psql -h $PGHOST -p $PGPORT -U pgrest -d db -At"
$PSQL -c '\dt public.*'
$PSQL -c "SELECT tablename FROM pg_tables WHERE schemaname='public';"
for t in b3lyp molecules dataCompound compound; do
  $PSQL -c "SELECT count(*) AS ${t}_count FROM public.\"$t\";" 2>/dev/null || echo "no table $t"
done
echo REVIVE_INSPECT_DONE
