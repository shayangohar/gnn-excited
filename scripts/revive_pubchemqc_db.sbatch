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
IMG=tmp/pg12_sandbox
if [ ! -d "$IMG" ]; then
  # NOTE: `apptainer pull` (OCI->SIF) segfaults mksquashfs on this system
  # (exit 139, both node-local and lustre tmp); --sandbox unpacks layers
  # without squashfs and sidesteps it entirely.
  apptainer build --sandbox "$IMG" docker://postgres:12.3
fi
# Run the sandbox's postgres binaries NATIVELY (no container mount
# namespaces: apptainer --bind mounts proved unreliable here). The sandbox
# dir is just files on lustre; host RHEL9 + LD_LIBRARY_PATH covers deps.
PGBIN=/projects/bhzu/gnn-excited/tmp/pg12_sandbox/usr/lib/postgresql/12/bin
PGLIB=/projects/bhzu/gnn-excited/tmp/pg12_sandbox/usr/lib/postgresql/12/lib:/projects/bhzu/gnn-excited/tmp/pg12_sandbox/usr/lib/x86_64-linux-gnu
ls "$PGBIN/postgres" "$PGBIN/psql" "$PGBIN/pg_ctl"
export LD_LIBRARY_PATH="$PGLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$PGBIN/postgres" 2>/dev/null | grep 'not found' || echo DEPS-OK
"$PGBIN/pg_ctl" -D "$DATADIR" -o "-c listen_addresses='' -c unix_socket_directories='$PGHOST' -p $PGPORT" -l /tmp/pg.log start
trap '"$PGBIN/pg_ctl" -D "$DATADIR" stop || true' EXIT
sleep 5
export PGPASSWORD='3oAXwRkZ5r809Tj1FVbnF)JrQ'
PSQL="$PGBIN/psql -h $PGHOST -p $PGPORT -U pgrest -d db -At"
$PSQL -c '\dt public.*'
$PSQL -c "SELECT tablename FROM pg_tables WHERE schemaname='public';"
for t in b3lyp molecules dataCompound compound; do
  $PSQL -c "SELECT count(*) AS ${t}_count FROM public.\"$t\";" 2>/dev/null || echo "no table $t"
done
echo REVIVE_INSPECT_DONE
