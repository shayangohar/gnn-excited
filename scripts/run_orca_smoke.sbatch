#!/bin/bash
#SBATCH --job-name=orca_smoke
#SBATCH --output=runs/orca_pilot/%x.%j.out
#SBATCH --error=runs/orca_pilot/%x.%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=FAIL

# TASK-267: gate for ORCA pilot arrays (submit them with --dependency=afterok:<this job>).
# Runs the pilot input (scripts/orca_steom_dlpno.inp) on formaldehyde with $ORCA_DIR and fails
# unless scripts/parse_orca_pilot.py reads all 5 STEOM roots from the output.
set -euo pipefail
cd /projects/bhzu/gnn-excited
ORCA_DIR=${ORCA_DIR:-$HOME/software/orca_6_1_1_linux_x86-64_shared_openmpi418}
WORK=/tmp/$USER/orca_smoke_$SLURM_JOB_ID
mkdir -p "$WORK" runs/orca_pilot; trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/mol.xyz" <<'XYZ'
4
formaldehyde (C=O 1.205, C-H 1.101 A, HCH 116.5 deg)
C   0.000000   0.000000   0.000000
O   0.000000   0.000000   1.205000
H   0.000000   0.936200  -0.579300
H   0.000000  -0.936200  -0.579300
XYZ
cp scripts/orca_steom_dlpno.inp "$WORK/mol.inp"
started=$(date +%s)
( cd "$WORK" && "$ORCA_DIR/orca" mol.inp > mol.out 2>&1 ) || status=$?
echo "ORCA exit ${status:-0} in $(( $(date +%s) - started ))s"
cp "$WORK/mol.out" "runs/orca_pilot/orca_smoke.$SLURM_JOB_ID.orca.out"
grep -m1 "Program Version" "$WORK/mol.out" || true
[ "${status:-0}" -eq 0 ] || { tail -30 "$WORK/mol.out"; exit 1; }
source /etc/profile.d/modules.sh
module purge
source envs/gnn-excited-delta/bin/activate
python - "$WORK/mol.out" <<'PY'
import sys
sys.path.insert(0, "scripts")
from parse_orca_pilot import parse_steom
roots = parse_steom(open(sys.argv[1], errors="replace").read())
print("STEOM roots (eV):", roots)
sys.exit(0 if roots else 1)
PY
echo ORCA_SMOKE_OK
