#!/bin/bash
#SBATCH --job-name=orca_pilot
#SBATCH --output=runs/orca_pilot/%x.%A_%a.out
#SBATCH --error=runs/orca_pilot/%x.%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-262 / KNOW-258: serial STEOM-DLPNO-CCSD/def2-TZVP, 5 singlet roots (scripts/orca_steom_dlpno.inp),
# on one QM9GWBSE train molecule per array task (LIST = file of molecule keys). Serial avoids
# matching ORCA's OpenMPI build; billing is max(2 cores, 8 GB x 0.512) ~ 4.1 core-h per hour.
# TASK-267: ORCA_DIR is the submitting user's own ORCA 6.1.1 install (personal academic licence;
# keep it owner-only). Gate new arrays on scripts/run_orca_smoke.sbatch.
set -euo pipefail
cd /projects/bhzu/gnn-excited
ORCA_DIR=${ORCA_DIR:-$HOME/software/orca_6_1_1_linux_x86-64_shared_openmpi418}
LIST=${LIST:?set LIST}
KEY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -n "$KEY" ] || { echo "no key for task $SLURM_ARRAY_TASK_ID"; exit 1; }
OUT=data/processed/orca_pilot/out
mkdir -p "$OUT"
WORK=/tmp/$USER/orca_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WORK"; trap 'rm -rf "$WORK"' EXIT
cp "data/processed/orca_pilot/xyz/$KEY.xyz" "$WORK/mol.xyz"
cp scripts/orca_steom_dlpno.inp "$WORK/mol.inp"
started=$(date +%s)
( cd "$WORK" && "$ORCA_DIR/orca" mol.inp > mol.out 2>&1 ) || status=$?
elapsed=$(( $(date +%s) - started ))
cp "$WORK/mol.out" "$OUT/$KEY.out"
echo "{\"key\": \"$KEY\", \"seconds\": $elapsed, \"exit\": ${status:-0}, \"node\": \"$(hostname)\", \"cpus\": $SLURM_CPUS_PER_TASK, \"orca\": \"$(basename "$ORCA_DIR")\"}" > "$OUT/$KEY.timing.json"
echo "ORCA $KEY exit ${status:-0} in ${elapsed}s"
exit ${status:-0}
