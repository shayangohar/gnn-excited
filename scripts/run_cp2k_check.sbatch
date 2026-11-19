#!/bin/bash
#SBATCH --job-name=cp2k_check
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-328 / KNOW-324: is CP2K a usable GW-BSE label source on Delta? Delta has no CP2K module, and the
# conda-forge cp2k 2026.2 builds do not run here (they pin libxsmm 2.0.0, which needs GLIBC_2.38 > Delta's 2.34,
# and link libint2.so.2 while the solver installs libint 2.14 / libint2.so.3). So this pulls the official
# container cp2k/cp2k:2026.2_openmpi_generic_psmp once into envs/cp2k_2026.2.sif and runs CP2K through
# Apptainer: first the CP2K manual's BSE_H2 example (reference S1 12.0361 eV), then BSE@evGW0@PBE/aug-cc-pVDZ
# singlets and triplets on three QM9GWBSE test molecules (3, 6 and 9 heavy atoms) and on the smallest C/H/N/O/S
# molecule of the OE62H dev half (scripts/cp2k_gwbse.py), with wall time, BSE memory estimate and the
# difference to the labels.
set -euo pipefail
cd /projects/bhzu/gnn-excited
SIF=/projects/bhzu/gnn-excited/envs/cp2k_2026.2.sif
export APPTAINER_CACHEDIR=/tmp/$USER/apptainer_$SLURM_JOB_ID && mkdir -p $APPTAINER_CACHEDIR
if [ ! -f $SIF ]; then
  apptainer pull $SIF docker://cp2k/cp2k:2026.2_openmpi_generic_psmp
fi
# --cleanenv: the container's OpenMPI (PRRTE) otherwise sees SLURM_* and tries srun (not in the image) with 1 slot.
# The image's launch entrypoint sets PATH, libraries and MPI settings.
RUN="apptainer exec --cleanenv --env OMP_NUM_THREADS=1,CP2K_DATA_DIR=/opt/cp2k/install/share/cp2k/data --bind /projects $SIF /opt/cp2k/install/bin/launch"
MPI="mpirun --bind-to none --map-by :OVERSUBSCRIBE"
$RUN bash -c 'which cp2k.psmp mpirun; ls $CP2K_DATA_DIR | head -5; cp2k.psmp --version | head -30'
W=runs/cp2k_check; mkdir -p $W/h2
cp data/external/cp2k/BSE_H2.inp data/external/cp2k/BASIS-aug $W/h2/
start=$(date +%s)
(cd $W/h2 && $RUN $MPI -n 4 cp2k.psmp -i BSE_H2.inp -o BSE_H2.out) || echo "H2 example failed"
echo "H2 example: $(( $(date +%s) - start )) s"
grep -E "Singlet +-ABBA-" $W/h2/BSE_H2.out | head -4 || true
PY=/projects/bhzu/gnn-excited/envs/gnn-excited-delta/bin/python
EXT=$($PY - <<'PY'
import csv
split = {r["molecule_key"]: r["half"] for r in csv.DictReader(open("configs/splits/external_dev_test.csv"))}
els = lambda r: set(r["elements"].split("-"))
rows = [r for r in csv.DictReader(open("data/processed/external_testsets/labels.csv"))
        if r["set"] == "OE62H" and split.get(r["molecule_key"]) == "dev" and "S" in els(r) and els(r) <= {"C", "H", "N", "O", "S"}]
print(min(rows, key=lambda r: int(r["natoms"]))["molecule_key"])
PY
)
echo "OE62H molecule: $EXT"
$PY scripts/cp2k_gwbse.py --cmd "$RUN $MPI -n 64 cp2k.psmp" --basis data/external/cp2k/BASIS-aug-ccpvdz \
  --qm9-heavy 3 6 9 --external "$EXT"
rm -rf $APPTAINER_CACHEDIR
echo CP2K_CHECK_DONE
