#!/bin/bash
#SBATCH --job-name=conformer_study
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-320 wave G conformer study (CPU only): the xTB-trained and B3LYP-trained s17 models predict every
# conformer of data/processed/conformers/pool_xtb_conf.npz (scripts/build_conformer_pool.py) through
# the trainer's inference mode, each built from its own training config; then scripts/conformer_report.py.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs/conformers
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 MPLBACKEND=Agg
PY=/projects/bhzu/gnn-excited/envs/uma-fairchem/bin/python
$PY - <<'PY'
import yaml
for name, config in (("xtb_s17", "configs/uma_qm9gwbse_richind_xtb_s17.yaml"), ("richind_s17", "configs/uma_qm9gwbse_richind_s17.yaml")):
    c = yaml.safe_load(open(config))
    c["training"].update(device="cpu", num_workers=4, batch_size=32)
    c["wandb"]["enabled"] = False
    c.pop("post_inference", None)
    c["inference"] = {"pool_path": "data/processed/conformers/pool_xtb_conf.npz",
                      "checkpoint_paths": [c["training"]["checkpoint_path"]],
                      "out_csvs": [f"runs/conformers/conf.uma_{name}.csv"]}
    yaml.safe_dump(c, open(f"runs/conformers/infer_{name}.yaml", "w"), sort_keys=False)
PY
for name in xtb_s17 richind_s17; do
  $PY scripts/train_uma_qm9gwbse.py --config runs/conformers/infer_${name}.yaml
done
conda deactivate
module purge
source envs/gnn-excited-delta/bin/activate
python scripts/conformer_report.py \
  --pred xtb_trained=runs/conformers/conf.uma_xtb_s17.csv --pred b3lyp_trained=runs/conformers/conf.uma_richind_s17.csv \
  --b3lyp xtb_trained=runs/cheap_geometry/b3lyp.uma_richind_xtb_s17.csv --b3lyp b3lyp_trained=runs/cheap_geometry/b3lyp.uma_richind_s17.csv
echo CONFORMER_STUDY_DONE
