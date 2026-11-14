#!/bin/bash
#SBATCH --job-name=gap_vs_s1
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

# How our qsGW-BSE targets relate to rCEG's B3LYP HOMO-LUMO gap (KNOW-293, KNOW-309).
# 1. CPU inference: uma_qm9gwbse_richind_he_heads s17-19 on rCEG's real-label pool at label geometries
#    (ref.npz: 1,693 train/val + 424 test) and on the 424 test molecules at ETKDG+UFF geometries.
# 2. scripts/gap_vs_s1_report.py -> runs/gap_vs_s1_report.json.
set -euo pipefail
cd /projects/bhzu/gnn-excited
mkdir -p runs/gap_vs_s1
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 MPLBACKEND=Agg
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/uma_qm9gwbse_richind_he_heads_s17.yaml"))
c["training"].update(device="cpu", num_workers=4, batch_size=32)
c["wandb"]["enabled"] = False
c.pop("post_inference", None)
for name, pool in (("ref", "data/processed/rceg_compare/ref.npz"), ("smiles", "data/processed/rceg_compare/pool_smiles.npz")):
    c["inference"] = {"pool_path": pool,
                      "checkpoint_paths": [f"checkpoints/uma_qm9gwbse_richind_he_heads_s{s}.pt" for s in (17, 18, 19)],
                      "out_csvs": [f"runs/gap_vs_s1/{name}_he_heads_s{s}.csv" for s in (17, 18, 19)]}
    yaml.safe_dump(c, open(f"runs/gap_vs_s1/infer_{name}.yaml", "w"), sort_keys=False)
PY
for name in smiles ref; do
  python scripts/train_uma_qm9gwbse.py --config runs/gap_vs_s1/infer_${name}.yaml
done
conda deactivate
module purge
source envs/gnn-excited-delta/bin/activate
python scripts/gap_vs_s1_report.py
echo GAP_VS_S1_DONE
