#!/bin/bash
#SBATCH --job-name=uma_weights
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=bhzu-delta-cpu
#SBATCH --constraint="projects&work"
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# Download one UMA checkpoint from the gated facebook/UMA repo, check its sha256 against the
# repo's LFS record, then smoke-test it with the feature extractor on CPU next to UMA-S.
#   sbatch scripts/download_uma_weights.sbatch                              # uma-s-1p2p1 (TASK-211)
#   sbatch --export=ALL,CKPT=uma-m-1p1 scripts/download_uma_weights.sbatch  # UMA-M (KNOW-326)
# Hugging Face gates access per repo, so the token that fetched uma-s-1p2p1 also covers uma-m-1p1.
# The HF token lives in ~/.config/hf/token on Delta, mode 600: NEVER commit or print it.
set -euo pipefail
cd /projects/bhzu/gnn-excited
CKPT=${CKPT:-uma-s-1p2p1}
SMOKE=${SMOKE:-64}
mkdir -p runs/uma_smoke checkpoints/uma
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/projects/bhzu/gnn-excited/envs/uma-fairchem/bin/python   # full path, not PATH lookup (KNOW-313)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

HF_TOKEN="$(cat ~/.config/hf/token)" CKPT=$CKPT $PY - <<'PYEOF'
import hashlib, os, sys, time
from huggingface_hub import HfApi, snapshot_download
token, name = os.environ["HF_TOKEN"], f"checkpoints/{os.environ['CKPT']}.pt"
started = time.perf_counter()
snapshot_download(repo_id="facebook/UMA", token=token, local_dir="checkpoints/uma",
                  allow_patterns=[name, "config.json", "LICENSE", "USE_POLICY.md"])
print(f"downloaded {name} in {time.perf_counter() - started:.0f}s")
info = HfApi(token=token).model_info("facebook/UMA", files_metadata=True)
lfs = next(s.lfs for s in info.siblings if s.rfilename == name)
want = getattr(lfs, "sha256", None) or lfs["sha256"]
digest = hashlib.sha256()
with open(f"checkpoints/uma/{name}", "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 24), b""):
        digest.update(chunk)
if digest.hexdigest() != want:
    sys.exit(f"SHA256 MISMATCH for {name}: {digest.hexdigest()} != {want}")
print(f"sha256 ok {want}")
PYEOF
ls -lh checkpoints/uma/checkpoints

CKPTS=$CKPT
if [ "$CKPT" != uma-s-1p2p1 ]; then CKPTS="$CKPT uma-s-1p2p1"; fi
for c in $CKPTS; do
  echo "SMOKE $c: pretrained features of the first $SMOKE QM9GWBSE molecules on CPU"
  $PY scripts/extract_uma_features.py --pretrained --checkpoint checkpoints/uma/checkpoints/$c.pt \
    --out runs/uma_smoke/$c --limit $SMOKE --device cpu --num-workers 0 --batch-size 16
done
echo UMA_WEIGHTS_DONE
