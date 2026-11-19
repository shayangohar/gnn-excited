#!/bin/bash
#SBATCH --job-name=profile_uma
#SBATCH --output=runs/%x.%j.%N.out
#SBATCH --error=runs/%x.%j.%N.err
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --account=bhzu-delta-gpu
#SBATCH --constraint="projects&work"
#SBATCH --no-requeue
#SBATCH --time=01:15:00
#SBATCH --mail-user=shayangohar@mail.fresnostate.edu
#SBATCH --mail-type=END,FAIL

# TASK-336: where does a UMA-S training step spend its time, and which change buys the most?
# Each variant is its own process (scripts/profile_uma_training.py), so one failure does not stop
# the rest. nvidia-smi samples the GPU once a second; the summary matches samples to each
# variant's timed window. 16 CPUs and 60G keep the billing at one A40 (KNOW-229).
set -uo pipefail
cd /projects/bhzu/gnn-excited
OUT=${OUT:-runs/profile_uma}
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /etc/profile.d/modules.sh
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/bhzu/gnn-excited/envs/uma-fairchem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/projects/bhzu/gnn-excited/envs/uma-fairchem/bin/python   # full path (KNOW-313)
P="$PY scripts/profile_uma_training.py --config ${CONFIG:-configs/uma_qm9gwbse_richind_s17.yaml}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -lms 1000 \
  > $OUT/nvsmi.csv &
SMI=$!

run() { local name=$1; shift; echo "== $name: $*"; $P "$@" --out $OUT/$name || echo "VARIANT_FAILED $name"; }
run baseline --mole split --profile-steps 10
run nosync   --mole split --no-sync
run fused    --mole split --fused-adam
run bmm      --mole bmm --check-grads --profile-steps 10
run combined --mole bmm --no-sync --fused-adam
echo "== pair: two baseline processes on one GPU"
$P --mole split --num-workers 6 --out $OUT/pair_a & A=$!
$P --mole split --num-workers 6 --out $OUT/pair_b & B=$!
wait $A || echo "VARIANT_FAILED pair_a"
wait $B || echo "VARIANT_FAILED pair_b"
echo "== compile (time-boxed to 15 min)"
timeout 900 $P --mole bmm --no-sync --fused-adam --compile --warmup 30 --out $OUT/compile || echo "VARIANT_FAILED compile"
kill $SMI 2>/dev/null

OUT=$OUT $PY - <<'EOF'
import csv, datetime, json, os
from pathlib import Path
out = Path(os.environ["OUT"])
samples = []
for row in csv.reader(open(out / "nvsmi.csv")):
    try:
        t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
        samples.append((t, float(row[1]), float(row[2]), float(row[3])))
    except (ValueError, IndexError):
        pass
results = {}
for path in sorted(out.glob("*.json")):
    if path.name == "summary.json":
        continue
    r = json.loads(path.read_text())
    window = [s for s in samples if r["timed_start_epoch"] <= s[0] <= r["timed_end_epoch"]]
    r["gpu_util_pct"] = sum(s[1] for s in window) / len(window) if window else None
    r["gpu_mem_used_MiB"] = max((s[2] for s in window), default=None)
    r["power_W"] = sum(s[3] for s in window) / len(window) if window else None
    results[path.stem] = r
base = results.get("baseline", {}).get("molecules_per_s")
print(f"{'variant':10s} {'ms/step':>8s} {'mol/s':>7s} {'speedup':>7s} {'util%':>6s} {'peakGB':>6s} {'wait%':>6s}")
for name, r in results.items():
    speed = r["molecules_per_s"] / base if base else float("nan")
    util = r["gpu_util_pct"] if r["gpu_util_pct"] is not None else float("nan")
    print(f"{name:10s} {r['ms_per_step']:8.1f} {r['molecules_per_s']:7.1f} {speed:7.2f} {util:6.0f} "
          f"{r['peak_allocated_GB']:6.1f} {100 * r['data_wait_fraction']:6.1f}")
if "pair_a" in results and "pair_b" in results:
    a, b = results["pair_a"], results["pair_b"]
    overlap = max(0.0, min(a["timed_end_epoch"], b["timed_end_epoch"]) - max(a["timed_start_epoch"], b["timed_start_epoch"]))
    span = max(a["timed_end_epoch"], b["timed_end_epoch"]) - min(a["timed_start_epoch"], b["timed_start_epoch"])
    total = a["molecules_per_s"] + b["molecules_per_s"]
    print(f"pair: combined {total:.1f} mol/s = {total / base:.2f}x one baseline process; timed windows overlap {overlap / span:.0%}")
for name in ("baseline", "bmm"):
    if "profile" in results.get(name, {}):
        print(name, "profile", json.dumps(results[name]["profile"]))
if "grad_check" in results.get("bmm", {}):
    print("bmm grad_check", json.dumps(results["bmm"]["grad_check"]))
(out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
EOF
echo PROFILE_UMA_DONE
