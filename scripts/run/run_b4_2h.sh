#!/bin/bash
# The batch-4 arm, capped to a ~2 h health proof.
#
# This reconstructs the run that logs/match_b4.log shows was in flight when the
# box died on 2026-08-08 at ~20:39 (it had reached step 300 of 3350). That run
# used --save-every 500 and had therefore written NOTHING when it died, so the
# whole 36 minutes was lost. Two changes here, both deliberate:
#
#   1. --iters 950 instead of 3350, sized to ~2 h from the measured rates in
#      match_b4.log: 145 s startup + 150 warmup steps at 3.2 s/it + 800 steps
#      at 8.2 s/it = 7185 s = 2.00 h.
#   2. --save-every 250 instead of 500, and everything under /workspace, which
#      is the only durable mount. The previous run wrote checkpoints to
#      /ckpt_archive and eval output to /exp; both were outside /workspace and
#      both were erased by the rebuild.
#
# Everything else matches match_b4.log exactly: 4 ranks x accum 1 = effective
# batch 4, the 2000-clip data/teacher bank, the 14B teacher under FSDP, init
# from checkpoints/dmd, guidance 5.0, critic warmup 150, roll-depth 0.
set -u
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

ITERS=${ITERS:-950}
ACCUM=${ACCUM:-1}
NPROC=${NPROC:-4}
OUT=${OUT:-checkpoints/t14b_b4_2h}
LOG=${LOG:-logs/b4_2h.log}
KEEP=${KEEP:-2}          # how many step*.pt to retain; see the disk note below

mkdir -p logs out "$OUT"

# Preflight. A missing input costs a ~2.5 minute startup before it surfaces as
# a stack trace, so fail here with the name of the thing that is missing.
for f in checkpoints/dmd/latest.pt checkpoints/dmd/critic_latest.pt data/prompts.pt; do
  test -e "$f" || { echo "FATAL: missing $f"; exit 1; }
done
for d in checkpoints/wan21_13b checkpoints/wan21_14b wan21_repo data/teacher; do
  test -d "$d" || { echo "FATAL: missing $d -- run ./scripts/run/setup_16gpu.sh"; exit 1; }
done
NCLIPS=$(ls data/teacher/*.pt 2>/dev/null | wc -l)
echo "=== $NCLIPS teacher clips | $NPROC ranks x accum $ACCUM = effective batch $((NPROC * ACCUM))"
echo "=== $ITERS iters | out $OUT | free $(df -h /workspace | awk 'NR==2{print $4}')"

# Disk janitor. Each save writes latest.pt AND step<N>.pt, 11.3 GB apiece, and
# train_dmd.py also saves unconditionally at --iters. At save-every 250 that is
# four step files for a 950-iter run: 45 GB of step files on top of latest.pt
# and the optimizer shards, against 65 GB free. Retaining the newest $KEEP caps
# the run at roughly 43 GB. Raise KEEP if you have freed space.
janitor () {
  while true; do
    n=$(ls -1t "$OUT"/step*.pt 2>/dev/null | wc -l)
    if [ "$n" -gt "$KEEP" ]; then
      ls -1t "$OUT"/step*.pt | tail -n +$((KEEP + 1)) | while read -r f; do
        echo "$(date -u +%H:%M:%S) janitor: pruning $(basename "$f")" >> logs/b4_2h_janitor.log
        rm -f "$f"
      done
      df -h /workspace | tail -1 >> logs/b4_2h_janitor.log
    fi
    sleep 60
  done
}
janitor & JANITOR=$!
trap 'kill $JANITOR 2>/dev/null' EXIT

torchrun \
  --nproc_per_node "$NPROC" --master_port 29682 \
  scripts/train_dmd.py \
    --data data/teacher \
    --init checkpoints/dmd/latest.pt \
    --init-critic checkpoints/dmd/critic_latest.pt \
    --teacher-ckpt checkpoints/wan21_14b --teacher-size 14B --fsdp \
    --accum "$ACCUM" \
    --guidance 5.0 \
    --iters "$ITERS" --critic-warmup 150 \
    --roll-depth 0 \
    --save-every 250 --sample-every 100 --log-every 10 \
    --out "$OUT" ${RESUME:+--resume} \
  2>&1 | tee "$LOG"

kill $JANITOR 2>/dev/null
echo "=== TRAIN DONE $(date -u +%H:%M:%S) ==="
echo "=== free $(df -h /workspace | awk 'NR==2{print $4}') | checkpoints:"
ls -la "$OUT"
