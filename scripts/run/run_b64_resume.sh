#!/bin/bash
# Resume the batch-64 run from step 750 and carry it to 3000.
#
# The run reached step 770 on 2026-08-08 and was stopped; checkpoints/t14b_b64
# holds latest.pt at step 750 plus all eight opt_rank shards at world_size 8,
# so this picks up the AdamW moments, the EMA, the LoRA critic and the step
# counter rather than restarting. 2250 iterations remain at the measured
# 67 s/it on 8 ranks, so about 42 h.
#
# Three things this handles that a bare `RESUME=1 ./scripts/run/run_16gpu.sh` does not:
#
# 1. PRESERVING STEP 750. train_dmd.py resumes from OUT/latest.pt and then
#    OVERWRITES that same path at the first save. Step 750 is the only b64
#    checkpoint that survived the container rebuild and it has already been
#    evaluated, so it is copied to step000750.pt first and the pruner below is
#    told never to touch it.
#
# 2. NOT CLOBBERING THE SURVIVING EVAL. run_16gpu.sh runs an evaluation after
#    training that writes out/b64_w{world}_{tag}; those directories still hold
#    the metrics.json files recovered from the crash. This sets TRAIN_ONLY=1.
#    Score afterwards with a different --prefix.
#
# 3. DISK. Nine saves remain at --save-every 250. Each writes an 11.35 GB
#    step<N>.pt and rewrites eight 2.38 GB optimizer shards via .tmp + rename,
#    so a save transiently needs ~30 GB free. Nine step files would be 102 GB
#    against 65 GB free. The pruner clears the rolling step file before each
#    save, keeping step000750.pt and the newest, which holds ~23 GB of margin.
set -u
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

OUT=${OUT:-checkpoints/t14b_b64}
ITERS=${ITERS:-3000}
ACCUM=${ACCUM:-8}
NPROC=${NPROC:-8}
LOG=${LOG:-logs/b64_resume.log}
PLOG=logs/b64_resume_pruner.log

test -e "$OUT/latest.pt" || { echo "FATAL: no $OUT/latest.pt to resume from"; exit 1; }
for r in 0 1 2 3 4 5 6 7; do
  test -e "$OUT/opt_rank00$r.pt" || { echo "FATAL: missing $OUT/opt_rank00$r.pt -- a resume needs the same world size (8) that wrote the shards"; exit 1; }
done

STEP=$(python3 -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['step'])" "$OUT/latest.pt")

# Name the preserved copy after the step it actually holds. This was hardcoded to
# step000750.pt for the 750 -> 3000 run, which was correct exactly once: extending
# from 3000 would have copied the step-3000 weights to a file called
# step000750.pt, and the pruner would then have protected that misnamed file for
# the whole run. Derive it instead.
KEEPSTEP=${KEEPSTEP:-step$(printf '%06d' "$STEP").pt}

[ "$ITERS" -gt "$STEP" ] || { echo "FATAL: --iters $ITERS is not beyond the checkpoint's step $STEP; nothing to do"; exit 1; }
echo "=== resuming $OUT from step $STEP -> $ITERS | $NPROC ranks x accum $ACCUM = effective batch $((NPROC*ACCUM))"
echo "=== $((ITERS-STEP)) iters left at ~67 s/it = $(python3 -c "print(f'{($ITERS-$STEP)*67/3600:.1f}')") h | free $(df -h /workspace | awk 'NR==2{print $4}')"

# 1. Preserve the checkpoint we are resuming from, before anything can overwrite it.
if [ ! -e "$OUT/$KEEPSTEP" ]; then
  echo "=== preserving step $STEP as $KEEPSTEP before the first save overwrites latest.pt"
  cp "$OUT/latest.pt" "$OUT/$KEEPSTEP"
fi

# 3. Pre-save pruner. Same reasoning as presave_pruner.sh: freeing space after a
#    save is useless, because the save is what runs out of space.
pruner () {
  say () { echo "$(date -u +%H:%M:%S) $*" >> "$PLOG"; }
  say "pruner started, protecting $KEEPSTEP"
  for SAVE in $(seq $((STEP + 250)) 250 "$ITERS"); do
    while kill -0 "$1" 2>/dev/null; do
      s=$(grep -oE 'it [0-9]+/' "$LOG" 2>/dev/null | tail -1 | tr -dc '0-9'); [ -n "$s" ] || s=0
      [ "$s" -ge $((SAVE - 30)) ] && break
      sleep 15
    done
    kill -0 "$1" 2>/dev/null || { say "trainer gone, pruner exiting"; return; }
    for f in "$OUT"/step*.pt; do
      [ -e "$f" ] || continue
      [ "$(basename "$f")" = "$KEEPSTEP" ] && continue
      say "pruning $(basename "$f") ahead of save $SAVE"; rm -f "$f"
    done
    free=$(df -k /workspace | awk 'NR==2{print $4}')
    say "free $((free/1048576)) GiB going into save $SAVE"
    [ "$free" -ge $((30*1048576)) ] || say "WARNING: under 30 GiB before save $SAVE"
    while kill -0 "$1" 2>/dev/null; do
      grep -q "saved step $SAVE" "$LOG" 2>/dev/null && { say "save $SAVE landed"; break; }
      sleep 15
    done
  done
  say "pruner done"
}

torchrun \
  --nproc_per_node "$NPROC" --master_port 29683 \
  scripts/train_dmd.py \
    --data data/teacher \
    --teacher-ckpt checkpoints/wan21_14b --teacher-size 14B --fsdp \
    --accum "$ACCUM" \
    --guidance 5.0 \
    --iters "$ITERS" --critic-warmup 150 \
    --roll-depth 0 \
    --save-every 250 --sample-every 100 --log-every 10 \
    --out "$OUT" --resume \
  > "$LOG" 2>&1 &
TRAIN=$!
pruner "$TRAIN" &
PRUNER=$!
trap 'kill $PRUNER 2>/dev/null' EXIT

tail -f "$LOG" --pid "$TRAIN" &
wait "$TRAIN"; rc=$?
kill $PRUNER 2>/dev/null
echo "=== TRAIN EXIT $rc at $(date -u +%H:%M:%S) | free $(df -h /workspace | awk 'NR==2{print $4}')"
ls -la "$OUT"
echo "=== score with a NEW prefix so the recovered out/b64_* dirs are not overwritten:"
echo "    NEW=$OUT/latest.pt PREFIX=b64_final ./scripts/eval/eval_b4_2h.sh"
