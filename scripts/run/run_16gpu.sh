#!/bin/bash
# THE 16xH200 RUN: the 14B-teacher SF-DMD distillation, done properly.
#
# What this is. FINDINGS §7 item 0 says the best configuration found in v5 is
# distillation from a Wan2.1-14B teacher into the same 1.3B student -- 3 wins /
# 1 tie / 0 losses across four worlds at IDENTICAL inference cost -- but that
# what shipped was a 400-iteration finetune at effective batch 4, not the
# published recipe. Causal-rCM (arXiv:2606.25473) Table 3 runs the SF-DMD stage
# at global batch 64. Batch 4 vs 64 is the single largest deviation from a
# recipe known to work, and it is a hardware problem, not a code problem.
#
# 16 GPUs x --accum 4 = effective batch 64. That is the whole point of the run.
#
# Measured on 2xH200 (this box), 14B teacher + 1.3B student, --fsdp:
#   7.7 s per micro-step per rank, 43 GiB peak at accum 1, 48 GiB at accum 4.
#   accum 4 measured 30.8 s/it end to end.
# Per-rank work does not change with world size, so 16 ranks x accum 4 is the
# same 30.8 s/it plus a larger all-reduce. 3000 iterations -> ~26 h.
#
# Shorter alternatives (same script, one flag):
#   --accum 2  -> effective batch 32, ~15.4 s/it, ~13 h for 3000 iters
#   --accum 1  -> effective batch 16, ~7.7 s/it,  ~6.5 h  (still 4x v5's batch)
set -u
cd "$(dirname "$0")/../.."
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
mkdir -p logs out

NPROC=${NPROC:-$(nvidia-smi --list-gpus | wc -l)}
ACCUM=${ACCUM:-4}
ITERS=${ITERS:-3000}
OUT=${OUT:-checkpoints/t14b_b64}

# Multi-node: a 16xH200 allocation is usually 2 nodes x 8. Set these and run the
# same command on both nodes. Single node with 16 visible GPUs needs none of it.
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29680}

# RESUME=1 continues an interrupted run from $OUT -- weights, LoRA critic, BOTH
# AdamW states, the EMA and the step counter. Must be the same world size that
# wrote the optimizer shards; train_dmd.py refuses loudly otherwise.
RESUME_ARGS=""
if [ "${RESUME:-0}" != "0" ]; then
  RESUME_ARGS="--resume"
  echo "=== RESUMING from $OUT"
fi

# Preflight. Everything below is referenced by the torchrun line or the eval that
# follows it, and a missing one costs a 4-minute startup before it surfaces as a
# stack trace. Fail here instead, with the name of the thing that is missing.
for f in checkpoints/dmd/latest.pt checkpoints/dmd/critic_latest.pt \
         checkpoints/p4_t14b/latest.pt data/prompts.pt \
         out/world_p0.pt out/world_p44.pt out/world_p60.pt out/world_p82.pt; do
  test -e "$f" || { echo "FATAL: missing $f -- it ships in the zip; did the extract complete?"; exit 1; }
done
for d in checkpoints/wan21_13b checkpoints/wan21_14b wan21_repo; do
  test -d "$d" || { echo "FATAL: missing $d -- run ./scripts/run/setup_16gpu.sh first"; exit 1; }
done
test -d data/teacher || { echo "FATAL: data/teacher missing -- run ./scripts/run/gen_teacher_all.sh"; exit 1; }
NCLIPS=$(ls data/teacher/*.pt 2>/dev/null | wc -l)
[ "$NCLIPS" -gt 0 ] || { echo "FATAL: data/teacher is empty"; exit 1; }
echo "=== $NCLIPS teacher clips | $NPROC ranks x $NNODES nodes x accum $ACCUM"
echo "=== effective batch $((NPROC * NNODES * ACCUM)) | $ITERS iters"

torchrun \
  --nnodes "$NNODES" --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  --nproc_per_node "$NPROC" \
  scripts/train_dmd.py \
    --init checkpoints/dmd/latest.pt \
    --init-critic checkpoints/dmd/critic_latest.pt \
    --teacher-ckpt checkpoints/wan21_14b --teacher-size 14B --fsdp \
    --accum "$ACCUM" \
    --guidance 5.0 \
    --iters "$ITERS" --critic-warmup 150 \
    --roll-depth 0 \
    --save-every 250 --sample-every 100 --log-every 10 \
    --out "$OUT" $RESUME_ARGS \
  2>&1 | tee "logs/t14b_b64_node${NODE_RANK}.log"

echo "=== TRAIN DONE $(date -u +%H:%M:%S) ==="
[ "$NODE_RANK" -eq 0 ] || exit 0
# TRAIN_ONLY=1 stops here. The evaluation below is ~40 min of single-GPU runs and
# is worth splitting off when you want the cluster back, or when re-scoring an
# intermediate checkpoint instead of the last one.
[ "${TRAIN_ONLY:-0}" = "0" ] || { echo "TRAIN_ONLY set, skipping eval"; exit 0; }

# Evaluate on ALL FOUR worlds. Single-world evaluation is how the frozen-scenery
# failure survived four versions of this project (MANIFEST, FINDINGS §4).
# Checkpoint choice is NOT monotone -- score the intermediates too, they are kept.
python scripts/eval_matrix.py \
    --ckpts "base=checkpoints/dmd/latest.pt,v5=checkpoints/p4_t14b/latest.pt,new=$OUT/latest.pt" \
    --latent-norms 0.0,1.0 --units 100 --block 3 --steps 2 --window 6 \
    --prefix b64 2>&1 | tail -20

i=0
for w in 44 60 82; do
  for spec in "base:checkpoints/dmd/latest.pt" "v5:checkpoints/p4_t14b/latest.pt" "new:$OUT/latest.pt"; do
    IFS=: read -r tag ck <<< "$spec"
    g=$((i % NPROC)); i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$g python scripts/demo.py --weights "$ck" --latent-norm 1.0 \
       --block 3 --steps 2 --window 6 --units 60 --prompt-idx "$w" \
       --world-cache "out/world_p$w.pt" --out "out/b64_w${w}_${tag}" \
       > "logs/b64_w${w}_${tag}.log" 2>&1 &
    [ $((i % NPROC)) -eq 0 ] && wait
  done
done
wait
python scripts/eval_matrix.py --table 'out/b64_w*' 2>&1 | head -20
echo "=== EVAL DONE $(date -u +%H:%M:%S) -- now LOOK AT THE PIXELS (FINDINGS §6) ==="
