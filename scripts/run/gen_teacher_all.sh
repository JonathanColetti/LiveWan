#!/bin/bash
# Regenerate the teacher-latent dataset across every GPU on the box.
#
# data/teacher is NOT in the v5 deliverable (regenerable, ~1.1 GB) but BOTH
# trainers require it: train_stage1.py and train_dmd.py open it through
# wanstreamer.data.TeacherLatents and raise if the directory is empty.
#
# Sharding is by sample index and each shard writes its own .pt files, so a
# crashed shard costs only its remainder and re-running skips what exists.
#
# Measured 25.5 s/clip on one H200 (57 s on an A100). The bank is now 1000
# prompts, so at 6 seeds each that is 6000 clips:
#   2 GPUs -> ~21 h,  8 GPUs -> ~5.3 h,  16 GPUs -> ~2.7 h
# and at 2 seeds each (2000 clips): 2 GPUs -> ~7.1 h, 16 GPUs -> ~53 min.
#
# READ THIS BEFORE RUNNING IT ON A DIFFERENT MACHINE. `data/prompts.pt` is
# hardware-dependent: umt5-xxl is bit-deterministic on one box but the same
# prompts encoded on an A100 and on an H200 differ by ~2.2% relative. Clips
# generated against one encoding and trained against another are a quiet
# mismatch, so COPY `data/prompts.pt` alongside `data/teacher` and do NOT
# re-run `encode_prompts.py` on the new machine unless you also regenerate
# every clip.
set -u
cd "$(dirname "$0")/../.."
source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs data/teacher

N=${1:-$(nvidia-smi --list-gpus | wc -l)}
PER_PROMPT=${2:-6}
echo "generating teacher clips on $N GPUs, $PER_PROMPT seeds/prompt"

for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/gen_teacher.py \
      --shard "$i" --num-shards "$N" --per-prompt "$PER_PROMPT" \
      > "logs/gen_teacher_$i.log" 2>&1 &
done
wait
echo "teacher clips: $(ls data/teacher/*.pt | wc -l)"
