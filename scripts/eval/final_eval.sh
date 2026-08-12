#!/bin/bash
# The final evaluation, run on IDLE GPUs after training completes.
#
# Why this must wait for training to stop: both mid-run evaluations were run
# against a training job saturating all 8 GPUs, and their streaming timings came
# out at 0.90x real time / budget NOT met, against 2.7x MET on idle hardware.
# That was contention, not the model -- sustained per-unit went 174 ms -> 533 ms,
# about the 3x you would predict. Quality metrics are frame-derived and were
# unaffected, but every timing number from those runs is junk. This re-measures
# all arms under identical idle conditions so the comparison is honest.
#
# Six arms x four worlds = 24 cells. FINDINGS section 4: single-world evaluation
# is how the frozen-scenery failure survived four versions of this project, so
# every arm is scored on all four worlds, always.
#
# The four b64 checkpoints are the trajectory READY_16GPU.md asks to be kept:
# checkpoint choice is NOT monotone -- v5 went -0.024 -> -0.256 -> -0.574 ->
# -0.115 on drift across four consecutive saves -- so the last checkpoint is not
# automatically the best one, and picking without scoring the intermediates is
# how you ship the wrong model.
set -u
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOT "final": logs/final_{base,v5,s160,s750,ctrl}_w*.log already exist from the
# 2026-08-08 sweep whose /exp output the rebuild erased, which makes those logs
# the only surviving record of it. Reusing the prefix would overwrite eight of
# them. Any new prefix must be checked against logs/ before use.
PREFIX=${PREFIX:-handoff}
NPROC=${NPROC:-$(nvidia-smi --list-gpus | wc -l)}
MS=checkpoints/t14b_b64/milestones

# Refuse to run while anything is on the GPUs -- that is the whole point.
#
# Ask the driver what is actually resident, rather than pattern-matching process
# names. `pgrep -f scripts/train_dmd.py` also matches any shell whose command
# line merely CONTAINS that string -- including the wait-loops used to watch for
# the run finishing -- which reported 5 phantom training processes after the run
# had cleanly exited. nvidia-smi cannot be fooled that way.
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$BUSY" -gt 0 ]; then
  echo "FATAL: $BUSY process(es) still resident on the GPUs. This evaluation needs"
  echo "       idle hardware or its timing numbers are meaningless (a contended run"
  echo "       measured 0.90x real time against 2.7x idle). Currently:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  exit 1
fi

# Override with ARMS_ONLY="s0750 s1500 s2250" to score a subset -- useful when
# base/v5/s3000 have already been measured on idle GPUs by another run and only
# the trajectory checkpoints are missing.
ALL_ARMS=(
  "base:checkpoints/dmd/latest.pt"
  "v5:checkpoints/p4_t14b/latest.pt"
  "s0750:$MS/step000750_noema.pt"
  "s1500:$MS/step001500_noema.pt"
  "s2250:$MS/step002250_noema.pt"
  "s3000:checkpoints/t14b_b64/latest.pt"
)
ARMS=()
for spec in "${ALL_ARMS[@]}"; do
  tag=${spec%%:*}
  if [ -z "${ARMS_ONLY:-}" ] || [[ " ${ARMS_ONLY} " == *" $tag "* ]]; then
    ARMS+=("$spec")
  fi
done
[ ${#ARMS[@]} -gt 0 ] || { echo "FATAL: ARMS_ONLY='${ARMS_ONLY:-}' matched no arms"; exit 1; }
for spec in "${ARMS[@]}"; do
  ck=${spec#*:}
  test -e "$ck" || { echo "FATAL: missing $ck"; exit 1; }
done

echo "=== final evaluation | ${#ARMS[@]} arms x 4 worlds = $(( ${#ARMS[@]} * 4 )) cells | $NPROC idle GPUs"
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo

i=0
for spec in "${ARMS[@]}"; do
  tag=${spec%%:*}; ck=${spec#*:}
  for w in 0 44 60 82; do
    g=$((i % NPROC)); i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$g python3 scripts/demo.py \
      --weights "$ck" --latent-norm 1.0 \
      --block 3 --steps 2 --window 6 --units 60 \
      --prompt-idx "$w" --world-cache "out/world_p$w.pt" \
      --out "out/${PREFIX}_${tag}_w${w}" \
      > "logs/${PREFIX}_${tag}_w${w}.log" 2>&1 &
    [ $((i % NPROC)) -eq 0 ] && wait
  done
done
wait

echo "=== EVAL DONE $(date -u +%H:%M:%S) -- $(ls -d out/${PREFIX}_*_w* 2>/dev/null | wc -l) cells"
echo "=== videos for the 1:1 pixel check (FINDINGS section 6):"
echo "    out/${PREFIX}_{base,v5,s0750,s1500,s2250,s3000}_w{0,44,60,82}/stream.mp4"
