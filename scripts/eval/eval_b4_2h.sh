#!/bin/bash
# Evaluation for the 2 h batch-4 proof. Run this AFTER run_b4_2h.sh finishes.
#
# This is the driver that logs/match_eval.log was waiting to run when the box
# died on 2026-08-08 (that log is 0 bytes -- it never got its checkpoint).
#
# Two rules from the project documents are encoded here and should not be
# relaxed. FINDINGS section 4: single-world evaluation is how the frozen-scenery
# failure survived four versions, so all four worlds are scored. FINDINGS
# section 6: nine automatic measurements in this project have pointed the wrong
# way and every one was caught by looking at pixels, so this writes stream.mp4
# per cell and the numbers below are a triage aid, not a verdict.
#
# Output goes under /workspace/ws/out, NOT /exp. The previous sweep wrote to
# /exp and the rebuild erased every mp4 it produced.
set -u
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEW=${NEW:-checkpoints/t14b_b4_2h/latest.pt}
PREFIX=${PREFIX:-b4_2h}
NPROC=${NPROC:-$(nvidia-smi --list-gpus | wc -l)}

test -e "$NEW" || { echo "FATAL: $NEW does not exist -- has run_b4_2h.sh finished?"; exit 1; }
STEP=$(python3 -c "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('step','?'))" "$NEW" 2>/dev/null)
echo "=== scoring $NEW (step $STEP) against base=dmd and v5=p4_t14b on 4 worlds"

# The three arms are the same comparison run_16gpu.sh makes: base is the init
# this run started from, v5 is the 400-iteration batch-4 finetune it is meant
# to supersede, new is what we just trained.
i=0
for w in 0 44 60 82; do
  for spec in "base:checkpoints/dmd/latest.pt" "v5:checkpoints/p4_t14b/latest.pt" "new:$NEW"; do
    IFS=: read -r tag ck <<< "$spec"
    g=$((i % NPROC)); i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$g python3 scripts/demo.py \
      --weights "$ck" --latent-norm 1.0 \
      --block 3 --steps 2 --window 6 --units 60 \
      --prompt-idx "$w" --world-cache "out/world_p$w.pt" \
      --out "out/${PREFIX}_w${w}_${tag}" \
      > "logs/${PREFIX}_w${w}_${tag}.log" 2>&1 &
    [ $((i % NPROC)) -eq 0 ] && wait
  done
done
wait

python3 scripts/eval_matrix.py --table "out/${PREFIX}_w*" 2>&1 | head -30

cat <<EOF

=== EVAL DONE $(date -u +%H:%M:%S) ===
Numbers above are proxies. The videos are out/${PREFIX}_w{0,44,60,82}_{base,v5,new}/stream.mp4
-- compare them 1:1 before concluding anything (FINDINGS section 6).

Note: demo.py defaults to --fps 25 for playback while the model's native rate is
16 fps (START_HERE.md correction 2), so these mp4s play 1.5625x fast. That is
left as-is deliberately: every prior artifact in out/ and every recovered number
in recovered/evals_from_logs.tsv was produced the same way, and changing it here
would break the 1:1 comparison. Pass --fps 16 if you want true-rate playback.
EOF
