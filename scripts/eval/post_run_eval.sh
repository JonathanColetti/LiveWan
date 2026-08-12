#!/bin/bash
# Everything that needs idle GPUs, to run AFTER the b64 resume finishes.
#
# Four jobs, in priority order. Each is here because it cannot be done while
# training holds the GPUs, or because a previous attempt was invalidated by
# doing it anyway.
#
#   1. Score step 3000 on all four worlds. The result the run exists for.
#   2. Score step 2250. Gives a four-point trajectory (750/1500/2250/3000)
#      against the non-monotone checkpoint problem in READY_16GPU.md.
#   3. Re-measure streaming timing. The 2026-08-09 13:26 evaluation ran
#      concurrently with training and reported 0.90x real time / budget NOT met,
#      against 2.7x MET on idle GPUs. That number is contention, not the model,
#      and it needs redoing. Same run regenerates b64_s1500_w82, whose
#      stream.mp4 was corrupted by that same contention (1 of 58 files).
#   4. --steps 2 vs --steps 4. The checkpoint's own args say the student was
#      distilled at steps_per_block=4 and demo.py defaults to --steps 4, but
#      EVERY evaluation in this project explicitly passes --steps 2. If that was
#      not deliberate, no absolute quality number here has ever reflected the
#      model as trained. This settles it on identical checkpoint, world and seed.
#
# Only the "new" arm is scored: base and v5 are fixed checkpoints whose numbers
# are already in recovered/evals_from_logs.tsv and do not change.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=checkpoints/t14b_b64
NPROC=${NPROC:-$(nvidia-smi --list-gpus | wc -l)}
WORLDS="0 44 60 82"

# Matching on the command line alone is not enough: `pgrep -f` and `ps | grep`
# both match any SHELL whose command text merely mentions the trainer, including
# this script's own invocation and unrelated tooling. Require the process to
# actually be a python interpreter. Observed self-matching and refusing to run on
# a completely idle box, 2026-08-10 17:08.
if ps -eo comm,cmd --no-headers | awk '$1 ~ /^python/ && /train_dmd/ {found=1} END{exit !found}'; then
  echo "REFUSING: training is still running. Timing measured under contention is"
  echo "meaningless and concurrent I/O already corrupted one output file."
  echo "Wait for 'done in' in logs/b64_resume.log, or set FORCE=1 to override."
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

run_cell () {   # $1 weights  $2 world  $3 steps  $4 outprefix  $5 gpu
  CUDA_VISIBLE_DEVICES=$5 python3 scripts/demo.py \
    --weights "$1" --latent-norm 1.0 \
    --block 3 --steps "$3" --window 6 --units 60 \
    --prompt-idx "$2" --world-cache "out/world_p$2.pt" \
    --out "out/$4_w$2" > "logs/$4_w$2.log" 2>&1
}

sweep () {      # $1 weights  $2 steps  $3 prefix
  echo "=== $3: $(basename "$1") at --steps $2, worlds $WORLDS"
  i=0
  for w in $WORLDS; do
    run_cell "$1" "$w" "$2" "$3" $((i % NPROC)) &
    i=$((i+1))
  done
  wait
}

STEP=$(python3 -c "import torch;print(torch.load('$OUT/latest.pt',map_location='cpu',mmap=True,weights_only=False)['step'])")
echo "=== final checkpoint is step $STEP"

sweep "$OUT/latest.pt"                              2 "final_s${STEP}"
sweep "$OUT/milestones/step002250_noema.pt"         2 "final_s2250"
sweep "$OUT/milestones/step001500_noema.pt"         2 "redo_s1500"     # also fixes the corrupt w82 cell
sweep "$OUT/latest.pt"                              4 "steps4_s${STEP}"

echo
echo "=================== TRAJECTORY (all at --steps 2) ==================="
python3 - "$STEP" <<'EOF'
import re,glob,csv,sys,os
final=sys.argv[1]
def scrape(f):
    if not os.path.exists(f): return None
    t=open(f,errors='ignore').read()
    q=re.search(r'sharp ([\d.]+)x.*?motion ([\d.]+)x.*?sharp-decay ([\-\d.]+)\s+world-cos-drift ([\-+\d.]+)',t)
    s=re.search(r'\(([\d.]+)x real time\)',t); b=re.search(r'budget [\d.]+ ms/unit => (\w+)',t)
    return (q.groups() if q else None,(s.group(1) if s else '?'),(b.group(1) if b else '?'))
rec={r['log']:r for r in csv.DictReader(open('recovered/evals_from_logs.tsv'),delimiter='\t')}
W=['0','44','60','82']
arms=[('v5 (target)',      lambda w: (rec[f'final_v5_w{w}']['sharp'],rec[f'final_v5_w{w}']['motion'],
                                      rec[f'final_v5_w{w}']['sharp_decay'],rec[f'final_v5_w{w}']['world_cos_drift'])),
      ('b64 @750',         lambda w: (rec[f'final_s750_w{w}']['sharp'],rec[f'final_s750_w{w}']['motion'],
                                      rec[f'final_s750_w{w}']['sharp_decay'],rec[f'final_s750_w{w}']['world_cos_drift'])),
      ('b64 @1500',        lambda w: (scrape(f'logs/redo_s1500_w{w}.log')  or [None])[0]),
      ('b64 @2250',        lambda w: (scrape(f'logs/final_s2250_w{w}.log') or [None])[0]),
      (f'b64 @{final}',    lambda w: (scrape(f'logs/final_s{final}_w{w}.log') or [None])[0])]
for i,lab in enumerate(['sharp','motion','sharp-decay','drift']):
    print(f"\n=== {lab} ===")
    print(f"{'':<16}"+''.join(f"{'w'+w:>9}" for w in W))
    for nm,fn in arms:
        cells=[]
        for w in W:
            v=fn(w); cells.append(f"{v[i]:>9}" if v else f"{'--':>9}")
        print(f"{nm:<16}"+''.join(cells))
print("\n=================== STEPS 2 vs STEPS 4 (step %s) ===================" % final)
print(f"{'world':<8}{'sharp 2':>9}{'sharp 4':>9}{'motion 2':>10}{'motion 4':>10}{'drift 2':>10}{'drift 4':>10}{'realtime 4':>12}")
for w in W:
    a=scrape(f'logs/final_s{final}_w{w}.log'); b=scrape(f'logs/steps4_s{final}_w{w}.log')
    if not(a and b and a[0] and b[0]): continue
    print(f"w{w:<7}{a[0][0]:>9}{b[0][0]:>9}{a[0][1]:>10}{b[0][1]:>10}{a[0][3]:>10}{b[0][3]:>10}{b[1]+'x ('+b[2]+')':>12}")
EOF
cat <<'EOF'

Numbers are proxies. FINDINGS section 6: nine automatic measurements in this
project have pointed the wrong way and every one was caught by looking at pixels.
The videos are out/<prefix>_w<world>/stream.mp4 -- compare them 1:1, and compare
steps4_* against final_* on the same world before concluding anything about the
step count.
EOF
