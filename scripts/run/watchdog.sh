#!/bin/bash
# Training watchdog for the b64 resume: detect sustained bad training, stop the
# run, and freeze everything needed for a root-cause analysis.
#
# Thresholds are NOT invented. They are mu +/- 5 sigma over the 60 healthy log
# points of the original b64 run (checkpoints/t14b_b64/history.json, steps
# 160-750, same config, same data, same hardware):
#
#   loss_gen        mean 0.3272  sd 0.0150   ->  band [0.24, 0.42]
#   loss_critic     mean 0.1209  sd 0.0076   ->  band [0.07, 0.18]
#   dmd_grad_norm   mean 868.3   sd 46.1     ->  band [600, 1200]
#   gn_gen          mean 0.0356  sd 0.0075   ->  ceiling 0.09
#
# SUSTAINED means SUSTAIN consecutive log lines outside band, i.e. 60 steps or
# about 70 minutes at the measured 71 s/it. A single excursion is explicitly NOT
# a trigger: the batch-4 proof run spiked to gen 0.437 at step 220 and recovered
# with no ill effect, and stopping on that would have been wrong. gn_gen is the
# most diagnostic of the four -- it is the generator's actual update magnitude
# against --clip-grad 1.0, and it stayed flat through that spike.
#
# Hard faults trip immediately, with no sustain window: NaN/Inf, a python
# traceback, a CUDA/NCCL error, a dead rank, a hang, or disk below the level
# where the next save cannot complete.
#
# COST OF A FALSE POSITIVE is bounded and small: saves happen every 250 steps
# and resume is verified, so the worst case is replaying up to 250 steps (~5 h).
# That is why the bands are 5 sigma and the sustain window is an hour, rather
# than something twitchier.
#
# This script STOPS and DIAGNOSES. It deliberately does not restart: choosing
# what to change belongs in the RCA, not in a shell loop.
set -u
cd "$(dirname "$0")/../.."

LOG=${LOG:-logs/b64_resume.log}
OUT=${OUT:-checkpoints/t14b_b64}
ALERT=logs/watchdog_alert.log          # the file the operator/agent monitors
WLOG=logs/watchdog.log                 # verbose trace
SUSTAIN=${SUSTAIN:-6}                  # consecutive bad log lines before stopping
STALL_MIN=${STALL_MIN:-30}             # no new log line for this long => hang
MIN_FREE_GIB=${MIN_FREE_GIB:-12}
NRANKS=${NRANKS:-8}

GEN_LO=0.24;  GEN_HI=0.42
CRI_LO=0.07;  CRI_HI=0.18
GRD_LO=600;   GRD_HI=1200
GNG_HI=0.09

say  () { echo "$(date -u +%H:%M:%S) $*" >> "$WLOG"; }
warn () { echo "$(date -u +%H:%M:%S) $*" | tee -a "$ALERT" >> "$WLOG"; }

rca () {   # $1 = one-line reason
  local reason="$1"
  local dir="logs/rca_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dir"
  warn "WATCHDOG TRIP: $reason"
  warn "WATCHDOG collecting diagnostics into $dir"

  echo "$reason" > "$dir/reason.txt"
  tail -400 "$LOG"                     > "$dir/train_tail.log"      2>&1
  cp "$OUT/history.json"                 "$dir/history.json"        2>/dev/null
  nvidia-smi                           > "$dir/nvidia-smi.txt"      2>&1
  nvidia-smi -q                        > "$dir/nvidia-smi-full.txt" 2>&1
  df -h                                > "$dir/df.txt"              2>&1
  free -g                              > "$dir/free.txt"            2>&1
  ps auxf                              > "$dir/ps.txt"              2>&1
  dmesg 2>/dev/null | tail -200        > "$dir/dmesg.txt"           2>&1
  ls -la "$OUT"                        > "$dir/ckpt_dir.txt"        2>&1
  cp logs/b64_resume_pruner.log          "$dir/" 2>/dev/null
  cp logs/b64_milestones.log             "$dir/" 2>/dev/null
  grep -nE "Traceback|Error|error|CUDA|NCCL|OOM|out of memory|Killed|assert" "$LOG" \
       | tail -80                      > "$dir/error_lines.txt"     2>&1

  # Stop the run only AFTER the snapshot, so the diagnostics describe the
  # failing state rather than the teardown.
  local tr
  tr=$(pgrep -f "torchrun .*train_dmd.py" | head -1)
  if [ -n "$tr" ]; then
    warn "WATCHDOG stopping torchrun pid $tr (SIGTERM, elastic will drain workers)"
    kill -TERM "$tr" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$tr" 2>/dev/null || break; sleep 2; done
    kill -0 "$tr" 2>/dev/null && { warn "WATCHDOG escalating to SIGKILL"; kill -9 "$tr" 2>/dev/null; }
  else
    warn "WATCHDOG: no torchrun found; run had already exited"
  fi
  pkill -f "archive_milestones.sh" 2>/dev/null

  echo "$dir" > logs/watchdog_last_rca.txt
  warn "WATCHDOG STOPPED. RCA bundle: $dir"
  warn "WATCHDOG last good checkpoint: $(ls -1t "$OUT"/step*.pt 2>/dev/null | head -1) | latest.pt step $(python3 -c "import torch,sys;print(torch.load('$OUT/latest.pt',map_location='cpu',mmap=True,weights_only=False)['step'])" 2>/dev/null || echo '?')"
  exit 1
}

say "watchdog armed: gen[$GEN_LO,$GEN_HI] critic[$CRI_LO,$CRI_HI] grad[$GRD_LO,$GRD_HI] gn_g<=$GNG_HI, sustain=$SUSTAIN lines, stall=${STALL_MIN}m"
bad=0; lastline=""; laststamp=$(date +%s)

while true; do
  sleep 60

  # --- normal completion, checked FIRST ---
  # This must precede every rank-count test. A finished run tears its ranks down
  # over several seconds, so the count passes transiently through 1..N-1; when
  # the "done in" guard sat inside the n==0 branch only, that transient fell
  # through to the "a rank died" test and tripped the watchdog on a run that had
  # completed perfectly. Observed 2026-08-10 17:08, one minute after step 3000.
  if grep -q "done in" "$LOG" 2>/dev/null; then
    say "training finished normally ('done in' present); watchdog exiting"
    exit 0
  fi

  # --- hard fault: ranks missing ---
  # Count ONLY real python processes. `pgrep -f scripts/train_dmd.py` matches any
  # process whose command line merely contains that string, which includes every
  # monitoring shell running `until pgrep -f 'scripts/train_dmd.py'` and the
  # watchdog's own helpers. During the 2026-08-10 run that inflated the count for
  # 41 h: a genuinely dead rank could have been masked by a wrapper holding the
  # total at or above NRANKS, leaving this check inert. Restricting to -C python3
  # counts processes whose EXECUTABLE is python, so shells cannot contribute.
  ranks () { ps -C python3 -o args= 2>/dev/null | grep -c "scripts/train_dmd\.py" || true; }
  n=$(ranks)
  if [ "$n" -eq 0 ]; then
    rca "all training ranks gone and no 'done in' in the log -- the run died"
  fi
  if [ "$n" -lt "$NRANKS" ]; then
    # Re-check after a grace period: a transient dip is not a dead rank.
    sleep 20
    grep -q "done in" "$LOG" 2>/dev/null && { say "run completed during grace period; exiting"; exit 0; }
    n2=$(ranks)
    [ "$n2" -lt "$NRANKS" ] && rca "only $n2 of $NRANKS rank processes alive across a 20s grace period -- a rank died"
    say "rank count dipped to $n and recovered to $n2; not a fault"
  fi

  # --- hard fault: errors in the log ---
  if grep -qE "Traceback|CUDA error|NCCL|out of memory|CUDA out of memory|No space left" "$LOG" 2>/dev/null; then
    rca "fatal error signature present in $LOG"
  fi

  # --- hard fault: disk ---
  freeg=$(df -k /workspace | awk 'NR==2{print int($4/1048576)}')
  if [ "$freeg" -lt "$MIN_FREE_GIB" ]; then
    rca "only ${freeg} GiB free on /workspace -- below the ${MIN_FREE_GIB} GiB needed to complete a save"
  fi

  # --- hard fault: hang ---
  cur=$(grep -E "it [0-9]+/" "$LOG" 2>/dev/null | tail -1)
  now=$(date +%s)
  if [ "$cur" != "$lastline" ] && [ -n "$cur" ]; then
    lastline="$cur"; laststamp=$now
  elif [ $(( (now - laststamp) / 60 )) -ge "$STALL_MIN" ]; then
    rca "no new training log line for $(( (now - laststamp) / 60 )) minutes -- run appears hung"
  fi

  # --- soft: sustained out-of-band metrics ---
  [ -n "$cur" ] || continue
  read -r step gen cri grd gng <<<"$(echo "$cur" | sed -nE 's/.*it ([0-9]+)\/[0-9]+ gen (\S+) critic (\S+) \|grad\| (\S+) norm \S+ gn_g (\S+) .*/\1 \2 \3 \4 \5/p')"
  [ -n "${gng:-}" ] || continue

  # During --critic-warmup the generator does not update and gen/|grad|/gn_g
  # print as nan BY DESIGN. This resume starts at step 751 so it never sees
  # that, but the guard matters if this watchdog is reused on a fresh run.
  if [ "${step:-0}" -le "${WARMUP:-160}" ]; then
    say "step $step still in warmup (<= ${WARMUP:-160}), metric checks skipped"
    continue
  fi

  if echo "$gen$cri$grd$gng" | grep -qiE "nan|inf"; then
    rca "NaN/Inf in training metrics at step $step: gen=$gen critic=$cri grad=$grd gn_g=$gng"
  fi

  verdict=$(python3 - "$gen" "$cri" "$grd" "$gng" <<'PY'
import sys
gen,cri,grd,gng = (float(x) for x in sys.argv[1:5])
b=[]
if not (0.24 <= gen <= 0.42): b.append(f"loss_gen={gen}")
if not (0.07 <= cri <= 0.18): b.append(f"loss_critic={cri}")
if not (600  <= grd <= 1200): b.append(f"grad_norm={grd}")
if gng > 0.09:                b.append(f"gn_gen={gng}")
print("|".join(b))
PY
)
  if [ -n "$verdict" ]; then
    bad=$((bad+1))
    warn "out of band ($bad/$SUSTAIN) at step $step: $verdict"
    [ "$bad" -ge "$SUSTAIN" ] && rca "sustained out-of-band training for $SUSTAIN consecutive log points (${SUSTAIN}0 steps); latest step $step: $verdict"
  else
    [ "$bad" -gt 0 ] && say "back in band at step $step, resetting counter (was $bad)"
    bad=0
  fi
done
