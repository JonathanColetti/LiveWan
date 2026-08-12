#!/bin/bash
# Pre-save disk pruner for run_b4_2h.sh. Replaces the post-save janitor, which
# freed space too late to matter.
#
# The arithmetic that forced this. Each save writes, in order:
#   latest.pt          11.35 GB, overwritten in place  -> no net change
#   step<N>.pt         11.35 GB, NEW every time        -> +11.35 GB
#   opt_rank00{0..3}   4.5 GB each, and train_dmd.py writes them as .tmp and
#                      then os.replace()s -- so the old 18 GB and the new 18 GB
#                      coexist for the duration of the write -> +18 GB transient
# Peak requirement per save is therefore ~29.4 GB free. After the step-250 save
# only 27 GB remained, so the step-500 save would have died mid-write, and a
# post-save janitor cannot help: by the time it runs the run is already dead.
#
# This prunes BEFORE each save instead, at a step where nothing is being
# written, which is deterministic rather than racing the writer. Pruning all
# step*.pt at each boundary yields ~38 GB free going in, ~9 GB margin at peak.
#
# Consequence, stated plainly: intermediate checkpoints are NOT retained. The
# run ends holding latest.pt and step000950.pt (identical content). Keeping a
# real intermediate needs ~12 GB more space than this volume has free.
set -u
cd "$(dirname "$0")/../.."

LOG=${LOG:-logs/b4_2h.log}
OUT=${OUT:-checkpoints/t14b_b4_2h}
PLOG=logs/b4_2h_pruner.log
TRAIN_PID=${TRAIN_PID:-0}
LEAD=${LEAD:-30}        # prune this many steps before the save boundary

say () { echo "$(date -u +%H:%M:%S) $*" >> "$PLOG"; }
step () { grep -oE 'it [0-9]+/' "$LOG" 2>/dev/null | tail -1 | tr -dc '0-9'; }
alive () { [ "$TRAIN_PID" = 0 ] || kill -0 "$TRAIN_PID" 2>/dev/null; }

say "pruner started (lead=$LEAD, watching $LOG, train pid $TRAIN_PID)"
for SAVE in 500 750 950; do
  say "waiting for step $((SAVE - LEAD)) to prune ahead of the $SAVE save"
  while alive; do
    s=$(step); [ -n "$s" ] || s=0
    [ "$s" -ge $((SAVE - LEAD)) ] && break
    sleep 10
  done
  alive || { say "training pid gone, pruner exiting"; exit 0; }

  before=$(df -k /workspace | awk 'NR==2{print $4}')
  for f in "$OUT"/step*.pt; do
    [ -e "$f" ] || continue
    say "pruning $(basename "$f") ahead of save $SAVE"
    rm -f "$f"
  done
  after=$(df -k /workspace | awk 'NR==2{print $4}')
  say "free ${before}K -> ${after}K ($(( (after-before)/1048576 )) GiB reclaimed)"
  need=$(( 30 * 1048576 ))
  [ "$after" -ge "$need" ] || say "WARNING: only $((after/1048576)) GiB free, save $SAVE needs ~30 GiB"

  # Do not race ahead to the next boundary until this save has landed.
  while alive; do
    grep -q "saved step $SAVE" "$LOG" 2>/dev/null && { say "save $SAVE landed"; break; }
    sleep 10
  done
  alive || { say "training pid gone, pruner exiting"; exit 0; }
done
say "pruner done"
