#!/bin/bash
# Keep a trajectory of evaluatable checkpoints for the b64 resume, cheaply.
#
# The problem. READY_16GPU.md says checkpoint choice is not monotone -- v5 went
# -0.024 -> -0.256 -> -0.574 -> -0.115 on drift across steps 1250/1500/1750/2000
# -- so "keep the intermediates" matters. But each save is 10.6 GiB and the
# pre-save pruner has to delete the rolling step file to make room for the next
# save, so by default this run would end holding only step 750 and step 3000.
#
# The lever. A checkpoint is exactly half EMA weights:
#     model  825 tensors  5.29 GiB  float32
#     ema    825 tensors  5.29 GiB  float32
# and every evaluation in this project, recovered and new, ran with
# use_ema=false -- demo.py only reads 'model' unless you pass --use-ema. So an
# archival copy with 'ema' dropped is 5.3 GiB instead of 10.6 and scores
# identically under the eval path actually in use.
#
# This is purely additive: it copies a stripped version aside BEFORE the pruner
# deletes the full file, so it can only increase what survives. It never touches
# latest.pt or the optimizer shards, which is what --resume actually reads, so
# resumability and the EMA in the live checkpoint are unaffected.
#
# Space, with milestones at 1500 and 2250 (2 x 5.3 GiB):
#   floor during a save = 54 - 10.6 (rolling) - 19 (shard .tmp) - 10.6 = 13.8 GiB
# Adding a third milestone drops the floor to ~8.5 GiB, which is still safe but
# is about as far as this volume goes.
set -u
cd "$(dirname "$0")/../.."

OUT=${OUT:-checkpoints/t14b_b64}
MS=${MS:-"1500 2250"}
DEST="$OUT/milestones"     # deliberately NOT matching the pruner's step*.pt glob
ALOG=logs/b64_milestones.log
mkdir -p "$DEST"

say () { echo "$(date -u +%H:%M:%S) $*" >> "$ALOG"; }
say "archiver started, milestones: $MS -> $DEST"

pending=$MS
while [ -n "$pending" ]; do
  still=""
  for m in $pending; do
    src="$OUT/step$(printf '%06d' "$m").pt"
    dst="$DEST/step$(printf '%06d' "$m")_noema.pt"
    if [ -e "$dst" ]; then continue; fi
    if [ -e "$src" ]; then
      say "archiving step $m (stripping EMA)"
      if python3 - "$src" "$dst" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
b = torch.load(src, map_location='cpu', mmap=True, weights_only=False)
torch.save({'model': b['model'], 'step': b['step'], 'args': b['args'],
            'note': 'EMA stripped for archival; eval path uses model only'},
           dst + '.tmp')
import os; os.replace(dst + '.tmp', dst)
PY
      then
        say "archived step $m -> $(du -h "$dst" | cut -f1) (free $(df -h /workspace | awk 'NR==2{print $4}'))"
      else
        say "ERROR archiving step $m; leaving it to the pruner"
        rm -f "$dst.tmp"
      fi
    else
      still="$still $m"
    fi
  done
  pending=$(echo "$still" | xargs || true)
  [ -n "$pending" ] || break
  pgrep -f "scripts/train_dmd.py" >/dev/null || { say "trainer gone; archiver exiting with pending:$pending"; exit 0; }
  sleep 300
done
say "archiver done; milestones held: $(ls "$DEST" 2>/dev/null | tr '\n' ' ')"
