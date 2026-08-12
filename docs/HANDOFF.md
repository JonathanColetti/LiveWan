# HANDOFF — wanstreamer batch-64 run, completed 2026-08-10

The 3000-iteration SF-DMD distillation at effective batch 64 is finished. This
document is the deliverable: which checkpoint to use, what was tested, where the
data is, and what is still open.

Read `STATUS.md` for the operational history of this box — it records four
things the shipped documents get wrong about *this* hardware, and you will lose
time without it.

---

## 1. The model to use

```
/workspace/ws/checkpoints/t14b_b64/milestones/step002250_noema.pt
```

**Not `latest.pt`.** The final step-3000 checkpoint is measurably worse, and
taking the last checkpoint on faith is the specific mistake `READY_16GPU.md`
warns about: *"checkpoint choice is NOT monotone — v5 went −0.024 → −0.256 →
−0.574 → −0.115 on drift across steps 1250/1500/1750/2000. Keep the
intermediates."* That warning earned its place here.

Load it exactly like any other checkpoint:

```bash
python3 scripts/demo.py \
  --weights checkpoints/t14b_b64/milestones/step002250_noema.pt \
  --latent-norm 1.0 --block 3 --steps 2 --window 6 --units 60 \
  --prompt-idx 7 --out out/mydemo
```

The `_noema` suffix means the EMA copy was stripped to halve the file (10.6 →
5.3 GiB). This changes nothing for inference: `demo.py` reads only `model`
unless given `--use-ema`, and every evaluation in this project — 106 recovered
and 37 new — ran with `use_ema: false`. If you ever need the EMA weights for
step 2250 they are gone; `latest.pt` still carries its own for step 3000.

### Why 2250 and not 3000

All six arms scored on **idle** GPUs, four worlds each, 2026-08-10 17:16–17:23:

| arm | step | mean \|decay−1\| | mean \|drift\| | mean motion | min real-time | budget |
|---|---|---|---|---|---|---|
| base (`dmd`) | 2000 | 0.0775 | 0.0160 | 1.05 | 2.72× | MET |
| v5 (`p4_t14b`) | 400 | 0.1325 | 0.0555 | 2.69 | 2.75× | MET |
| b64 | 750 | 0.0750 | 0.0915 | 1.91 | 2.49× | MET |
| b64 | 1500 | 0.0725 | 0.0580 | 1.93 | 2.43× | MET |
| **b64** | **2250** | **0.0575** | **0.0178** | 2.17 | 2.77× | MET |
| b64 | 3000 | 0.1275 | 0.0240 | 2.21 | 2.76× | MET |

Step 3000 more than doubles the sharpness-decay deviation against step 2250 and
lands at v5's level. Per world, the damage is almost entirely world 60:

```
sharp_decay     w0     w44    w60    w82        |deviation from 1.0|
s2250          0.90   1.00   0.91   1.04        0.10  0.00  0.09  0.04
s3000          0.94   0.92   1.30   1.07        0.06  0.08  0.30  0.07
```

Sharpness *rising* 30% across a 30-second stream is anomalous, not a wash. World
60 is the volatile one throughout — 1.05 → 0.76 → 0.91 → 1.30 across the four
checkpoints — which is why per-world reading is mandatory (`FINDINGS §4`).

**The honest counter-argument.** Step 3000 is sharper (w44 3.22 vs 2.65) and has
more motion (w60 4.86 vs 4.09), both closer to v5's character. If the pixel check
shows the w60 decay behaviour is benign, 3000 is defensible. The numbers alone
say 2250; they are proxies, and `FINDINGS §6` records nine occasions where
proxies in this project pointed the wrong way and pixels caught every one.

**Second choice:** `checkpoints/t14b_b64/latest.pt` (step 3000, carries EMA).

---

## 2. What was tested

**Correctness gates, after training** — `logs/gates_final_2026-08-10.log`:

```
verify_blockcausal   ALL CHECKS PASSED     peak 28.2 GiB
verify_fsdp          ALL CHECKS PASSED     7 checks; control correctly raises
verify_rolling       ALL CHECKS PASSED     controls 6 and 7 correctly FAIL
```

Identical to the pre-run result, so 41.6 hours of training did not disturb the
block-causal attention, the FSDP sharding or the rolling K/V path. Note these
gates did **not** pass when this box was first set up — see `STATUS.md §5`.

**Checkpoint integrity** — all five load, 825 tensors each, **zero non-finite
values**, identical weight-norm ranges (0.018 … 115.4).

**Streaming, 24 cells on idle GPUs** — every arm × every world met the real-time
budget, 2.43–2.79× real time. The step-2250 arm is the fastest of the b64
checkpoints at 2.77× minimum.

**End-to-end smoke test on an unseen prompt** (`logs/smoke_s2250_p7.log`) — prompt
7, not one of the four evaluation worlds, world generated from scratch rather
than cached:

```
loaded model from step002250_noema.pt (step 2250, all 825 tensors matched)
sustained 181.2 ms/unit -> 66.2 pixel FPS (2.65x real time), budget MET
decoded 513 frames (20.5s) in 11.2s
sharp-decay 0.99   world-cos-drift -0.008
```

One caveat from that test: `sharp 0.25x` is lower than any of the four
evaluation worlds gave for this checkpoint (0.96 / 2.65 / 2.62 / 0.49). One
unseen prompt with a freshly generated world is not enough to call it a problem,
but it is worth a look when you do the pixel pass.

---

## 3. Data for charts

`docs/data/`, long format, no plotting applied:

| file | rows | what |
|---|---|---|
| `training_curves.csv` | 472 | every log point across 5 runs: losses, grad norms, `gn_gen`, depth |
| `step_times.csv` | 467 | **instantaneous** s/it per interval |
| `eval_matrix.csv` | 157 | every scored evaluation cell ever run on this box |
| `checkpoint_traj.csv` | 6 | the aggregated per-arm comparison above |

Regenerate with `python3 scripts/eval/export_charts.py`.

**Two traps baked into these files, do not undo them.**

*Never chart the trainer's printed `s/it`.* It is elapsed÷steps-since-start — a
cumulative average that climbs all run because the critic-warmup steps are cheap,
and it reads as a progressive slowdown that is not happening. The main b64 run
appeared to go 24 → 59 s/it; differenced from timestamps it was **flat at 66.2
s/it at step 160 and 66.3 at step 750**. `step_times.csv` and the
`instantaneous_s_per_it` column are already differenced.

*Respect `timing_valid` in `eval_matrix.csv`.* 15 of 157 cells were measured
while something else held the GPUs and their timings are meaningless — a
contended run reports 0.90× real time against 2.7× idle, purely from contention.
Quality metrics in those rows are fine; timing is not.

---

## 4. The run itself

| | |
|---|---|
| completed | `done in 41.57 h`, `TRAIN EXIT 0` |
| steps | 750 → 3000, resumed rather than restarted |
| config | 8 ranks × accum 8 = effective batch 64, 14B teacher under FSDP |
| rate | **66.5 s/it**, flat over 225 log points |
| interventions | none |

Health over the full resume against the 60-point pre-crash reference:

| metric | pre-crash | resume (n=225) |
|---|---|---|
| `loss_gen` | 0.3272 ± 0.0150 | 0.3137 ± 0.0160 |
| `loss_critic` | 0.1209 ± 0.0076 | 0.1258 ± 0.0076 |
| `dmd_grad_norm` | 868.3 ± 46.1 | 845.5 ± 49.8 |
| `gn_gen` | 0.0356 ± 0.0075 | 0.0393 ± 0.0059 |

Every mean within one standard deviation of reference. That is the strongest
available evidence the resume restored real optimizer state rather than
something merely loadable — a subtly broken restore shows as a shifted mean or a
changed variance, and neither appeared.

**Losses do not decrease in this trainer, and should not.** The critic is
retrained every step, so the generator is holding position against a strengthening
opponent rather than descending a fixed landscape. Across steps 160–3000 no
metric has a trend exceeding its own noise. A steadily falling `loss_gen` would
more likely mean the critic had collapsed.

---

## 5. Still open

**Whether batch 64 is worth it — the premise of the entire run — is not
settled.** The batch-64 arm clearly beats v5 and beats the batch-4 proof, but it
has seen 144,000 samples against the proof's 3,800, so the comparison is
confounded with simply having trained longer. A clean batch-size claim needs a
compute-matched batch-4 arm, which is a separate run of comparable length.

**The pixel check has not been done.** `FINDINGS §6`: nine automatic
measurements in this project have pointed the wrong way and every one was caught
by looking at pixels at 1:1. Everything in section 1 above rests on proxies. The
videos exist this time — 25 directories, 3.9 GB:

```
out/handoff_{s0750,s1500,s2250}_w{0,44,60,82}/stream.mp4   the trajectory
out/FINAL_w{0,44,60,82}_{base,v5,new}/stream.mp4           base / v5 / step 3000
out/smoke_s2250_p7/stream.mp4                              unseen prompt
```

The comparison that decides section 1 is **`handoff_s2250_w60` against
`FINAL_w60_new`**, side by side.

**World 82 remains unexplained.** Every trained arm drifts on it while untrained
`base` sits at −0.001. Step 2250 largely resolved it (+0.042 against v5's
+0.165), but nobody knows why w82 behaves differently.

---

## 6. Housekeeping

`step003000.pt` has been deleted — all 1650 tensors and the `args` dict were
verified identical to `latest.pt` first. (The 12,960-byte size difference between
them was never content: torch names the entries inside its zip container after
the file's own basename.) That reclaimed 10.6 GiB.

**The eight `opt_rank*.pt` shards, 17.7 GiB, are deliberately kept** — the run is
to be extended past step 3000, and they are the only thing that makes that a
resume rather than a restart. They require world size 8, unchanged here.

## 7. Extending past step 3000

```bash
ITERS=4000 ./scripts/run/run_b64_resume.sh
```

It resumes from `latest.pt`, preserves it as `step003000.pt` before the first
save can overwrite it, and prunes ahead of each save at 3250 / 3500 / 3750 / 4000.

One bug was fixed for this path: `KEEPSTEP` was hardcoded to `step000750.pt`,
correct exactly once. Extending from 3000 would have copied step-3000 weights
into a file named `step000750.pt` and then protected that misnamed file for the
whole run. It is now derived from the checkpoint's actual step. A guard also
rejects an `--iters` that is not beyond the current step.

**Space for the extend.** A save needs ~30 GiB transiently (10.6 for the step
file, 17.7 for the shard `.tmp` writes). With the 9 GiB handoff zip still on the
volume the trough is 20.3 GiB; move the zip off and it is 29.3 GiB. Both clear
the watchdog's 12 GiB floor, so the extend is safe either way — but move the zip
off the box once you have it.

Scripts added this session are listed in `STATUS.md §10`. The ones worth keeping
are `presave_pruner.sh` (frees space *before* a save, which is the only time it
helps), `archive_milestones.sh` (the EMA-stripping trick), and `watchdog.sh`
(bands derived from your own healthy data, not invented).
