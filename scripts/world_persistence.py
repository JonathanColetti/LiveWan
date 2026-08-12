"""Separate three temporal failure modes that the eval-matrix columns conflate,
because phase 4 produced a case where `motR` jumped 5x and that could equally
have meant "fixed a frozen stream" or "started flickering":

  frozen    -- the stream barely evolves at all
  coherent  -- the scene changes smoothly and remains the same scene
  churn     -- the scene re-hallucinates, breaking the persistent-world contract

Two numbers are needed, and using either alone is wrong:

  tstd   absolute per-pixel temporal std, in grey levels. Detects FROZEN.
  ratio  |d(4)| / |d(80)|, scale-free. Detects CHURN (~1.0 == no memory beyond
         a block, because a 0.16 s gap already decorrelates as much as a 3.2 s one).

Development history, kept because it is the point: the first two versions of this
script ran on LATENTS and both contradicted direct frame inspection -- they scored
a visually static clip as more dynamic than a visibly moving one, because latent
distance is dominated by per-frame sampling noise rather than scene content, and
because a scale-free normalisation cannot see "frozen" by construction. Only the
pixel-domain absolute measure agreed with the frames. Per SCALING_PROMPT.md §7,
this shortlists and explains; it does not rank, and every claim it supports in
FINDINGS.md is also confirmed by 1:1 inspection.

Controls (`--controls`) rebuild known-answer clips from a real one: a frame-shuffled
clip must read CHURN and a repeated-frame clip must read frozen. Run them.
"""
import argparse
import subprocess

import numpy as np

FROZEN_TSTD = 8.0   # grey levels; below this the stream is not meaningfully evolving
CHURN_RATIO = 0.85  # at/above this, a 0.16 s gap decorrelates like a 3.2 s one


def load(path, n=600, w=80, h=46):
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-vf', f'scale={w}:{h}',
           '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8).astype(np.float32)
    return a[:len(a) // (w * h) * (w * h)].reshape(-1, h * w)


def score(f, lags=(4, 20, 80)):
    d = {L: float(np.abs(f[L:] - f[:-L]).mean()) if L < len(f) else float('nan')
         for L in lags}
    tstd = float(f.std(0).mean())
    ratio = d[lags[0]] / d[lags[-1]] if d[lags[-1]] > 1e-9 else float('nan')
    # Churn is tested FIRST: a stream with no memory beyond one block is churning
    # whatever its amplitude, and testing `frozen` first misclassifies a
    # low-amplitude shuffled clip (the control caught exactly this).
    if ratio >= CHURN_RATIO:
        v = 'CHURN'
    elif tstd < FROZEN_TSTD:
        v = 'frozen'
    else:
        v = 'coherent'
    return tstd, d, ratio, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', help='out/<run> directories (uses stream.mp4)')
    ap.add_argument('--controls', metavar='RUN',
                    help='build known-answer controls from this run and score them. '
                         'Pass a clip that actually MOVES -- shuffling a static clip '
                         'tests nothing, which is how the verdict precedence bug was found.')
    args = ap.parse_args()

    rows = []
    if args.controls:
        f = load(f'{args.controls.rstrip("/")}/stream.mp4')
        rng = np.random.default_rng(0)
        rows.append(('ctl_churn  (want CHURN)', f[rng.permutation(len(f))]))
        rows.append(('ctl_frozen (want frozen)', np.repeat(f[:1], len(f), 0)))
    for r in args.runs:
        rows.append((r.rstrip('/').split('/')[-1], load(f'{r.rstrip("/")}/stream.mp4')))

    print(f"{'run':<26}{'tstd':>7}{'|d4|':>7}{'|d20|':>7}{'|d80|':>7}{'ratio':>8}   verdict")
    print('-' * 72)
    for name, f in rows:
        tstd, d, ratio, v = score(f)
        print(f'{name:<26}{tstd:7.2f}{d[4]:7.2f}{d[20]:7.2f}{d[80]:7.2f}{ratio:8.3f}   {v}')


if __name__ == '__main__':
    main()
