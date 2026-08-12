"""Objective quality statistics for a streamed video.

There is no reference video to compare against -- the stream is a continuation
that never existed -- so these are no-reference statistics chosen to catch the
three failure modes actually observed in this project (PROGRESS.md §8):

  collapse   the model loses signal and output greys out.
             -> latent std ratio vs the world, and pixel contrast.
  blur       output stays plausible but soft, losing high frequencies.
             -> Laplacian variance (standard sharpness proxy), relative to the
                world frames produced by the same VAE at the same resolution, so
                the number is a ratio against a known-good reference.
  freeze/flicker  motion either stops or becomes incoherent.
             -> mean |frame_t - frame_{t-1}|; a healthy talking-head sits well
                above 0 and well below the world's own inter-frame delta * ~2.

Every statistic is reported for the WORLD segment and the GENERATED segment
separately, plus their ratio. Ratios near 1.0 mean the stream matches the quality
of the bidirectional teacher output it continues -- that is the target.

`drift` measures whether the stream wanders away from the established scene:
cosine similarity in latent space between each generated frame and the mean world
latent. A monotone decline is drift; a flat line is a stable identity.
"""
import numpy as np


def _lap_var(gray):
    """Variance of the 3x3 Laplacian -- higher is sharper. gray: [H, W] float."""
    lap = (-4.0 * gray[1:-1, 1:-1]
           + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def _blockiness(gray, period):
    """Ratio of gradient energy ON a `period`-aligned grid to gradient energy off
    it. ~1.0 means no grid structure; >1.2 is visible blocking.

    This exists because Laplacian sharpness is gameable: the highest-sharpness arm
    in out_sweep/round2 scored 0.38 while decoding to a blocky checkerboard, and a
    periodic artifact grid raises high-frequency energy exactly like real detail
    does. Wan's VAE has spatial stride 8 and the DiT patch is 2 latent cells, so
    artifacts land on 8- and 16-pixel grids; measuring those periods separates
    structure from detail.
    """
    dv = np.abs(np.diff(gray, axis=1))       # [H, W-1], boundary between x,x+1
    dh = np.abs(np.diff(gray, axis=0))
    out = []
    for d, axis in ((dv, 1), (dh, 0)):
        n = d.shape[axis]
        on_idx = np.arange(period - 1, n, period)
        if len(on_idx) < 2:
            continue
        mask = np.zeros(n, dtype=bool)
        mask[on_idx] = True
        on = d[:, mask] if axis == 1 else d[mask, :]
        off = d[:, ~mask] if axis == 1 else d[~mask, :]
        if off.size and off.mean() > 1e-8:
            out.append(float(on.mean() / off.mean()))
    return max(out) if out else float('nan')


def video_stats(pix, split):
    """pix: uint8 [N, H, W, 3] numpy. split: index where generated frames begin."""
    g = pix.astype(np.float32).mean(axis=3) / 255.0        # [N, H, W] luma
    sharp = np.array([_lap_var(g[i]) for i in range(g.shape[0])])
    contrast = g.reshape(g.shape[0], -1).std(axis=1)
    delta = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))   # length N-1

    def seg(a, lo, hi):
        s = a[lo:hi]
        return float(s.mean()) if len(s) else float('nan')

    n = g.shape[0]
    out = {
        'sharpness_world': seg(sharp, 0, split),
        'sharpness_gen': seg(sharp, split, n),
        'contrast_world': seg(contrast, 0, split),
        'contrast_gen': seg(contrast, split, n),
        # deltas are between frames i and i+1, so the generated segment's own
        # deltas start at `split` (the seam frame is excluded as it spans both)
        'interframe_world': seg(delta, 0, max(0, split - 1)),
        'interframe_gen': seg(delta, split, n - 1),
    }
    # blockiness on the last 8 generated frames (artifacts accumulate, so the tail
    # is where they show) vs the world's own baseline from the same VAE
    for per in (8, 16):
        wb = np.mean([_blockiness(g[i], per) for i in range(min(split, 8))])
        gb = np.mean([_blockiness(g[i], per) for i in range(max(split, n - 8), n)])
        out[f'block{per}_world'] = float(wb)
        out[f'block{per}_gen'] = float(gb)
    out['blockiness'] = float(max(out['block8_gen'] / max(out['block8_world'], 1e-6),
                                 out['block16_gen'] / max(out['block16_world'], 1e-6)))

    for key in ('sharpness', 'contrast', 'interframe'):
        w, gg = out[f'{key}_world'], out[f'{key}_gen']
        out[f'{key}_ratio'] = float(gg / w) if w else float('nan')
    # trend over the generated segment: first vs last third, to expose decay
    gs = sharp[split:]
    if len(gs) >= 6:
        third = len(gs) // 3
        out['sharpness_first_third'] = float(gs[:third].mean())
        out['sharpness_last_third'] = float(gs[-third:].mean())
        out['sharpness_decay'] = float(gs[-third:].mean() / gs[:third].mean())
    return out


def channel_drift(world_lat, gen_lat, tail_frac=0.34):
    """Per-channel first/second-moment departure from the world, in latent space.

    Every no-reference statistic in this file, and the reference-based Frechet
    one in scripts/fwd_score.py, failed to rank the colour/saturation artifact
    (FINDINGS.md §5a: the Frechet metric put the most saturated run first and
    the only clean run fifth). They fail for the same reason -- each pools away
    the axis the artifact actually lives on.

    That axis is not hidden. `FewStepStreamer._renorm`, the inference patch that
    hides the artifact, works by matching the generated block's PER-CHANNEL mean
    and std to the world's. So the artifact is, by construction, a per-channel
    moment departure, and measuring it needs no network at all:

        mu_shift[c] = |mean(gen[c]) - mean(world[c])| / std(world[c])
        sd_ratio[c] =  std(gen[c])  / std(world[c])

    Reported as the WORST channel, not the mean: a cast in one or two of the 16
    latent channels is exactly what a magenta face is, and averaging over
    channels would dilute it back into invisibility. Measured over the tail of
    the rollout, where drift has accumulated.

    Note this is only meaningful with the latent-norm patch OFF; with it on it
    is being directly optimised, and reads ~0 by construction.
    """
    w = world_lat.float()
    g = gen_lat.float()
    n_tail = max(1, int(g.shape[1] * tail_frac))
    t = g[:, -n_tail:]
    wm = w.mean(dim=(1, 2, 3))
    ws = w.std(dim=(1, 2, 3)).clamp_min(1e-6)
    mu_shift = ((t.mean(dim=(1, 2, 3)) - wm).abs() / ws)
    sd_ratio = (t.std(dim=(1, 2, 3)) / ws)
    return {
        'chan_mu_shift_worst': float(mu_shift.max()),
        'chan_mu_shift_mean': float(mu_shift.mean()),
        'chan_sd_ratio_worst': float(sd_ratio.max()),
        'chan_sd_ratio_min': float(sd_ratio.min()),
        'chan_worst_index': int(mu_shift.argmax()),
    }


def latent_stats(world_lat, gen_lat):
    """world_lat/gen_lat: torch [C, F, H, W] float. Adds drift + collapse terms."""
    w = world_lat.float()
    g = gen_lat.float()
    ref = w.mean(dim=1, keepdim=True).flatten()
    ref = ref / (ref.norm() + 1e-8)
    cos = []
    for i in range(g.shape[1]):
        f = g[:, i].flatten()
        cos.append(float((f / (f.norm() + 1e-8) @ ref)))
    per_frame_std = [float(g[:, i].std()) for i in range(g.shape[1])]
    third = max(1, len(cos) // 3)
    return {
        'std_world': float(w.std()),
        'std_gen': float(g.std()),
        'std_ratio': float(g.std() / w.std()),
        'std_first_frame': per_frame_std[0],
        'std_last_frame': per_frame_std[-1],
        'std_frame_decay': float(per_frame_std[-1] / per_frame_std[0]),
        'world_cos_first_third': float(np.mean(cos[:third])),
        'world_cos_last_third': float(np.mean(cos[-third:])),
        'world_cos_drift': float(np.mean(cos[-third:]) - np.mean(cos[:third])),
    }


def summary_line(vs, ls):
    return (f"sharp {vs['sharpness_ratio']:.2f}x  "
            f"contrast {vs['contrast_ratio']:.2f}x  "
            f"motion {vs['interframe_ratio']:.2f}x  "
            f"blockiness {vs['blockiness']:.2f}x  "
            f"latent-std {ls['std_ratio']:.2f}x  "
            f"sharp-decay {vs.get('sharpness_decay', float('nan')):.2f}  "
            f"world-cos-drift {ls['world_cos_drift']:+.3f}")
