"""Run streaming configurations in parallel across all GPUs and tabulate them.

Every arm streams from the SAME cached world (--world-cache), so differences in
the table are attributable to the streaming settings and not to a different base
video. The world is generated once by a throwaway prep run before the fan-out.

  python tools/sweep.py round1              # named group defined below
  python tools/sweep.py round1 --units 16   # shorter arms while iterating
  python tools/sweep.py --list

Results land in out_sweep/<group>/<arm>/ and a summary table is printed and
written to out_sweep/<group>/summary.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

# Each arm is a dict of run_streaming.py flags (without leading --).
GROUPS = {
    # Does the corrected weights finding hold end-to-end, and does CFG help?
    'round1': {
        'orig_cfg6':        dict(weights='orig', steps=8, guidance=6.0, t_max=1.0),
        'orig_nocfg':       dict(weights='orig', steps=8, guidance=0.0, t_max=1.0),
        'orig_cfg6_warm':   dict(weights='orig', steps=8, guidance=6.0, t_max=0.85),
        'ft_cfg6':          dict(weights='ft', steps=8, guidance=6.0, t_max=0.7,
                                 world_anchor=0.35),
    },
    # round1 showed the failure is a SCALE divergence, not a content failure.
    # Does per-chunk moment matching (latent_norm) fix the rollout? `ft_best` is
    # the calibration reference: the previously-accepted config, re-scored with
    # the new metrics so every other number has something to be compared against.
    'round2': {
        'ft_best':          dict(weights='ft', steps=3, guidance=0.0, t_max=0.7,
                                 world_anchor=0.35),
        'ft_best_ln1':      dict(weights='ft', steps=3, guidance=0.0, t_max=0.7,
                                 world_anchor=0.35, latent_norm=1.0),
        'orig_ln1_cfg0':    dict(weights='orig', steps=8, guidance=0.0, latent_norm=1.0),
        'orig_ln1_cfg3':    dict(weights='orig', steps=8, guidance=3.0, latent_norm=1.0),
        'orig_ln1_cfg6':    dict(weights='orig', steps=8, guidance=6.0, latent_norm=1.0),
        'orig_ln05_cfg3':   dict(weights='orig', steps=8, guidance=3.0, latent_norm=0.5),
        'orig_ln1_cfg3_warm': dict(weights='orig', steps=8, guidance=3.0,
                                   t_max=0.9, latent_norm=1.0),
        'orig_ln1_cfg3_wa02': dict(weights='orig', steps=8, guidance=3.0,
                                   t_max=0.9, world_anchor=0.2, latent_norm=1.0),
    },
    # Chunk size = interaction granularity. Bigger chunks give the model more
    # bidirectional context per step but coarsen the streaming unit.
    'chunks': {
        'chunk1':           dict(weights='orig', steps=8, guidance=6.0, chunk_frames=1),
        'chunk2':           dict(weights='orig', steps=8, guidance=6.0, chunk_frames=2),
        'chunk4':           dict(weights='orig', steps=8, guidance=6.0, chunk_frames=4),
        'chunk8':           dict(weights='orig', steps=8, guidance=6.0, chunk_frames=8),
    },
    # final.pt was fine-tuned on 480p HDTF latents (train_streaming.py
    # --data-dir /workspace/hdtf_latents_480p), so the paper's 640x368 target is
    # off-distribution for the checkpoint. Does running at a native Wan size, or
    # giving the stream a longer world to anchor on, recover quality?
    'round3': {
        'r640_368':   dict(size='640*368'),
        'r832_480':   dict(size='832*480'),
        'r480_832':   dict(size='480*832'),
        'r640_world33': dict(size='640*368', world_frames=33),
    },
    # Refine the knobs at the best-looking resolution from round3 (480x832,
    # judged at 1:1 pixel scale -- diag/res_compare.py -- because the sharpness
    # ratio is not comparable across aspect ratios).
    'round4': {
        'ref':         dict(steps=3, t_max=0.7, world_anchor=0.35),
        'steps6':      dict(steps=6, t_max=0.7, world_anchor=0.35),
        'steps6_t085': dict(steps=6, t_max=0.85, world_anchor=0.35),
        'tmax050':     dict(steps=3, t_max=0.50, world_anchor=0.35),
        'wa015':       dict(steps=3, t_max=0.7, world_anchor=0.15),
        'wa050':       dict(steps=3, t_max=0.7, world_anchor=0.50),
        'cfg15':       dict(steps=3, t_max=0.7, world_anchor=0.35, guidance=1.5),
        'win12':       dict(steps=3, t_max=0.7, world_anchor=0.35, window_frames=12),
    },
    # t_max=0.5 topped round4 on sharpness, but t_max trades motion for detail by
    # construction (the chunk starts closer to the previous frame). Find where the
    # trade-off turns, and whether extra steps buy anything on top.
    'round5': {
        't040':        dict(steps=3, t_max=0.40),
        't050':        dict(steps=3, t_max=0.50),
        't060':        dict(steps=3, t_max=0.60),
        't050_steps6': dict(steps=6, t_max=0.50),
        't050_steps10': dict(steps=10, t_max=0.50),
        't050_win12':  dict(steps=3, t_max=0.50, window_frames=12),
        't050_chunk2': dict(steps=3, t_max=0.50, chunk_frames=2),
        't050_wa015':  dict(steps=3, t_max=0.50, world_anchor=0.15),
    },
}

# Applied to every arm in a group unless the arm overrides the key. Keeps the
# sweep tables honest: only the varied knob differs.
GROUP_DEFAULTS = {
    'round3': dict(weights='ft', steps=3, t_max=0.7, world_anchor=0.35,
                   latent_norm=1.0),
    'round4': dict(weights='ft', latent_norm=1.0, size='480*832', world_frames=33),
    'round5': dict(weights='ft', latent_norm=1.0, size='480*832', world_frames=33,
                   world_anchor=0.35),
}

ap = argparse.ArgumentParser()
ap.add_argument('group', nargs='?')
ap.add_argument('--list', action='store_true')
ap.add_argument('--units', type=int, default=24)
ap.add_argument('--size', default='640*368')
ap.add_argument('--world-frames', type=int, default=17)
ap.add_argument('--base-steps', type=int, default=30)
ap.add_argument('--gpus', default='0,1,2,3')
args = ap.parse_args()

if args.list or not args.group:
    for g, arms in GROUPS.items():
        print(f'{g}:')
        for a, kw in arms.items():
            print(f'    {a:18} {kw}')
    sys.exit(0)

defaults = dict(size=args.size, world_frames=args.world_frames)
defaults.update(GROUP_DEFAULTS.get(args.group, {}))
arms = {name: {**defaults, **kw} for name, kw in GROUPS[args.group].items()}
gpus = [g.strip() for g in args.gpus.split(',')]
base = os.path.join('out_sweep', args.group)
os.makedirs(os.path.join(ROOT, base), exist_ok=True)


def world_path(kw):
    """One cached world per (size, world_frames) so resolution arms are each
    compared against a world generated at their own resolution."""
    return os.path.join(base,
                        f'world_{kw["size"].replace("*", "x")}_'
                        f'{kw["world_frames"]}.pt')


def cmd_for(kw, out):
    c = [PY, 'run_streaming.py', '--base-steps', str(args.base_steps),
         '--world-cache', world_path(kw), '--units', str(args.units),
         '--out', out]
    for k, v in kw.items():
        c += [f'--{k.replace("_", "-")}', str(v)]
    return c


# ---- prep: build each distinct world once (a 1-chunk run is the cheapest way) --
for wkw in {(kw['size'], kw['world_frames']) for kw in arms.values()}:
    kw = dict(size=wkw[0], world_frames=wkw[1])
    wp = world_path(kw)
    if os.path.exists(os.path.join(ROOT, wp)):
        continue
    print(f'=== prep: generating world -> {wp} ===')
    prep = [PY, 'run_streaming.py', '--base-steps', str(args.base_steps),
            '--world-cache', wp, '--size', kw['size'],
            '--world-frames', str(kw['world_frames']), '--units', '1',
            '--steps', '1', '--weights', 'ft',
            '--out', os.path.join(base, '_prep'), '--no-sheet']
    r = subprocess.run(prep, cwd=ROOT,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpus[0]},
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        sys.exit('prep run failed')
    print('    world cached')

# ---- fan out, at most len(gpus) at a time -----------------------------------
procs, queue, running = [], list(arms.items()), {}
t_start = time.time()
logs = {}
while queue or running:
    while queue and len(running) < len(gpus):
        arm, kw = queue.pop(0)
        gpu = [g for g in gpus if g not in {v[1] for v in running.values()}][0]
        out = os.path.join(base, arm)
        logp = os.path.join(ROOT, base, f'{arm}.log')
        lf = open(logp, 'w')
        p = subprocess.Popen(cmd_for(kw, out), cwd=ROOT, stdout=lf,
                             stderr=subprocess.STDOUT,
                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
        running[p.pid] = (arm, gpu, p, lf)
        logs[arm] = logp
        print(f'  [gpu {gpu}] launched {arm}: {dict(kw)}')
    time.sleep(3)
    for pid in list(running):
        arm, gpu, p, lf = running[pid]
        if p.poll() is not None:
            lf.close()
            status = 'ok' if p.returncode == 0 else f'FAILED({p.returncode})'
            print(f'  [gpu {gpu}] {arm}: {status}  ({time.time()-t_start:.0f}s elapsed)')
            if p.returncode != 0:
                print(f'      see {logs[arm]}')
            del running[pid]

# ---- tabulate ----------------------------------------------------------------
rows = []
for arm in arms:
    mp = os.path.join(ROOT, base, arm, 'metrics.json')
    if not os.path.exists(mp):
        print(f'  {arm}: no metrics.json (run failed)')
        continue
    with open(mp) as f:
        m = json.load(f)
    vs, ls = m['video_stats'], m['latent_stats']
    rows.append({
        'arm': arm, 'sharp_ratio': vs['sharpness_ratio'],
        'contrast_ratio': vs['contrast_ratio'],
        'motion_ratio': vs['interframe_ratio'],
        'sharp_decay': vs.get('sharpness_decay', float('nan')),
        'blockiness': vs['blockiness'],
        'std_ratio': ls['std_ratio'], 'cos_drift': ls['world_cos_drift'],
        'ms_per_chunk': m['sustained_mean_ms'],
        'fps': m['sustained_pixel_fps'], 'rt': m['realtime_met'],
        'peak_gib': m['peak_gib'],
    })

hdr = (f'{"arm":<18}{"sharp":>7}{"contr":>7}{"block":>7}{"motion":>7}{"decay":>7}'
       f'{"lat-std":>8}{"drift":>8}{"ms/chunk":>10}{"px-fps":>8}{"RT":>4}')
print(f'\n=== {args.group}: {args.units} chunks, {args.size}, shared world ===')
print('    ratios are generated-vs-world; 1.00 = matches the teacher output')
print(hdr)
print('-' * len(hdr))
for r in sorted(rows, key=lambda r: -r['sharp_ratio']):
    print(f'{r["arm"]:<18}{r["sharp_ratio"]:>7.2f}{r["contrast_ratio"]:>7.2f}'
          f'{r["blockiness"]:>7.2f}{r["motion_ratio"]:>7.2f}{r["sharp_decay"]:>7.2f}'
          f'{r["std_ratio"]:>8.2f}{r["cos_drift"]:>+8.3f}'
          f'{r["ms_per_chunk"]:>10.1f}{r["fps"]:>8.1f}'
          f'{"yes" if r["rt"] else "no":>4}')
with open(os.path.join(ROOT, base, 'summary.json'), 'w') as f:
    json.dump({'group': args.group, 'units': args.units, 'size': args.size,
               'arms': arms, 'rows': rows}, f, indent=2)
print(f'\nwrote {base}/summary.json  ({time.time()-t_start:.0f}s total)')
