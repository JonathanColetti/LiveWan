"""Run a matrix of streaming evaluations across the GPUs, then tabulate them.

Every comparison in this project has to be like-for-like or it is worthless:
same world (SCALING_PROMPT §7), same seed, same operating point, and the prior
checkpoint present as the before/after control. Doing that by hand is how you
end up comparing a 4-step window-12 run against a 2-step window-6 run and
concluding something about the weights. This expands one grid, runs it, and
prints the results in one table so the confound is visible if it exists.

It deliberately does NOT pick a winner. The tabulated statistics are the
no-reference ones that have misled five times; use them to shortlist, then rank
with scripts/fwd_score.py (reference-based) and look at the shortlist at 1:1
with scripts/compare_frames.py. The footer prints those two commands, populated
with the runs it just produced.

  python scripts/eval_matrix.py \
      --ckpts base=checkpoints/dmd/latest.pt,g1.5=checkpoints/dmd_g1.5/latest.pt \
      --latent-norms 0.0,1.0 --units 100 --prefix sweep
  python scripts/eval_matrix.py --table 'out/sweep_*'
"""
import os, sys, glob, json, time, argparse, subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--ckpts', default='',
                help='comma-separated name=path pairs')
ap.add_argument('--latent-norms', default='0.0')
ap.add_argument('--ema', default='0', help='comma-separated 0/1')
ap.add_argument('--units', type=int, default=100)
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--steps', type=int, default=2)
ap.add_argument('--window', type=int, default=6)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--world-cache', default=os.path.join(HERE, 'out/world_p0.pt'))
ap.add_argument('--prompt-idx', type=int, default=0)
ap.add_argument('--gpus', default='0,1,2,3')
ap.add_argument('--prefix', default='m')
ap.add_argument('--table', default='', help='glob of out/ dirs; tabulate only')
args = ap.parse_args()

COLS = [
    ('run', lambda d: d['_name'], '{:<26s}', 26),
    ('ln', lambda d: d.get('latent_norm'), '{:>4}', 4),
    ('ema', lambda d: int(bool(d.get('use_ema'))), '{:>3}', 3),
    ('decay', lambda d: d['video_stats'].get('sharpness_decay'), '{:>7.3f}', 7),
    ('drift', lambda d: d['latent_stats'].get('world_cos_drift'), '{:>+8.4f}', 8),
    ('sharpR', lambda d: d['video_stats'].get('sharpness_ratio'), '{:>7.3f}', 7),
    ('contR', lambda d: d['video_stats'].get('contrast_ratio'), '{:>6.3f}', 6),
    ('motR', lambda d: d['video_stats'].get('interframe_ratio'), '{:>6.3f}', 6),
    ('block', lambda d: d['video_stats'].get('blockiness'), '{:>6.3f}', 6),
    ('stdR', lambda d: d['latent_stats'].get('std_ratio'), '{:>6.3f}', 6),
    ('ms/u', lambda d: d.get('sustained_ms_per_unit'), '{:>7.1f}', 7),
]


def tabulate(dirs):
    rows = []
    for dd in sorted(dirs):
        p = os.path.join(dd, 'metrics.json')
        if not os.path.exists(p):
            print(f'  (no metrics.json in {dd})')
            continue
        d = json.load(open(p))
        d['_name'] = os.path.basename(dd)
        d['_dir'] = dd
        rows.append(d)
    if not rows:
        return rows
    hdr = ' '.join(f'{c[0]:>{c[3]}}' if c[0] != 'run' else f'{c[0]:<{c[3]}}'
                   for c in COLS)
    print(hdr)
    print('-' * len(hdr))
    for d in rows:
        cells = []
        for name, get, fmt, w in COLS:
            try:
                cells.append(fmt.format(get(d)))
            except (TypeError, ValueError, KeyError):
                cells.append(' ' * (w - 1) + '-')
        print(' '.join(cells))
    print(f'\n{len(rows)} runs. Reminder (SCALING_PROMPT §7): sharpness reads '
          f'colour artifacts as detail --\nthese columns shortlist, they do not '
          f'rank. Finish with:')
    lat = ' '.join(os.path.join(d['_dir'], 'latents.pt') for d in rows)
    print(f'  python scripts/fwd_score.py {lat} --ref data/fwd_ref.npz')
    vids = ' '.join(os.path.join(d['_dir'], 'stream.mp4') for d in rows)
    labels = ','.join(d['_name'] for d in rows)
    print(f'  python scripts/compare_frames.py {vids} \\\n'
          f'      --labels {labels} --frames 32,240,520,800,1050 '
          f'--crop 400x230+120+69 --out out/cmp.png')
    return rows


if args.table:
    tabulate([d for d in glob.glob(args.table) if os.path.isdir(d)])
    sys.exit(0)

if not args.ckpts:
    raise SystemExit('need --ckpts name=path,... or --table GLOB')

ckpts = [kv.split('=', 1) for kv in args.ckpts.split(',')]
lns = [float(x) for x in args.latent_norms.split(',')]
emas = [int(x) for x in args.ema.split(',')]
gpus = [int(x) for x in args.gpus.split(',')]

jobs = []
for name, path in ckpts:
    for ln in lns:
        for em in emas:
            tag = f'{args.prefix}_{name}_ln{ln:g}' + ('_ema' if em else '')
            out = os.path.join(HERE, 'out', tag)
            cmd = [sys.executable, os.path.join(HERE, 'scripts/demo.py'),
                   '--weights', path, '--latent-norm', str(ln),
                   '--block', str(args.block), '--steps', str(args.steps),
                   '--window', str(args.window), '--units', str(args.units),
                   '--seed', str(args.seed), '--prompt-idx', str(args.prompt_idx),
                   '--world-cache', args.world_cache, '--out', out]
            if em:
                cmd.append('--use-ema')
            jobs.append((tag, out, cmd))

print(f'{len(jobs)} runs over {len(gpus)} GPUs '
      f'({args.units} units, block {args.block}, steps {args.steps}, '
      f'window {args.window}, world {os.path.basename(args.world_cache)})')
os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)

running, queue, done = {}, list(jobs), []
t0 = time.time()
free = list(gpus)
while queue or running:
    while queue and free:
        g = free.pop(0)
        tag, out, cmd = queue.pop(0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        lf = open(os.path.join(HERE, 'logs', f'eval_{tag}.log'), 'w')
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        running[p] = (tag, out, g, lf)
        print(f'  [gpu {g}] start {tag}', flush=True)
    time.sleep(3)
    for p in list(running):
        if p.poll() is None:
            continue
        tag, out, g, lf = running.pop(p)
        lf.close()
        free.append(g)
        ok = p.returncode == 0 and os.path.exists(os.path.join(out, 'metrics.json'))
        done.append((tag, out, ok))
        print(f'  [gpu {g}] {"done " if ok else "FAILED"} {tag} '
              f'(rc={p.returncode}, {time.time()-t0:.0f}s elapsed)', flush=True)

bad = [t for t, _, ok in done if not ok]
print(f'\n{len(done)-len(bad)}/{len(done)} succeeded in {(time.time()-t0)/60:.1f} min'
      + (f' | FAILED: {bad} (see logs/eval_<tag>.log)' if bad else ''))
print()
tabulate([o for _, o, ok in done if ok])
