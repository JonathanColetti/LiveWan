"""Collect run metrics.json files into one markdown table."""
import os, sys, json, glob, argparse

ap = argparse.ArgumentParser()
ap.add_argument('dirs', nargs='+')
ap.add_argument('--labels', default='')
args = ap.parse_args()

labels = args.labels.split(',') if args.labels else []
rows = []
for i, d in enumerate(args.dirs):
    p = os.path.join(d, 'metrics.json')
    if not os.path.exists(p):
        print(f'(skip {d}: no metrics.json)', file=sys.stderr)
        continue
    m = json.load(open(p))
    vs, ls = m['video_stats'], m['latent_stats']
    rows.append((labels[i] if i < len(labels) else os.path.basename(d), m, vs, ls))

hdr = ['run', 'unit (lat.f)', 'steps', 'ms/unit', 'pixel FPS', 'x real-time',
       'peak GiB', 'seconds', 'sharp ratio', 'sharp decay', 'motion ratio',
       'blockiness', 'latent std', 'world-cos drift']
print('| ' + ' | '.join(hdr) + ' |')
print('|' + '|'.join(['---'] * len(hdr)) + '|')
for name, m, vs, ls in rows:
    fps = m['sustained_pixel_fps']
    print('| ' + ' | '.join([
        name,
        str(m['block_frames']),
        str(m['steps_per_unit']),
        f"{m['sustained_ms_per_unit']:.0f}",
        f"{fps:.1f}",
        f"{fps/25:.2f}",
        f"{m['peak_gib']:.1f}",
        f"{m['video_seconds']:.1f}",
        f"{vs.get('sharpness_ratio', float('nan')):.2f}",
        f"{vs.get('sharpness_decay', float('nan')):.2f}",
        f"{vs.get('interframe_ratio', float('nan')):.2f}",
        f"{vs.get('blockiness', float('nan')):.2f}",
        f"{ls.get('std_ratio', float('nan')):.2f}",
        f"{ls.get('world_cos_drift', 0):+.3f}",
    ]) + ' |')
