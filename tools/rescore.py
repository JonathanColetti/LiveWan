"""Recompute video statistics for finished runs straight from their mp4s.

Lets a new metric be validated against arms whose failure mode is already known by
eye, without regenerating anything. Used to check that `blockiness` actually
separates the checkerboard-artifact arm from the good ones.

  python tools/rescore.py out_sweep/round2/*/
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'wan21_repo'))
from wanstreamer.metrics import video_stats                       # noqa: E402

dirs = [d for d in sys.argv[1:] if os.path.isdir(d)]
rows = []
for d in sorted(dirs):
    mp = os.path.join(d, 'metrics.json')
    vp = os.path.join(d, 'streamed.mp4')
    if not (os.path.exists(mp) and os.path.exists(vp)):
        continue
    with open(mp) as f:
        m = json.load(f)
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    pix = np.stack(frames)
    vs = video_stats(pix, m['world_pixel_frames'])
    m['video_stats'] = vs
    with open(mp, 'w') as f:
        json.dump(m, f, indent=2)
    rows.append((os.path.basename(d.rstrip('/')), vs, m['latent_stats']))

hdr = (f'{"arm":<22}{"sharp":>7}{"contr":>7}{"block":>7}{"blk8":>7}{"blk16":>7}'
       f'{"lat-std":>9}')
print(hdr)
print('-' * len(hdr))
for name, vs, ls in sorted(rows, key=lambda r: -r[1]['sharpness_ratio']):
    print(f'{name:<22}{vs["sharpness_ratio"]:>7.2f}{vs["contrast_ratio"]:>7.2f}'
          f'{vs["blockiness"]:>7.2f}{vs["block8_gen"]:>7.2f}'
          f'{vs["block16_gen"]:>7.2f}{ls["std_ratio"]:>9.2f}')
