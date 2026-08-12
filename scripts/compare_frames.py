"""Side by side frames at 1:1 pixel scale, for judging quality by eye.

The prior work's own record is that the Laplacian sharpness statistic was fooled
three times and that only inspection at 1:1 caught it. So: no resampling here,
ever. Rows are runs, columns are the same absolute frame index in each, and if a
crop is requested it is the same pixel window in every run.
"""
import os, sys, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('videos', nargs='+', help='mp4 paths')
ap.add_argument('--labels', default='')
ap.add_argument('--frames', default='', help='comma-separated frame indices')
ap.add_argument('--n', type=int, default=6, help='evenly spaced frames if --frames unset')
ap.add_argument('--crop', default='', help='WxH+X+Y, same window in every video')
ap.add_argument('--out', required=True)
ap.add_argument('--pad', type=int, default=4)
args = ap.parse_args()

import imageio.v2 as imageio
from PIL import Image, ImageDraw

labels = args.labels.split(',') if args.labels else \
    [os.path.basename(os.path.dirname(v)) or os.path.basename(v) for v in args.videos]


def read(path):
    rd = imageio.get_reader(path)
    fr = [np.asarray(f) for f in rd]
    rd.close()
    return fr


vids = [read(v) for v in args.videos]
n_min = min(len(v) for v in vids)
if args.frames:
    idx = [int(x) for x in args.frames.split(',')]
else:
    idx = list(np.linspace(0, n_min - 1, args.n).astype(int))
idx = [i for i in idx if i < n_min]

if args.crop:
    wh, x, y = args.crop.split('+')[0], int(args.crop.split('+')[1]), int(args.crop.split('+')[2])
    cw, ch = (int(v) for v in wh.split('x'))
else:
    cw = ch = None

tiles = []
for v in vids:
    row = []
    for i in idx:
        f = v[i]
        if cw:
            f = f[y:y + ch, x:x + cw]
        row.append(f)
    tiles.append(row)

th, tw = tiles[0][0].shape[:2]
lab_w = 210
W = lab_w + len(idx) * (tw + args.pad) + args.pad
H = 26 + len(vids) * (th + args.pad) + args.pad
canvas = Image.new('RGB', (W, H), (16, 16, 18))
d = ImageDraw.Draw(canvas)
for c, i in enumerate(idx):
    d.text((lab_w + c * (tw + args.pad) + 4, 6), f'frame {i}', fill=(200, 200, 200))
for r, row in enumerate(tiles):
    y0 = 26 + r * (th + args.pad)
    d.text((6, y0 + th // 2 - 6), labels[r][:30], fill=(230, 230, 120))
    for c, f in enumerate(row):
        canvas.paste(Image.fromarray(f), (lab_w + c * (tw + args.pad), y0))
canvas.save(args.out)
print(f'wrote {args.out}  {W}x{H}  ({len(vids)} runs x {len(idx)} frames, '
      f'{"crop " + args.crop if args.crop else "full frame"}, 1:1 scale)')
