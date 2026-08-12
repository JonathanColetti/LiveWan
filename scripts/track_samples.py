"""Track DMD progress on the rollouts the trainer already dumped.

Each dump is a self-forcing rollout: teacher world frames followed by the
student's own continuation. Comparing the generated region against the world
region *within the same clip* controls for prompt and content, so the ratio is
comparable across iterations even though every dump uses a different prompt.

Runs on CPU by default so it cannot OOM the training job it is watching.
"""
import os, sys, glob, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS
from wan.modules.vae import WanVAE
from wanstreamer.metrics import video_stats

ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=os.path.join(HERE, 'checkpoints/dmd/samples'))
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--device', default='cpu')
ap.add_argument('--every', type=int, default=1, help='use every Nth dump')
ap.add_argument('--out', default=os.path.join(HERE, 'out/sample_track.json'))
args = ap.parse_args()

dev = torch.device(args.device)
cfg = WAN_CONFIGS['t2v-1.3B']
vae = WanVAE(vae_pth=f'{args.ckpt}/{cfg.vae_checkpoint}', device=dev)

files = sorted(glob.glob(os.path.join(args.dir, '*.pt')))[::args.every]
rows = []
for f in files:
    d = torch.load(f, map_location='cpu')
    lat = d['latents'].float().to(dev)
    wf = int(d['world_frames'])
    with torch.amp.autocast('cuda', dtype=cfg.param_dtype,
                            enabled=(args.device == 'cuda')):
        vid = vae.decode([lat.clamp(-4, 4)])[0].cpu()
    pix = ((vid.permute(1, 2, 3, 0) + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).numpy()
    world_px = (wf - 1) * 4 + 1
    vs = video_stats(pix, world_px)
    row = {'step': int(d['step']), 'world_frames': wf,
           'sharpness_ratio': vs['sharpness_ratio'],
           'contrast_ratio': vs['contrast_ratio'],
           'motion_ratio': vs['interframe_ratio'],
           'blockiness': vs['blockiness'],
           'prompt': d.get('prompt', '')[:48]}
    rows.append(row)
    print(f'it {row["step"]:6d}  world {wf:2d}f  sharp {row["sharpness_ratio"]:5.2f}  '
          f'contrast {row["contrast_ratio"]:5.2f}  motion {row["motion_ratio"]:5.2f}  '
          f'block {row["blockiness"]:4.2f}  | {row["prompt"]}', flush=True)

if rows:
    print(f'\nsharpness ratio: first {rows[0]["sharpness_ratio"]:.2f} -> '
          f'last {rows[-1]["sharpness_ratio"]:.2f}  '
          f'(mean of last 3: {np.mean([r["sharpness_ratio"] for r in rows[-3:]]):.2f})')
with open(args.out, 'w') as fh:
    json.dump(rows, fh, indent=1)
print(f'wrote {args.out}')
