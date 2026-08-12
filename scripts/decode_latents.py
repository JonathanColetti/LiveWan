"""VAE-decode latent .pt files to mp4 + a contact sheet, for eyeballing."""
import os, sys, glob, argparse, subprocess
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS
from wan.modules.vae import WanVAE

ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs='+')
ap.add_argument('--out', default=os.path.join(HERE, 'out/decoded'))
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--fps', type=int, default=25)
ap.add_argument('--key', default='latents')
ap.add_argument('--sheet-cols', type=int, default=8)
ap.add_argument('--sheet-frames', type=int, default=16)
ap.add_argument('--device', default='cuda')
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
dev = torch.device(args.device)
cfg = WAN_CONFIGS['t2v-1.3B']
vae = WanVAE(vae_pth=f'{args.ckpt}/{cfg.vae_checkpoint}', device=dev)


def encode_mp4(path, arr, fps):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{arr.shape[2]}x{arr.shape[1]}',
           '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
           '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(arr).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')


def sheet(path, arr, cols, n):
    idx = np.linspace(0, arr.shape[0] - 1, min(n, arr.shape[0])).astype(int)
    fr = arr[idx]
    rows = int(np.ceil(len(fr) / cols))
    h, w = fr.shape[1], fr.shape[2]
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, f in enumerate(fr):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    import imageio.v2 as imageio
    imageio.imwrite(path, canvas)


files = []
for pat in args.inputs:
    files.extend(sorted(glob.glob(pat)) if any(c in pat for c in '*?[') else [pat])

for f in files:
    d = torch.load(f, map_location='cpu')
    lat = (d[args.key] if isinstance(d, dict) else d).float().to(dev)
    with torch.amp.autocast('cuda', dtype=cfg.param_dtype, enabled=(args.device=='cuda')):
        vid = vae.decode([lat.clamp(-4, 4)])[0].cpu()
    pix = ((vid.permute(1, 2, 3, 0) + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).numpy()
    base = os.path.join(args.out, os.path.splitext(os.path.basename(f))[0])
    encode_mp4(base + '.mp4', pix, args.fps)
    sheet(base + '_sheet.png', pix, args.sheet_cols, args.sheet_frames)
    print(f'{f} -> {base}.mp4 ({pix.shape[0]} frames {pix.shape[2]}x{pix.shape[1]})')
