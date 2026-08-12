"""Streaming latency in isolation: no world generation, no VAE, no decode.

Sweeps the knobs that trade interaction granularity against throughput, and
reports each against the papers' operating point (640x368, 25 FPS, a 160 ms
streaming unit -- one latent frame at VAE temporal stride 4). A unit of `block`
latent frames must be produced in block*160 ms to sustain real time.

Steady state only: the first units are dropped, and with a bounded event window
the cache length is constant afterwards, so the reported number is the one that
holds for an arbitrarily long stream.
"""
import os, sys, json, time, argparse, itertools
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import WanModel
from safetensors.torch import load_file

from wanstreamer.core import latent_geometry
from wanstreamer.stream import FewStepStreamer

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='640*368')
ap.add_argument('--weights', default='')
ap.add_argument('--use-ema', action='store_true')
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--blocks', default='1,3', help='latent frames per unit')
ap.add_argument('--steps', default='3,4')
ap.add_argument('--windows', default='6,12')
ap.add_argument('--world', type=int, default=9)
ap.add_argument('--units', type=int, default=14)
ap.add_argument('--warmup-units', type=int, default=6)
ap.add_argument('--fps', type=int, default=25)
ap.add_argument('--out', default='')
args = ap.parse_args()

dev = torch.device('cuda')
cfg = WAN_CONFIGS['t2v-1.3B']
DTYPE = cfg.param_dtype
size = SIZE_CONFIGS[args.size]
h_lat, w_lat, hp, wp = latent_geometry(size[0], size[1])
S = hp * wp

m = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
             num_heads=cfg.num_heads, num_layers=cfg.num_layers,
             window_size=cfg.window_size, qk_norm=True, cross_attn_norm=True,
             eps=1e-6)
m.load_state_dict(load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'),
                  strict=True)
if args.weights:
    sd = torch.load(args.weights, map_location='cpu')
    m.load_state_dict(sd.get('ema' if args.use_ema else 'model', sd), strict=False)
model = m.to(dev).eval().requires_grad_(False)
print(f'{torch.cuda.get_device_name(0)} | {args.size} -> {S} tokens/latent frame')

ctx = torch.randn(1, 512, 4096, device=dev) * 0.4
with torch.amp.autocast('cuda', enabled=False):
    ctx_emb = model.text_embedding(ctx)
world = torch.randn(16, args.world, h_lat, w_lat, device=dev) * 0.9

rows = []
for b, st, win in itertools.product(
        [int(x) for x in args.blocks.split(',')],
        [int(x) for x in args.steps.split(',')],
        [int(x) for x in args.windows.split(',')]):
    stm = FewStepStreamer(model, width=size[0], height=size[1],
                          max_frames=args.world + args.units * b + 8, device=dev,
                          dtype=DTYPE, window_frames=win, time_scale=1000.0,
                          cache_frames=args.world + win + b + 2, block_frames=b,
                          num_steps=st)
    stm.set_text(ctx_emb)
    torch.cuda.reset_peak_memory_stats()
    stm.set_world(world)
    _, times = stm.stream(args.units, seed=0, log_every=0)
    t = np.array(times[args.warmup_units:])
    unit_ms_budget = b * 4 / args.fps * 1000
    ms = t.mean() * 1000
    row = {'block_frames': b, 'unit_ms_of_video': unit_ms_budget, 'steps': st,
           'window': win, 'forwards_per_unit': st + 1,
           'cache_tokens': int(stm.cache.num_tokens),
           'ms_per_unit': float(ms), 'ms_std': float(t.std() * 1000),
           'ms_per_forward': float(ms / (st + 1)),
           'pixel_fps': float(b * 4 / t.mean()),
           'realtime_x': float(b * 4 / t.mean() / args.fps),
           'realtime_met': bool(ms <= unit_ms_budget),
           'peak_gib': torch.cuda.max_memory_allocated() / 2**30}
    rows.append(row)
    print(f'  block {b} ({unit_ms_budget:4.0f} ms video) x {st} steps, window {win:2d} '
          f'-> {ms:7.1f} ms/unit ({row["ms_per_forward"]:6.1f} ms/forward, '
          f'{row["cache_tokens"]:5d} keys) | {row["pixel_fps"]:5.1f} px FPS '
          f'= {row["realtime_x"]:.2f}x real time | '
          f'{"MET" if row["realtime_met"] else "over"} | '
          f'{row["peak_gib"]:.1f} GiB', flush=True)
    del stm
    torch.cuda.empty_cache()

if args.out:
    with open(args.out, 'w') as f:
        json.dump({'device': torch.cuda.get_device_name(0), 'size': args.size,
                   'tokens_per_latent_frame': S, 'rows': rows}, f, indent=2)
    print(f'wrote {args.out}')
