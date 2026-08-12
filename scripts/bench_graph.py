"""CUDA graph streaming: is it faster, and is it the same answer?

A speedup that changes the output is not a speedup. Both streams are driven from
the same seed and the same world, so eager and graphed must agree to bf16
tolerance; the test fails loudly if they do not.
"""
import os, sys, time, argparse
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
ap.add_argument('--weights', default=os.path.join(HERE, 'checkpoints/dmd/latest.pt'))
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--steps', type=int, default=4)
ap.add_argument('--window', type=int, default=12)
ap.add_argument('--world', type=int, default=9)
ap.add_argument('--units', type=int, default=16)
ap.add_argument('--warmup-units', type=int, default=8)
ap.add_argument('--fps', type=int, default=25)
args = ap.parse_args()

dev = torch.device('cuda')
cfg = WAN_CONFIGS['t2v-1.3B']
DTYPE = cfg.param_dtype
h_lat, w_lat, hp, wp = latent_geometry(*SIZE_CONFIGS['640*368'])
m = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
             num_heads=cfg.num_heads, num_layers=cfg.num_layers,
             window_size=cfg.window_size, qk_norm=True, cross_attn_norm=True,
             eps=1e-6)
m.load_state_dict(load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'),
                  strict=True)
if args.weights:
    sd = torch.load(args.weights, map_location='cpu')
    m.load_state_dict(sd.get('model', sd), strict=False)
model = m.to(dev).eval().requires_grad_(False)

torch.manual_seed(0)
ctx = torch.randn(1, 512, 4096, device=dev) * 0.4
with torch.amp.autocast('cuda', enabled=False):
    ctx_emb = model.text_embedding(ctx)
world = torch.randn(16, args.world, h_lat, w_lat, device=dev) * 0.9


def run(graphs):
    st = FewStepStreamer(model, width=640, height=368,
                         max_frames=args.world + args.units * args.block + 8,
                         device=dev, dtype=DTYPE, window_frames=args.window,
                         time_scale=1000.0,
                         cache_frames=args.world + args.window + args.block + 2,
                         block_frames=args.block, num_steps=args.steps,
                         cuda_graphs=graphs)
    st.set_text(ctx_emb)
    torch.cuda.reset_peak_memory_stats()
    st.set_world(world)
    lat, times = st.stream(args.units, seed=7, log_every=0)
    t = np.array(times[args.warmup_units:])
    return lat, t, st, torch.cuda.max_memory_allocated() / 2**30


lat_e, t_e, st_e, mem_e = run(False)
lat_g, t_g, st_g, mem_g = run(True)

budget = args.block * 4 / args.fps * 1000
rel = (lat_g - lat_e).norm().item() / max(lat_e.norm().item(), 1e-12)
print(f'\nblock {args.block} x {args.steps} steps, window {args.window}, '
      f'{st_e.cache.num_tokens} keys, unit = {budget:.0f} ms of video')
for name, t, mem in (('eager ', t_e, mem_e), ('graphed', t_g, mem_g)):
    ms = t.mean() * 1000
    print(f'  {name}: {ms:7.1f} ms/unit  ({ms/(args.steps+1):6.1f} ms/forward)  '
          f'{args.block*4/t.mean():5.1f} px FPS = {args.block*4/t.mean()/args.fps:.2f}x '
          f'real time  {"MET" if ms <= budget else "over"}  {mem:.1f} GiB')
print(f'  speedup {t_e.mean()/t_g.mean():.2f}x   '
      f'graphs captured {st_g.graphs.captures}, replays {st_g.graphs.replays}')
print(f'  output agreement (graphed vs eager): rel_err {rel:.5f} '
      f'{"OK" if rel < 2e-2 else "FAIL -- graphs changed the answer"}')
sys.exit(0 if rel < 2e-2 else 1)
