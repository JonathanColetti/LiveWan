"""Generate the teacher ODE dataset: original bidirectional Wan2.1-1.3B samples.

This is the distribution the causal student is distilled onto. It is teacher
output rather than real video on purpose -- at deployment the student's context
is *its own* continuation of a teacher-generated world, never real footage, so
teacher samples are the deployment distribution and real data would be a domain
shift, not an upgrade (README.md §8.1).

Shards over GPUs by sample index; each shard writes independent .pt files, so a
crashed shard costs only its own remainder and re-running skips what exists.
"""
import os, sys, time, json, math, argparse
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import WanModel
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from safetensors.torch import load_file

from wanstreamer.core import latent_geometry

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='640*368')
ap.add_argument('--frames', type=int, default=81, help='pixel frames')
ap.add_argument('--steps', type=int, default=40)
ap.add_argument('--cfg', type=float, default=7.0)
ap.add_argument('--shift', type=float, default=5.0)
ap.add_argument('--per-prompt', type=int, default=6)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--num-shards', type=int, default=1)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--prompts', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--out', default=os.path.join(HERE, 'data/teacher'))
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
dev = torch.device('cuda')
cfg = WAN_CONFIGS['t2v-1.3B']
DTYPE = cfg.param_dtype
size = SIZE_CONFIGS[args.size]
h_lat, w_lat, hp, wp = latent_geometry(size[0], size[1])
F_lat = (args.frames - 1) // cfg.vae_stride[0] + 1
seq_len = math.ceil(hp * wp * F_lat)

blob = torch.load(args.prompts, map_location='cpu')
pos_all, neg = blob['pos'], blob['neg'][0].float().to(dev)
n_prompts = pos_all.shape[0]

samples = [(p, s) for p in range(n_prompts) for s in range(args.per_prompt)]
samples = samples[args.shard::args.num_shards]
if args.limit:
    samples = samples[:args.limit]
print(f'[shard {args.shard}/{args.num_shards}] {len(samples)} samples, '
      f'{args.size} {args.frames}px -> {F_lat} latent frames, seq_len {seq_len}')

model = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
model.load_state_dict(
    load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'), strict=True)
model = model.to(dev).eval().requires_grad_(False)

t0_all = time.time()
done = 0
for i, (pi, si) in enumerate(samples):
    path = os.path.join(args.out, f'p{pi:04d}_s{si:02d}.pt')
    if os.path.exists(path):
        continue
    g = torch.Generator(device='cpu').manual_seed(pi * 1000 + si)
    lat = torch.randn(16, F_lat, h_lat, w_lat, generator=g).to(dev)
    pos = pos_all[pi].float().to(dev)

    sch = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=cfg.num_train_timesteps, shift=1,
        use_dynamic_shifting=False)
    tsteps, _ = retrieve_timesteps(
        sch, device=dev, sigmas=get_sampling_sigmas(args.steps, args.shift))

    t0 = time.time()
    with torch.amp.autocast('cuda', dtype=DTYPE), torch.no_grad():
        for tv in tsteps:
            tt = torch.stack([torch.tensor(tv, device=dev)] * 2)
            # cond and uncond in one batch: same FLOPs, half the launches
            out = model([lat, lat], t=tt, context=[pos, neg], seq_len=seq_len)
            pred = out[1] + args.cfg * (out[0] - out[1])
            lat = sch.step(pred.unsqueeze(0), tv, lat.unsqueeze(0),
                           return_dict=False)[0].squeeze(0)
    dt = time.time() - t0
    torch.save({'latents': lat.to(torch.float16).cpu(), 'prompt_idx': pi,
                'seed': si, 'size': args.size, 'frames': args.frames,
                'steps': args.steps, 'cfg': args.cfg}, path)
    done += 1
    if done % 5 == 1 or done == 1:
        el = time.time() - t0_all
        rate = el / max(done, 1)
        print(f'  [{args.shard}] {i+1}/{len(samples)} {dt:.1f}s/video '
              f'| elapsed {el/60:.1f}m | eta {(len(samples)-i-1)*rate/60:.0f}m',
              flush=True)

print(f'[shard {args.shard}] wrote {done} samples in '
      f'{(time.time()-t0_all)/60:.1f} min')
