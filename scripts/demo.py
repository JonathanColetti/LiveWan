"""End-to-end streaming demo and benchmark at the papers' operating point.

    Stage 1  original bidirectional Wan2.1-1.3B, CFG    -> the WORLD W
    Stage 2  VAE encode                                 -> world latents
    Stage 3  prime block-causal K/V from W              -> v0.2 "KV construction"
    Stage 4  stream N event units autoregressively      -> the event stream
    Stage 5  VAE decode + no-reference quality stats

Reports the numbers the target operating point is defined by: 640x368 at 25 FPS,
a 160 ms streaming unit, and the model-side latency per unit. A unit here is
`--block` latent frames; at VAE temporal stride 4 one latent frame is 160 ms of
video, so `--block 1` is the papers' unit exactly and `--block 3` trades
interaction granularity for throughput.

  python scripts/demo.py --weights checkpoints/dmd/latest.pt --use-ema \
      --units 60 --block 3 --steps 4 --window 12 --out out/demo
"""
import os, sys, gc, time, json, math, argparse, subprocess
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import WanModel
from wan.modules.vae import WanVAE
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from safetensors.torch import load_file

from wanstreamer.core import latent_geometry
from wanstreamer.stream import FewStepStreamer
from wanstreamer.metrics import (video_stats, latent_stats, summary_line,
                                 channel_drift)

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='640*368')
ap.add_argument('--weights', default=os.path.join(HERE, 'checkpoints/dmd/latest.pt'))
ap.add_argument('--use-ema', action='store_true')
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--prompts', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--prompt-idx', type=int, default=0)
ap.add_argument('--prompt', default='')
ap.add_argument('--world-frames', type=int, default=81,
                help='pixel frames the teacher generates. Wan2.1-1.3B is trained '
                     'at 81; asking it for a short clip measurably degrades it, '
                     'so generate the native length and keep a prefix.')
ap.add_argument('--world-latent-frames', type=int, default=9,
                help='latent frames of that clip actually used as the world W')
ap.add_argument('--base-steps', type=int, default=32)
ap.add_argument('--base-cfg', type=float, default=7.0)
ap.add_argument('--units', type=int, default=60)
ap.add_argument('--block', type=int, default=3, help='latent frames per unit')
ap.add_argument('--steps', type=int, default=4, help='denoising steps per unit')
ap.add_argument('--window', type=int, default=12, help='event K/V window, latent frames')
ap.add_argument('--sampler', default='renoise', choices=['renoise', 'euler'],
                help='renoise = the DMD few-step sampler the student is trained '
                     'under; euler = deterministic, for the undistilled model')
ap.add_argument('--latent-norm', type=float, default=0.0,
                help='per-channel moment match each finished unit towards the '
                     'world latents (0=off, 1=full); counters colour/saturation '
                     'drift over a long rollout')
ap.add_argument('--noisy-context', action='store_true',
                help='commit the last denoising step K/V instead of re-running '
                     'the finished unit at t=0: N forwards per unit instead of '
                     'N+1. Needs a student trained under the same convention '
                     '(see FewStepStreamer.noisy_context) -- use it to measure '
                     'the latency headroom, not to ship.')
ap.add_argument('--sched-shift', type=float, default=1.0,
                help='1.0 = uniform few-step schedule; 5.0 = Wan inference schedule')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--out', default=os.path.join(HERE, 'out/demo'))
ap.add_argument('--world-cache', default='')
ap.add_argument('--fps', type=int, default=25)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
dev = torch.device('cuda')
cfg = WAN_CONFIGS['t2v-1.3B']
DTYPE = cfg.param_dtype
size = SIZE_CONFIGS[args.size]
h_lat, w_lat, hp, wp = latent_geometry(size[0], size[1])
S = hp * wp
torch.manual_seed(args.seed)

blob = torch.load(args.prompts, map_location='cpu')
if args.prompt:
    raise SystemExit('--prompt needs a T5 pass; use --prompt-idx into data/prompts.pt')
prompt = blob['prompts'][args.prompt_idx]
ctx_pos = blob['pos'][args.prompt_idx].float().to(dev).unsqueeze(0)
ctx_neg = blob['neg'][0].float().to(dev).unsqueeze(0)
print(f'=== {args.size} -> {S} tokens/latent frame | unit = {args.block} latent '
      f'frames = {args.block*4/args.fps*1000:.0f} ms of video ===')
print(f'prompt[{args.prompt_idx}]: {prompt}')


def new_model(weights=None, use_ema=False):
    m = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
    m.load_state_dict(load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'),
                      strict=True)
    if weights:
        sd = torch.load(weights, map_location='cpu')
        key = 'ema' if (use_ema and 'ema' in sd) else 'model'
        if use_ema and 'ema' not in sd:
            raise SystemExit(f'--use-ema but {weights} has no EMA weights')
        src = sd.get(key, sd)
        # strict=False is needed because the EMA dict deliberately omits the
        # non-float `freqs` buffer, but a checkpoint that silently failed to
        # load would report base-model quality as if it were trained -- so
        # check that every parameter actually came from the file.
        res = m.load_state_dict(src, strict=False)
        got = {n for n, _ in m.named_parameters()} - set(res.missing_keys)
        if len(got) != len(list(m.named_parameters())) or res.unexpected_keys:
            raise SystemExit(f'bad checkpoint load: {len(res.missing_keys)} '
                             f'missing, {len(res.unexpected_keys)} unexpected')
        print(f'    loaded {key} from {os.path.basename(weights)} '
              f'(step {sd.get("step", "?")}, all {len(got)} tensors matched)')
    return m.to(dev).eval().requires_grad_(False)


def encode_mp4(path, arr, fps, crf=17):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt',
           'rgb24', '-s', f'{arr.shape[2]}x{arr.shape[1]}', '-r', str(fps),
           '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(arr).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path}')


vae = WanVAE(vae_pth=f'{args.ckpt}/{cfg.vae_checkpoint}', device=dev)

# ============================================================ Stage 1/2: world
wc = args.world_cache
if wc and os.path.exists(wc):
    print(f'\n--- Stage 1/2: cached world {wc} ---')
    wb = torch.load(wc, map_location='cpu')
    world_pix, world_lat = wb['pixels'], wb['latents'].to(dev)
    base_time = wb['base_seconds']
else:
    print(f'\n--- Stage 1: world via original Wan2.1 ({args.world_frames} px '
          f'frames, {args.base_steps} steps, CFG {args.base_cfg}) ---')
    base = new_model()
    Fl = (args.world_frames - 1) // cfg.vae_stride[0] + 1
    lat = torch.randn(16, Fl, h_lat, w_lat, device=dev)
    sch = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, shift=1,
                                          use_dynamic_shifting=False)
    tsteps, _ = retrieve_timesteps(sch, device=dev,
                                   sigmas=get_sampling_sigmas(args.base_steps, 5.0))
    t0 = time.time()
    with torch.amp.autocast('cuda', dtype=DTYPE), torch.no_grad():
        for tv in tsteps:
            tt = torch.stack([tv.clone().to(dev)] * 2)
            o = base([lat, lat], t=tt, context=[ctx_pos[0], ctx_neg[0]],
                     seq_len=S * Fl)
            pred = o[1] + args.base_cfg * (o[0] - o[1])
            lat = sch.step(pred.unsqueeze(0), tv, lat.unsqueeze(0),
                           return_dict=False)[0].squeeze(0)
    base_time = time.time() - t0
    with torch.amp.autocast('cuda', dtype=DTYPE):
        wpx = vae.decode([lat])[0]
    world_pix = ((wpx.permute(1, 2, 3, 0).cpu() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    del base
    gc.collect(); torch.cuda.empty_cache()
    print('--- Stage 2: VAE encode world ---')
    wt = world_pix.float().permute(3, 0, 1, 2) / 127.5 - 1.0
    world_lat = vae.encode([wt.to(dev)])[0]
    if wc:
        torch.save({'pixels': world_pix, 'latents': world_lat.cpu(),
                    'base_seconds': base_time}, wc)
if args.world_latent_frames and world_lat.shape[1] > args.world_latent_frames:
    world_lat = world_lat[:, :args.world_latent_frames]
    world_pix = world_pix[:(args.world_latent_frames - 1) * 4 + 1]
print(f'    world: {world_pix.shape[0]} px frames -> latents {list(world_lat.shape)} '
      f'({base_time:.1f}s)')

# ====================================================== Stage 3/4: prime, stream
print('\n--- Stage 3: prime block-causal K/V from W ---')
student = new_model(args.weights, args.use_ema)
with torch.amp.autocast('cuda', enabled=False):
    ctx_emb = student.text_embedding(ctx_pos)
nw = world_lat.shape[1]
total_frames = nw + args.units * args.block + 4
st = FewStepStreamer(student, width=size[0], height=size[1],
                     max_frames=total_frames, device=dev, dtype=DTYPE,
                     window_frames=args.window, time_scale=1000.0,
                     cache_frames=nw + args.window + args.block + 2,
                     block_frames=args.block, num_steps=args.steps,
                     shift=args.sched_shift, sampler=args.sampler,
                     noisy_context=args.noisy_context)
st.set_text(ctx_emb)
st.latent_norm = args.latent_norm
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
st.set_world(world_lat)
prime_ms = (time.time() - t0) * 1000
print(f'    primed {nw} world frames ({st.cache.num_tokens} tokens) in '
      f'{prime_ms:.0f} ms | K/V buffer {st.cache.memory_bytes()/2**20:.0f} MiB')

print(f'\n--- Stage 4: stream {args.units} units x {args.steps} steps '
      f'(block={args.block}f window={args.window}f) ---')
lat_ev, times = st.stream(args.units, seed=args.seed,
                          log_every=max(1, args.units // 6))
times = np.array(times)
peak = torch.cuda.max_memory_allocated() / 2**30
sust = times[1:] if len(times) > 1 else times
px_per_unit = 4 * args.block
budget = px_per_unit / args.fps
print(f'\n    per-unit ms: mean {times.mean()*1000:.1f}  median '
      f'{np.median(times)*1000:.1f}  min {times.min()*1000:.1f}  '
      f'max {times.max()*1000:.1f}')
print(f'    sustained (excl. first): {sust.mean()*1000:.1f} ms/unit '
      f'-> {px_per_unit/sust.mean():.1f} pixel FPS '
      f'({px_per_unit/sust.mean()/args.fps:.2f}x real time)')
print(f'    real-time budget {budget*1000:.0f} ms/unit => '
      f'{"MET" if sust.mean() <= budget else "NOT MET"}')
print(f'    peak GPU memory {peak:.1f} GiB')

# ============================================================== Stage 5: decode
print('\n--- Stage 5: decode + quality ---')
full = torch.cat([world_lat, lat_ev.to(world_lat.dtype)], dim=1)
t0 = time.time()
with torch.amp.autocast('cuda', dtype=DTYPE):
    vid = vae.decode([full.clamp(-4, 4)])[0].cpu()
dec = time.time() - t0
pix = ((vid.permute(1, 2, 3, 0) + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).numpy()
n = pix.shape[0]
world_px = (nw - 1) * 4 + 1
vs = video_stats(pix, world_px)
ls = latent_stats(world_lat, lat_ev)
# Per-channel colour drift: a catastrophe detector, not a ranker
# (FINDINGS.md §5a). Reads ~0 whenever --latent-norm is on, since
# that patch optimises exactly this quantity.
ls.update(channel_drift(world_lat, lat_ev))
print(f'    decoded {n} frames ({n/args.fps:.1f}s) in {dec:.1f}s '
      f'({dec/n*1000:.1f} ms/frame); world = first {world_px}')
print(f'    {summary_line(vs, ls)}')
print(f'    sharpness world {vs["sharpness_world"]:.5f} -> gen {vs["sharpness_gen"]:.5f}')

encode_mp4(os.path.join(args.out, 'world.mp4'), world_pix.numpy(), args.fps)
encode_mp4(os.path.join(args.out, 'stream.mp4'), pix, args.fps)
meta = {'size': args.size, 'weights': args.weights, 'use_ema': args.use_ema,
        'sampler': args.sampler, 'sched_shift': args.sched_shift,
        'noisy_context': args.noisy_context,
        'latent_norm': args.latent_norm,
        'block_frames': args.block, 'unit_ms': args.block * 4 / args.fps * 1000,
        'steps_per_unit': args.steps, 'window_frames': args.window,
        'units': args.units, 'tokens_per_frame': S, 'seed': args.seed,
        'prompt': prompt, 'prompt_idx': args.prompt_idx,
        'world_latent_frames': int(nw), 'world_pixel_frames': int(world_px),
        'total_pixel_frames': int(n), 'video_seconds': n / args.fps,
        'base_generation_s': base_time, 'prime_ms': prime_ms,
        'per_unit_ms': (times * 1000).tolist(),
        'sustained_ms_per_unit': float(sust.mean() * 1000),
        'sustained_pixel_fps': float(px_per_unit / sust.mean()),
        'realtime_budget_ms': budget * 1000,
        'realtime_met': bool(sust.mean() <= budget),
        'peak_gib': peak, 'decode_ms_per_frame': dec / n * 1000,
        'video_stats': vs, 'latent_stats': ls}
with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
    json.dump(meta, f, indent=2)
torch.save({'latents': full.to(torch.float16).cpu(), 'world_frames': int(nw),
            'prompt_idx': args.prompt_idx, 'prompt': prompt},
           os.path.join(args.out, 'latents.pt'))
np.save(os.path.join(args.out, 'frames_idx.npy'),
        np.linspace(0, n - 1, min(30, n)).astype(int))
print(f'\nwrote {args.out}/stream.mp4 and metrics.json')
