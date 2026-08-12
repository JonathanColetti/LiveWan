"""What actually fits, at 1.3B and at 14B, measured rather than estimated.

SCALING_PROMPT §2 asserts a 14B student needs >=16xH100-80GB and that 4xA100-40GB
cannot do it. That is an arithmetic claim, and arithmetic is how this project got
several things wrong before. This runs the REAL forward -- the block-causal one,
at 640x368, 21 latent frames, with a clean K/V prefix -- under FSDP2 at a chosen
size and precision, and reports what the allocator actually did.

Three regimes, because they answer three different questions:

  frozen   bf16, no grads, no optimizer. The DMD *teacher* and LoRA-critic base.
           This is the regime that decides whether a 14B teacher can score a
           1.3B student on this box -- which is what the published recipe
           (Causal-rCM, arXiv:2606.25473) does, and it needs no 14B student.
  fwdbwd   bf16 params + grads, no optimizer. Proves the sharded compute path
           scales to a size, separately from whether the optimizer state fits.
  train    fp32 params + grads + AdamW. The full training footprint.

The footer prints the per-GPU breakdown and how many 40 GB / 80 GB cards the
`train` regime would need, from the measured parameter count rather than a
remembered figure.

  torchrun --nproc_per_node=4 scripts/fsdp_scale_check.py --size 1.3B --mode train
  torchrun --nproc_per_node=4 scripts/fsdp_scale_check.py --size 14B  --mode frozen
  torchrun --nproc_per_node=4 scripts/fsdp_scale_check.py --size 14B  --mode fwdbwd
"""
import os, sys, time, argparse, json

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import WanModel

from wanstreamer import blockcausal as bc
from wanstreamer.core import latent_geometry, make_rope_table
from wanstreamer.fsdp import shard_model, full_state_dict

SIZES = {
    # dim, ffn_dim, num_heads, num_layers, in/out_dim -- from each repo's own
    # config.json. head_dim is 128 in all three, which is why one RoPE table
    # implementation serves them all.
    '1.3B': dict(dim=1536, ffn_dim=8960, num_heads=12, num_layers=30, in_dim=16),
    '14B': dict(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40, in_dim=16),
    # Wan2.2-TI2V-5B. 48 latent channels and a 16x-spatial VAE, so it needs
    # --vae-stride 16 and a resolution divisible by 32 (640x368 is not: 368/16
    # = 23 latent rows, which the 2x2 patch cannot halve).
    '5B': dict(dim=3072, ffn_dim=14336, num_heads=24, num_layers=30, in_dim=48),
}

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='1.3B', choices=list(SIZES))
ap.add_argument('--mode', default='train', choices=['frozen', 'fwdbwd', 'train'])
ap.add_argument('--res', default='640*368')
ap.add_argument('--vae-stride', type=int, default=8,
                help='VAE spatial stride: 8 for Wan2.1, 16 for Wan2.2-TI2V-5B')
ap.add_argument('--frames', type=int, default=21)
ap.add_argument('--world-frames', type=int, default=9)
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--steps', type=int, default=3)
ap.add_argument('--reshard', action='store_true',
                help='reshard after forward (less memory, more collectives)')
ap.add_argument('--no-fsdp', action='store_true', help='replicate, for contrast')
ap.add_argument('--out', default='')
args = ap.parse_args()

rank = int(os.environ.get('RANK', 0))
world = int(os.environ.get('WORLD_SIZE', 1))
local = int(os.environ.get('LOCAL_RANK', 0))
dist.init_process_group('nccl')
torch.cuda.set_device(local)
dev = torch.device(f'cuda:{local}')
main = rank == 0
GIB = 2 ** 30


def log(m):
    if main:
        print(m, flush=True)


size = SIZE_CONFIGS[args.res] if args.res in SIZE_CONFIGS else \
    tuple(int(x) for x in args.res.split('*'))
h_lat, w_lat, hp, wp = latent_geometry(
    size[0], size[1], vae_stride=(4, args.vae_stride, args.vae_stride))
S = hp * wp
cfg = dict(SIZES[args.size])
CH = cfg.pop('in_dim')
DT = torch.bfloat16 if args.mode != 'train' else torch.float32

log(f'=== FSDP scale check | {args.size} | mode {args.mode} | {world} x '
    f'{torch.cuda.get_device_name(local)} ===')
log(f'    {args.res} -> {S} tokens/latent frame, {args.frames} frames, '
    f'block {args.block}, param dtype {str(DT).split(".")[-1]}')

torch.manual_seed(0)
model = WanModel(qk_norm=True, cross_attn_norm=True, eps=1e-6, freq_dim=256,
                 text_len=512, window_size=(-1, -1), in_dim=CH, out_dim=CH,
                 **cfg)
n_params = sum(p.numel() for p in model.parameters())
model = model.to(DT)
if args.mode == 'frozen':
    model.requires_grad_(False).eval()
else:
    model.train()

if not args.no_fsdp:
    shard_model(model, reshard_after_forward=args.reshard)
model = model.to(dev)
n_local = sum(p.to_local().numel() if hasattr(p, 'to_local') else p.numel()
              for p in model.parameters())
log(f'    {n_params/1e9:.2f}B params -> {n_local/1e9:.3f}B local per rank')

opt = None
if args.mode == 'train':
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95),
                            weight_decay=1e-4, foreach=True)

rope = make_rope_table(model, hp, wp, args.frames + 8, dev)
ctx = torch.randn(1, 512, cfg['dim'], device=dev, dtype=DT)
z0 = torch.randn(1, CH, args.frames, h_lat, w_lat, device=dev)

torch.cuda.reset_peak_memory_stats()
after_model = torch.cuda.memory_allocated() / GIB
times, oom = [], None

try:
    for it in range(args.steps):
        t0 = time.time()
        # The real Stage-1 step: lay a clean K/V prefix down exactly as
        # deployment does, then train every event block against its own slice.
        with torch.no_grad():
            kv = bc.build_clean_kv(model, z0.to(DT), rope, ctx, None,
                                   args.world_frames, args.block, dtype=DT)
        starts = bc.block_starts(args.frames, args.world_frames, args.block)
        grad = args.mode != 'frozen'
        with torch.set_grad_enabled(grad):
            loss = 0.0
            for s in starts:
                v = bc.block_forward(
                    model, z0[:, :, s:s + args.block], 0.5, s, rope, ctx, None,
                    kv=kv, prefix_upto=s * S, dtype=DT, grad_checkpoint=grad)
                loss = loss + v.float().pow(2).mean()
            if grad:
                loss.backward()
        if opt is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        elif grad:
            model.zero_grad(set_to_none=True)
        del kv
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        log(f'    step {it+1}/{args.steps}: {times[-1]:.2f}s  '
            f'peak {torch.cuda.max_memory_allocated()/GIB:.1f} GiB')
except torch.cuda.OutOfMemoryError as e:
    oom = str(e).splitlines()[0][:120]
    log(f'    OOM: {oom}')

peak = torch.cuda.max_memory_allocated() / GIB
tot = torch.cuda.get_device_properties(local).total_memory / GIB

# ---------------------------------------------------------------- the verdict
bpp = {'frozen': 2, 'fwdbwd': 4, 'train': 16}[args.mode]   # bytes per parameter
log(f'\n--- {args.size} / {args.mode} on {world} GPUs ---')
log(f'    model resident after shard : {after_model:6.1f} GiB/GPU')
log(f'    peak allocated             : {peak:6.1f} GiB/GPU  of {tot:.0f} GiB'
    + ('  <-- OOM' if oom else ''))
if times:
    log(f'    step time                  : {min(times):6.2f} s '
        f'(best of {len(times)})')
log(f'    state ({bpp} B/param) sharded /{world} : '
    f'{n_params*bpp/world/GIB:6.1f} GiB/GPU '
    f'(replicated would be {n_params*bpp/GIB:.1f})')

if main and args.mode == 'train':
    full = n_params * 16 / GIB
    log(f'\n    A full fp32+AdamW student of this size is {full:.0f} GiB of '
        f'optimizer state alone.')
    for cap, name in ((40, 'A100-40GB'), (80, 'H100-80GB')):
        # leave ~35% for activations, the K/V cache and the frozen teacher
        need = int(-(-full // (cap * 0.65)))
        log(f'      -> needs >= {need:3d} x {name} just to hold it '
            f'(at 65% of {cap} GiB for state)')

if main and args.out:
    json.dump({'size': args.size, 'mode': args.mode, 'world': world,
               'params': n_params, 'local_params': n_local,
               'peak_gib': peak, 'model_gib': after_model,
               'step_s': min(times) if times else None, 'oom': oom},
              open(args.out, 'w'), indent=1)

dist.barrier()
dist.destroy_process_group()
