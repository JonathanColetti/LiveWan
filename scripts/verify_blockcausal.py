"""Correctness gate for wanstreamer/blockcausal.py.

Every claim the training code rests on is checked against the already-tested
baseline (wanstreamer/pipeline.py), with a deliberately broken control for each
so a passing number means something:

  1. block_forward(uniform t)      == pipeline.chunk_forward          (same path)
  2. per-frame t, all equal        == uniform t                       (new code
     collapses to upstream when the timestep vector is constant)
  3. per-frame t, actually varying != uniform t                       (control:
     the new conditioning must do something)
  4. parallel_blocks_forward       == sequential block_forward, block by block,
     against the same clean K/V     (the batched trainer sees the deployment
     context exactly)
  5. control for 4: shuffle the key-padding lengths -> must diverge   (proves
     the prefix slicing is load-bearing, not incidental)
"""
import os, sys, argparse
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import WanModel
from safetensors.torch import load_file

from wanstreamer.core import latent_geometry, make_rope_table, make_cache
from wanstreamer.pipeline import StreamingGenerator
from wanstreamer import blockcausal as bc

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='640*368')
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--world-frames', type=int, default=9)
ap.add_argument('--frames', type=int, default=21)
ap.add_argument('--block', type=int, default=3)
args = ap.parse_args()

torch.manual_seed(0)
dev = torch.device('cuda:0')
cfg = WAN_CONFIGS['t2v-1.3B']
size = SIZE_CONFIGS[args.size]
h_lat, w_lat, hp, wp = latent_geometry(size[0], size[1])
S = hp * wp
print(f'{args.size} -> latent {h_lat}x{w_lat}, {S} tokens/latent frame')

model = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
model.load_state_dict(load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'),
                      strict=True)
model = model.to(dev).eval().requires_grad_(False)
print(f'loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, '
      f'dtype {next(model.parameters()).dtype}')

ctx = torch.randn(1, 512, 4096, device=dev) * 0.5
with torch.amp.autocast('cuda', enabled=False):
    ctx_emb = model.text_embedding(ctx)
ctx_lens = torch.tensor([512], device=dev, dtype=torch.long)

F = args.frames
z = torch.randn(1, 16, F, h_lat, w_lat, device=dev)
rope = make_rope_table(model, hp, wp, F + 8, dev)


def rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def report(name, err, tol, control=None, direction='below'):
    ok = err < tol if direction == 'below' else err > tol
    tag = 'PASS' if ok else 'FAIL'
    extra = f'  (control {control:.4f})' if control is not None else ''
    print(f'  [{tag}] {name:52s} rel_err {err:.5f}  tol {tol}{extra}')
    return ok


ok = True
print('\n--- 1. block_forward vs the tested pipeline.chunk_forward ---')
gen = StreamingGenerator(model, ctx_emb, width=size[0], height=size[1],
                         max_frames=F + 8, device=dev, dtype=torch.bfloat16,
                         time_scale=1000.0)
gen.set_world(z[0, :, :args.world_frames])
t_test = 0.6
zb = torch.randn(1, 16, args.block, h_lat, w_lat, device=dev)
ref = gen.chunk_forward(zb, t_test, t_start=args.world_frames).float()

cache2 = make_cache(model, S, F + 8, dev, torch.bfloat16)
gen2 = StreamingGenerator(model, ctx_emb, width=size[0], height=size[1],
                          max_frames=F + 8, device=dev, dtype=torch.bfloat16,
                          time_scale=1000.0)
gen2.set_world(z[0, :, :args.world_frames])
kvb = bc.BufferKV(gen2.cache)
new = bc.block_forward(model, zb, t_test, args.world_frames, rope, ctx_emb,
                       ctx_lens, kv=kvb, time_scale=1000.0)[0].float()
ok &= report('block_forward == chunk_forward', rel(new, ref), 5e-3)

print('\n--- 2/3. per-frame timestep conditioning ---')
gen2.cache.length = args.world_frames * S
tvec_const = torch.full((args.block,), t_test, device=dev)
pf = bc.block_forward(model, zb, tvec_const, args.world_frames, rope, ctx_emb,
                      ctx_lens, kv=bc.BufferKV(gen2.cache), per_frame=True,
                      time_scale=1000.0)[0].float()
ok &= report('per-frame t (constant) == uniform t', rel(pf, ref), 5e-3)

gen2.cache.length = args.world_frames * S
tvec_var = torch.tensor([0.2, 0.6, 0.9][:args.block], device=dev)
pv = bc.block_forward(model, zb, tvec_var, args.world_frames, rope, ctx_emb,
                      ctx_lens, kv=bc.BufferKV(gen2.cache), per_frame=True,
                      time_scale=1000.0)[0].float()
ok &= report('CONTROL per-frame t (varying) != uniform t', rel(pv, ref), 0.05,
             direction='above')

print('\n--- 4/5. batched-block trainer vs sequential, same clean K/V ---')
kv = bc.build_clean_kv(model, z, rope, ctx_emb, ctx_lens, args.world_frames,
                       args.block)
starts = bc.block_starts(F, args.world_frames, args.block)
print(f'  clean K/V: {kv.tokens} tokens ({kv.tokens//S} latent frames), '
      f'blocks at {starts}')
t_rows = torch.tensor([0.9, 0.7, 0.5, 0.3][:len(starts)], device=dev)
z_noisy = torch.randn_like(z)

seq = []
for i, s in enumerate(starts):
    seq.append(bc.block_forward(model, z_noisy[:, :, s:s + args.block],
                                float(t_rows[i]), s, rope, ctx_emb, ctx_lens,
                                kv=kv, prefix_upto=s * S)[0].float())
seq = torch.stack(seq)
par = bc.parallel_blocks_forward(model, z_noisy, t_rows, starts, rope, ctx_emb,
                                 ctx_lens, kv, args.block,
                                 grad_checkpoint=False).float()
ok &= report('parallel_blocks_forward == sequential', rel(par, seq), 2e-2)

bad = bc.parallel_blocks_forward(model, z_noisy, t_rows, list(reversed(starts)),
                                 rope, ctx_emb, ctx_lens, kv, args.block,
                                 grad_checkpoint=False).float()
ok &= report('CONTROL reversed block starts != sequential',
             rel(bad, torch.stack(list(reversed(list(seq))))), 0.05,
             direction='above')

print('\n--- 6/7. bidirectional path used for teacher and critic ---')
# block_forward with no K/V prefix over the whole clip IS the bidirectional
# forward; the DMD score networks rely on that, so check it against upstream.
from wanstreamer.dmd import bidirectional_velocity
t_bi = 0.4
z_bi = torch.randn(1, 16, F, h_lat, w_lat, device=dev)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    up = model([z_bi[0]], t=torch.tensor([t_bi * 1000.0], device=dev),
               context=[ctx[0]], seq_len=F * S)[0].float()
mine = bidirectional_velocity(model, z_bi, t_bi, rope, ctx_emb, ctx_lens,
                              torch.bfloat16)[0].float()
ok &= report('bidirectional_velocity == WanModel.forward', rel(mine, up), 2e-2)

# The teacher/critic base is held in bf16 to fit three networks on one card.
import copy
m16 = copy.deepcopy(model).to(torch.bfloat16)
rope16 = make_rope_table(m16, hp, wp, F + 8, dev)
with torch.amp.autocast('cuda', enabled=False):
    ctx16 = m16.text_embedding(ctx.to(torch.bfloat16))
half = bidirectional_velocity(m16, z_bi, t_bi, rope16, ctx16, ctx_lens,
                              torch.bfloat16)[0].float()
ok &= report('bf16 weights == fp32 weights (same path)', rel(half, mine), 5e-2)
del m16

print(f'\npeak GPU memory {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
