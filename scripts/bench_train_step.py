"""Where the Stage-1 step time goes, and whether the batched-block trainer is
actually faster than running the blocks sequentially.

The batched path needs a key-padding mask (rows have different prefix lengths),
and a masked SDPA call cannot use the flash backend. The sequential path needs
no mask at all and touches strictly fewer key tokens, at the cost of 4x the
kernel launches. Which wins is an empirical question, so measure it.
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as tnnF

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import WanModel
from safetensors.torch import load_file

from wanstreamer import blockcausal as bc
from wanstreamer.core import latent_geometry, make_rope_table
from wanstreamer.data import shifted_uniform_t, add_noise

ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=21)
ap.add_argument('--world', type=int, default=9)
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--iters', type=int, default=3)
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
args = ap.parse_args()

dev = torch.device('cuda')
cfg = WAN_CONFIGS['t2v-1.3B']
h_lat, w_lat, hp, wp = latent_geometry(*SIZE_CONFIGS['640*368'])
S = hp * wp
model = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
model.load_state_dict(
    load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'), strict=True)
model = model.to(dev).train()

F = args.frames
z0 = torch.randn(1, 16, F, h_lat, w_lat, device=dev)
ctx = torch.randn(1, 512, 4096, device=dev) * 0.4
with torch.amp.autocast('cuda', enabled=False):
    ctx_emb = model.text_embedding(ctx).detach()
ctx_lens = torch.tensor([512], device=dev, dtype=torch.long)
rope = make_rope_table(model, hp, wp, F + 8, dev)
starts = bc.block_starts(F, args.world, args.block)
print(f'F={F} world={args.world} block={args.block} starts={starts} '
      f'tokens/frame={S}')


def timed(fn, n, label):
    torch.cuda.synchronize()
    fn()                                  # warm up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f'  {label:44s} {dt*1000:8.1f} ms')
    return dt


kv_holder = {}


def clean_kv():
    kv_holder['kv'] = bc.build_clean_kv(model, z0, rope, ctx_emb, ctx_lens,
                                        args.world, args.block)


t_clean = timed(clean_kv, args.iters, 'build_clean_kv (no_grad, sequential)')
kv = kv_holder['kv']
print(f'  clean K/V holds {kv.tokens} tokens '
      f'({kv.tokens*1536*2*2*30/2**30:.2f} GiB)')

t_rows = shifted_uniform_t(len(starts), 5.0, device=dev)
blocks0 = torch.cat([z0[:, :, s:s + args.block] for s in starts], 0)
z_t, target, _ = add_noise(blocks0, t_rows)
z_full = z0.clone()
for i, s in enumerate(starts):
    z_full[:, :, s:s + args.block] = z_t[i:i + 1]


def batched():
    model.zero_grad(set_to_none=True)
    v = bc.parallel_blocks_forward(model, z_full, t_rows, starts, rope, ctx_emb,
                                   ctx_lens, kv, args.block, grad_checkpoint=True)
    tnnF.mse_loss(v.float(), target.float()).backward()


def sequential():
    model.zero_grad(set_to_none=True)
    loss = 0.
    for i, s in enumerate(starts):
        v = bc.block_forward(model, z_full[:, :, s:s + args.block],
                             float(t_rows[i]), s, rope, ctx_emb, ctx_lens,
                             kv=kv, prefix_upto=s * S, grad_checkpoint=True)
        loss = loss + tnnF.mse_loss(v.float(), target[i:i + 1].float())
    (loss / len(starts)).backward()


t_b = timed(batched, args.iters, 'batched blocks (masked SDPA) fwd+bwd')
t_s = timed(sequential, args.iters, 'sequential blocks (no mask) fwd+bwd')
print(f'\n  step total, batched   {(t_clean+t_b)*1000:7.0f} ms')
print(f'  step total, sequential{(t_clean+t_s)*1000:7.0f} ms')
print(f'  peak memory {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
