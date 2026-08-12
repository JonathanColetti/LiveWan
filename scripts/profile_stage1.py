"""Attribute the Stage-1 step time. The microbenchmark says ~3.5 s; the trainer
measures ~10 s. Find the difference before spending GPU-hours on it."""
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
ap.add_argument('--world', type=int, default=6)
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--iters', type=int, default=4)
ap.add_argument('--no-opt', action='store_true')
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
opt = None if args.no_opt else torch.optim.AdamW(
    model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=1e-4,
    foreach=True)

F = args.frames
z0 = torch.randn(1, 16, F, h_lat, w_lat, device=dev)
ctx_t5 = torch.randn(1, 512, 4096, device=dev) * 0.4
ctx_lens = torch.tensor([512], device=dev, dtype=torch.long)
rope = make_rope_table(model, hp, wp, F + 8, dev)
starts = bc.block_starts(F, args.world, args.block)
print(f'world={args.world} starts={starts}  optimizer={"no" if args.no_opt else "yes"}')
print(f'alloc conf: {os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)")}')

acc = {}


def mark(name, t0):
    torch.cuda.synchronize()
    t = time.perf_counter()
    acc[name] = acc.get(name, 0.) + (t - t0)
    return t


for it in range(args.iters + 1):
    if it == 1:
        acc.clear()                                # discard warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if opt:
        opt.zero_grad(set_to_none=True)
    t0 = mark('zero_grad', t0)

    with torch.amp.autocast('cuda', enabled=False):
        ctx_emb = model.text_embedding(ctx_t5)
    t0 = mark('text_embed', t0)

    kv = bc.build_clean_kv(model, z0, rope, ctx_emb, ctx_lens, args.world,
                           args.block)
    t0 = mark('build_clean_kv', t0)

    t_rows = shifted_uniform_t(len(starts), 5.0, device=dev)
    blocks0 = torch.cat([z0[:, :, s:s + args.block] for s in starts], 0)
    z_t, target, _ = add_noise(blocks0, t_rows)
    z_full = z0.clone()
    for i, s in enumerate(starts):
        z_full[:, :, s:s + args.block] = z_t[i:i + 1]
    t0 = mark('noise', t0)

    outs = [bc.block_forward(model, z_full[:, :, s:s + args.block],
                             t_rows[i:i + 1], s, rope, ctx_emb, ctx_lens, kv=kv,
                             prefix_upto=s * S, grad_checkpoint=True)[0]
            for i, s in enumerate(starts)]
    v = torch.stack(outs)
    loss = tnnF.mse_loss(v.float(), target.float())
    t0 = mark('forward', t0)

    loss.backward()
    del kv, outs, v
    t0 = mark('backward', t0)

    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    t0 = mark('clip_grad', t0)
    if opt:
        opt.step()
    t0 = mark('opt_step', t0)

tot = sum(acc.values()) / args.iters
for k, v in acc.items():
    print(f'  {k:16s} {v/args.iters*1000:8.1f} ms  ({v/sum(acc.values())*100:4.1f}%)')
print(f'  {"TOTAL":16s} {tot*1000:8.1f} ms')
print(f'  peak alloc {torch.cuda.max_memory_allocated()/2**30:.1f} GiB  '
      f'reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB')
