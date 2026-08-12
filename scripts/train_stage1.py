"""Stage 1 -- block-causal adaptation of Wan2.1-1.3B on teacher ODE samples.

What this stage is for: the base model is bidirectional. Before any distillation
can be asked of it, it has to become a model that predicts a block of latent
frames given a *clean K/V prefix* -- the papers' "KV construction" -- rather
than given the whole clip at once. That is a change of conditioning, not of
capability, so it converges fast and stably; the few-step and exposure-bias
problems are Stage 2's job.

Every training forward is the deployment forward. `build_clean_kv` lays the past
down exactly as streaming inference lays it down (world primed as one
bidirectional block, then each event block re-run at t=0 and committed), and
`parallel_blocks_forward` then trains every event block against the prefix slice
its start index implies. No masking approximation, no teacher-forcing shortcut
that deployment cannot reproduce.

Deliberately NOT done here:
  * no noise on the context -- the prefix is clean at deployment too
  * no causal-probability mixing (the old train_streaming.py ran 70% bidirectional,
    which is why final.pt was only weakly causal)
  * temporal RoPE at true absolute indices, and the [0,1000] timestep convention
    of the original weights, both throughout

Launch:
  torchrun --nproc_per_node=4 scripts/train_stage1.py --steps 6000
"""
import os, sys, time, math, json, argparse
import torch
import torch.distributed as dist
import torch.nn.functional as tnnF
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import WanModel
from safetensors.torch import load_file

from wanstreamer import blockcausal as bc
from wanstreamer.core import latent_geometry, make_rope_table
from wanstreamer.data import TeacherLatents, shifted_uniform_t, add_noise

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='640*368')
ap.add_argument('--data', default=os.path.join(HERE, 'data/teacher'))
ap.add_argument('--prompts', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--init', default='', help='resume/init weights (.pt state dict)')
ap.add_argument('--out', default=os.path.join(HERE, 'checkpoints/stage1'))
ap.add_argument('--frames', type=int, default=21)
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--world-min', type=int, default=3)
ap.add_argument('--world-max', type=int, default=9)
ap.add_argument('--rope-gap-max', type=int, default=240,
                help='max simulated distance (latent frames) between the pinned '
                     'world and the current event window; 0 disables')
ap.add_argument('--rope-gap-prob', type=float, default=0.7)
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--warmup', type=int, default=200)
ap.add_argument('--accum', type=int, default=2)
ap.add_argument('--shift', type=float, default=5.0)
ap.add_argument('--clip-grad', type=float, default=1.0)
ap.add_argument('--save-every', type=int, default=1000)
ap.add_argument('--log-every', type=int, default=20)
ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--fsdp', action='store_true',
                help='shard with FSDP2 instead of replicating with DDP. '
                     'Required for any base larger than 1.3B, where a full '
                     'fp32 copy plus grads plus Adam moments per rank does not '
                     'fit. See wanstreamer/fsdp.py.')
ap.add_argument('--fsdp-reshard', action='store_true',
                help='reshard after forward: less memory, more collectives')
args = ap.parse_args()

# ------------------------------------------------------------------ distributed
rank = int(os.environ.get('RANK', 0))
world = int(os.environ.get('WORLD_SIZE', 1))
local = int(os.environ.get('LOCAL_RANK', 0))
if world > 1:
    dist.init_process_group('nccl')
torch.cuda.set_device(local)
dev = torch.device(f'cuda:{local}')
main = rank == 0


def log(msg):
    if main:
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


cfg = WAN_CONFIGS['t2v-1.3B']
size = SIZE_CONFIGS[args.size]
h_lat, w_lat, hp, wp = latent_geometry(size[0], size[1])
S = hp * wp
TIME_SCALE = 1000.0                      # original-Wan convention, kept throughout

model = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
model.load_state_dict(
    load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'), strict=True)
if args.init:
    sd = torch.load(args.init, map_location='cpu')
    model.load_state_dict(sd.get('model', sd), strict=True)
    log(f'initialised from {args.init}')
if args.fsdp:
    if world == 1:
        raise SystemExit('--fsdp needs world_size > 1')
    from wanstreamer.fsdp import shard_model, full_state_dict
    shard_model(model, reshard_after_forward=args.fsdp_reshard)
    log(f'FSDP2: sharded over {world} ranks '
        f'(reshard_after_forward={args.fsdp_reshard})')
model = model.to(dev)
model.train()


class Stage1Net(torch.nn.Module):
    """The whole training forward, as one module.

    DDP registers its gradient-reduction hooks when *its own* forward runs, so a
    training step that calls the bare WanModel through helper functions would
    silently never all-reduce. Wrapping the entire step -- text embedding, clean
    K/V construction, and the batched block forward -- keeps every parameter on
    the DDP path and lets reduction overlap the backward pass.
    """

    def __init__(self, net, rope, block, time_scale):
        super().__init__()
        self.net = net
        self.rope = rope
        self.block = block
        self.time_scale = time_scale

    def forward(self, z0, ctx_t5, wf, starts, t_rows, z_full, rope_gap=0):
        with torch.amp.autocast('cuda', enabled=False):
            ctx_emb = self.net.text_embedding(ctx_t5)
        ctx_lens = torch.tensor([ctx_t5.shape[1]], device=z0.device,
                                dtype=torch.long)
        kv = bc.build_clean_kv(self.net, z0, self.rope, ctx_emb, ctx_lens, wf,
                               self.block, time_scale=self.time_scale,
                               rope_gap=rope_gap)
        # Blocks run one at a time rather than batched. Batching them needs a
        # key-padding mask (rows have different prefix lengths), which rules out
        # the flash SDPA backend; measured on this machine the masked batched
        # path costs 2949 ms against 2561 ms sequential for the same math
        # (scripts/bench_train_step.py), and sequential is also the literal
        # inference path. Verified equivalent in scripts/verify_blockcausal.py.
        outs = []
        for i, s in enumerate(starts):
            outs.append(bc.block_forward(
                self.net, z_full[:, :, s:s + self.block], t_rows[i:i + 1],
                bc.event_rope_index(s, wf, rope_gap),
                self.rope, ctx_emb, ctx_lens, kv=kv,
                prefix_upto=s * self.rope.seq, time_scale=self.time_scale,
                grad_checkpoint=True)[0])
        del kv
        return torch.stack(outs)

raw = model
rope = make_rope_table(raw, hp, wp, args.frames + args.rope_gap_max + 8, dev)
net = Stage1Net(raw, rope, args.block, TIME_SCALE)
if args.fsdp:
    # No DDP wrapper, and Stage1Net is used only as a plain callable. FSDP2's
    # all-gathers hang off the per-layer shard units that `blockcausal._run`
    # enters, so unlike DDP it does not care which module the step was launched
    # from -- the reason Stage1Net exists in the first place stops applying.
    ddp = net
elif world > 1:
    ddp = DDP(net, device_ids=[local], gradient_as_bucket_view=True,
              broadcast_buffers=False)
else:
    ddp = net

opt = torch.optim.AdamW(raw.parameters(), lr=args.lr, betas=(0.9, 0.95),
                        weight_decay=1e-4, eps=1e-8, foreach=True)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


ds = TeacherLatents(args.data, args.prompts, frames=args.frames)
sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if world > 1 else None
dl = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=(sampler is None),
                num_workers=args.workers, pin_memory=True, drop_last=True,
                persistent_workers=args.workers > 0)
log(f'{len(ds)} teacher clips | {args.frames} latent frames | block {args.block} '
    f'| {S} tokens/frame | world {args.world_min}-{args.world_max}')
log(f'world_size {world} | accum {args.accum} | effective batch '
    f'{world * args.accum} clips/step | lr {args.lr}')

os.makedirs(args.out, exist_ok=True)

step = 0
t_start = time.time()
ema_loss = None
hist = []
gen = torch.Generator(device='cpu').manual_seed(1234 + rank)
it = iter(dl)
while step < args.steps:
    opt.zero_grad(set_to_none=True)
    for micro in range(args.accum):
        try:
            lat, ctx, pidx = next(it)
        except StopIteration:
            if sampler:
                sampler.set_epoch(step)
            it = iter(dl)
            lat, ctx, pidx = next(it)
        z0 = lat.to(dev, non_blocking=True)                     # [1,16,F,H,W]
        ctx_t5 = ctx.to(dev, non_blocking=True)                 # [1,512,4096]

        wf = int(torch.randint(args.world_min, args.world_max + 1, (1,),
                               generator=gen).item())
        starts = bc.block_starts(args.frames, wf, args.block)

        # one noise level per block; every block is trained in one forward
        t_rows = shifted_uniform_t(len(starts), args.shift, device=dev)
        blocks0 = torch.cat([z0[:, :, s:s + args.block] for s in starts], 0)
        z_t, target, _ = add_noise(blocks0, t_rows)
        z_full = z0.clone()
        for i, s in enumerate(starts):
            z_full[:, :, s:s + args.block] = z_t[i:i + 1]

        gap = 0
        if args.rope_gap_max and torch.rand(1, generator=gen).item() < args.rope_gap_prob:
            gap = int(torch.randint(1, args.rope_gap_max + 1, (1,),
                                    generator=gen).item())

        sync = (micro == args.accum - 1) or world == 1
        if args.fsdp:
            # FSDP2's equivalent of DDP.no_sync(): skip the reduce-scatter on
            # every micro-step but the last, so accumulation costs one
            # collective per optimizer step rather than one per micro-step.
            for m in list(raw.causal_layers) + [raw.causal_head]:
                m.set_requires_gradient_sync(sync)
            cm = torch.enable_grad()
        else:
            cm = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
        with cm:
            v = ddp(z0, ctx_t5, wf, starts, t_rows, z_full, gap)
            loss = tnnF.mse_loss(v.float(), target.float()) / args.accum
            loss.backward()

    for g in opt.param_groups:
        g['lr'] = lr_at(step)
    gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.clip_grad)
    opt.step()
    step += 1

    l = float(loss.detach()) * args.accum
    ema_loss = l if ema_loss is None else 0.98 * ema_loss + 0.02 * l
    if step % args.log_every == 0:
        el = time.time() - t_start
        log(f'step {step}/{args.steps} loss {l:.4f} ema {ema_loss:.4f} '
            f'gn {float(gn):.2f} lr {lr_at(step):.2e} '
            f'{el/step:.2f}s/step eta {(args.steps-step)*el/step/60:.0f}m '
            f'mem {torch.cuda.max_memory_allocated()/2**30:.1f}G')
        hist.append({'step': step, 'loss': l, 'ema': ema_loss,
                     'grad_norm': float(gn), 'lr': lr_at(step)})
    if step % args.save_every == 0 or step == args.steps:
        # Collective under FSDP: every rank must reach it, only rank 0 writes.
        msd = full_state_dict(raw) if args.fsdp else raw.state_dict()
        if main:
            p = os.path.join(args.out, f'step{step:06d}.pt')
            blob = {'model': msd, 'step': step, 'args': vars(args)}
            torch.save(blob, p)
            torch.save(blob, os.path.join(args.out, 'latest.pt'))
            with open(os.path.join(args.out, 'history.json'), 'w') as f:
                json.dump(hist, f, indent=1)
            log(f'saved {p}')
        del msd

log(f'done in {(time.time()-t_start)/60:.1f} min')
if world > 1:
    dist.destroy_process_group()
