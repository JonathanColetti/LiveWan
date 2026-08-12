"""Stage 2 -- self-forcing DMD: the fix for the quality gap.

The failure Stage 1 cannot address is exposure bias. A model trained to predict
the next block from *clean teacher context* is never shown the context it
actually gets at deployment: its own output, blurred a little, four steps at a
time, compounding. Measured in the prior work here as one-step error 0.057 on
teacher context against a rollout that visibly degrades.

Self-forcing removes the mismatch by construction -- the training sample IS a
rollout through the deployment loop -- and DMD supplies the learning signal
without any ground-truth continuation to regress onto (there is none; the
continuation never existed):

  1. roll the student out through `FewStepStreamer` (no grad), world primed from
     a teacher clip, K blocks x `--steps` denoising steps, clean K/V committed
     after each block, exactly as inference does it
  2. pick one block and one of its denoising steps at random and redo it with
     gradient against the detached prefix
  3. DMD2 gradient on that block: teacher-with-CFG says where the sample should
     have come from, the critic says where it did come from
  4. update the critic (a LoRA on the frozen teacher base) on the rollout

Gradient truncation in step 2 is what makes this affordable: memory is one
block's activations, not the whole rollout's.

Launch:
  torchrun --nproc_per_node=4 scripts/train_dmd.py --init checkpoints/stage1/latest.pt
"""
import os, sys, time, math, json, copy, argparse
import torch
import torch.distributed as dist
import torch.nn.functional as tnnF
from torch.utils.data import DataLoader, DistributedSampler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import WanModel
from safetensors.torch import load_file

from wanstreamer import blockcausal as bc
from wanstreamer import dmd as D
from wanstreamer.core import latent_geometry, make_rope_table
from wanstreamer.data import TeacherLatents, shifted_uniform_t, add_noise
from wanstreamer.lora import inject_lora, lora_enabled, lora_state_dict
from wanstreamer.stream import FewStepStreamer

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='640*368')
ap.add_argument('--data', default=os.path.join(HERE, 'data/teacher'))
ap.add_argument('--prompts', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--init', default=os.path.join(HERE, 'checkpoints/stage1/latest.pt'))
ap.add_argument('--out', default=os.path.join(HERE, 'checkpoints/dmd'))
ap.add_argument('--frames', type=int, default=21)
ap.add_argument('--world', type=int, default=9,
                help='latent frames of teacher world; also the max when '
                     '--world-choices is set')
ap.add_argument('--world-choices', default='3,6,9',
                help='sampled per iteration. A smaller world means a DEEPER '
                     'self-forcing rollout at the same clip length (world 3 -> '
                     '6 blocks, world 9 -> 4), which is where the exposure-bias '
                     'correction actually comes from.')
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--steps-per-block', type=int, default=4)
ap.add_argument('--rope-gap-max', type=int, default=240,
                help='max simulated world->event RoPE distance; 0 disables')
ap.add_argument('--rope-gap-prob', type=float, default=0.7)
ap.add_argument('--iters', type=int, default=4000)
ap.add_argument('--lr-gen', type=float, default=2e-6)
ap.add_argument('--lr-critic', type=float, default=4e-5)
ap.add_argument('--lora-rank', type=int, default=32)
ap.add_argument('--guidance', type=float, default=5.0)
ap.add_argument('--critic-warmup', type=int, default=150)
ap.add_argument('--gen-every', type=int, default=1)
ap.add_argument('--grad-blocks', type=int, default=3,
                help='how many of the rollout blocks get backprop per iteration. '
                     'The score forwards are shared across all of them, so this '
                     'trades generator gradient per iteration against iteration '
                     'count -- and iteration count also buys critic updates and '
                     'fresh rollouts, so it is not simply "more is better".')
ap.add_argument('--critic-every', type=int, default=1)
ap.add_argument('--t-min', type=float, default=0.02)
ap.add_argument('--t-max', type=float, default=0.98)
ap.add_argument('--shift', type=float, default=5.0)
ap.add_argument('--clip-grad', type=float, default=1.0)
ap.add_argument('--ema-decay', type=float, default=0.95)
ap.add_argument('--ema-every', type=int, default=8)
ap.add_argument('--save-every', type=int, default=500)
ap.add_argument('--sample-every', type=int, default=500)
ap.add_argument('--log-every', type=int, default=10)
args = ap.parse_args()

rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
local = int(os.environ.get('LOCAL_RANK', 0))
if world_size > 1:
    dist.init_process_group('nccl')
torch.cuda.set_device(local)
dev = torch.device(f'cuda:{local}')
main = rank == 0


def log(m):
    if main:
        print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


cfg = WAN_CONFIGS['t2v-1.3B']
size = SIZE_CONFIGS[args.size]
h_lat, w_lat, hp, wp = latent_geometry(size[0], size[1])
S = hp * wp
TIME_SCALE = 1000.0
DTYPE = torch.bfloat16
BASE_SD = load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors')


def new_model():
    m = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
    m.load_state_dict(BASE_SD, strict=True)
    return m


# the student
student = new_model()
if args.init and os.path.exists(args.init):
    sd = torch.load(args.init, map_location='cpu')
    student.load_state_dict(sd.get('model', sd), strict=True)
    log(f'student initialised from {args.init} (step {sd.get("step", "?")})')
else:
    log('WARNING: no Stage-1 checkpoint; starting DMD from base weights')
student = student.to(dev).train()

# frozen teacher + LoRA fake score
# One bf16 base serves both roles: LoRA off == the original bidirectional
# teacher (the real score), LoRA on == the trainable critic (the fake score).
base = new_model().to(dev).to(DTYPE).eval().requires_grad_(False)
lora_params, n_lora_mods, n_lora = inject_lora(base, rank=args.lora_rank)
log(f'critic: LoRA rank {args.lora_rank} on {n_lora_mods} modules, '
    f'{n_lora/1e6:.1f}M trainable params on a frozen bf16 base')

opt_g = torch.optim.AdamW([p for p in student.parameters()], lr=args.lr_gen,
                          betas=(0.0, 0.99), weight_decay=0.0, eps=1e-8,
                          foreach=True)
opt_c = torch.optim.AdamW(lora_params, lr=args.lr_critic, betas=(0.0, 0.99),
                          weight_decay=0.0, eps=1e-8, foreach=True)

rope = make_rope_table(student, hp, wp, args.frames + args.rope_gap_max + 8, dev)
rope_b = make_rope_table(base, hp, wp, args.frames + 8, dev)

streamer = FewStepStreamer(student, width=size[0], height=size[1],
                           max_frames=args.frames + args.rope_gap_max + 8,
                           device=dev, dtype=DTYPE,
                           window_frames=None, time_scale=TIME_SCALE,
                           cache_frames=args.frames + 1, block_frames=args.block,
                           num_steps=args.steps_per_block, rope=rope)

ds = TeacherLatents(args.data, args.prompts, frames=args.frames)
sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if world_size > 1 else None
dl = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=(sampler is None),
                num_workers=2, pin_memory=True, drop_last=True,
                persistent_workers=True)
neg_t5 = ds.neg[0].float().to(dev).unsqueeze(0)
WORLDS = [int(x) for x in args.world_choices.split(',')] if args.world_choices \
    else [args.world]
log(f'{len(ds)} clips | {args.frames} frames | world {WORLDS} -> '
    f'{[(args.frames - w) // args.block for w in WORLDS]} blocks x {args.block} '
    f'| {args.steps_per_block} denoising steps | guidance {args.guidance} '
    f'| world_size {world_size}')

os.makedirs(args.out, exist_ok=True)
# EMA lives on the host: a second 5.7 GB fp32 copy on-device would not fit
# alongside the student's AdamW state, the frozen base and the K/V cache.
# Updated every `--ema-every` steps so the 5.7 GB device->host copy is amortised.
ema = {k: v.detach().to('cpu', torch.float32).clone()
       for k, v in student.state_dict().items() if v.is_floating_point()}


def allreduce_(params):
    if world_size == 1:
        return
    hs = []
    for p in params:
        if p.grad is not None:
            hs.append(dist.all_reduce(p.grad, async_op=True))
    for h in hs:
        h.wait()
    for p in params:
        if p.grad is not None:
            p.grad /= world_size


def sample_t(n=1):
    t = shifted_uniform_t(n, args.shift, device=dev)
    return t.clamp(args.t_min, args.t_max)


gen_params = list(student.parameters())
hist, it_dl = [], iter(dl)
t_start = time.time()
ema_stats = {}
phase = {}
rng = torch.Generator(device=dev).manual_seed(777 + rank)


def tick(name, t0):
    torch.cuda.synchronize()
    t = time.time()
    phase[name] = 0.9 * phase.get(name, t - t0) + 0.1 * (t - t0)
    return t

for step in range(1, args.iters + 1):
    try:
        lat, ctx, pidx = next(it_dl)
    except StopIteration:
        if sampler:
            sampler.set_epoch(step)
        it_dl = iter(dl)
        lat, ctx, pidx = next(it_dl)
    z_teacher = lat.to(dev, non_blocking=True)
    ctx_t5 = ctx.to(dev, non_blocking=True)

    with torch.no_grad():
        ctx_s = student.text_embedding(ctx_t5)
        ctx_pos_b = base.text_embedding(ctx_t5.to(DTYPE))
        ctx_neg_b = base.text_embedding(neg_t5.to(DTYPE))
    ctx_lens = torch.tensor([ctx_t5.shape[1]], device=dev, dtype=torch.long)
    streamer.set_text(ctx_s, ctx_lens)

    # self-rollout
    t0 = time.time()
    n_world = WORLDS[int(torch.randint(len(WORLDS), (1,), device=dev,
                                       generator=rng).item())]
    n_blocks = (args.frames - n_world) // args.block
    streamer.rope_gap = 0
    if args.rope_gap_max and torch.rand(1, device=dev, generator=rng).item() \
            < args.rope_gap_prob:
        streamer.rope_gap = int(torch.randint(1, args.rope_gap_max + 1, (1,),
                                              device=dev, generator=rng).item())
    with torch.no_grad():
        streamer.set_world(z_teacher[:, :, :n_world])
        blocks, recs = streamer.rollout_record(n_blocks, generator=rng)
    x0_full = torch.cat([z_teacher[:, :, :n_world]] +
                        [b.detach() for b in blocks], dim=2)
    t0 = tick('rollout', t0)

    stats = {}
    # generator DMD update
    if step > args.critic_warmup and step % args.gen_every == 0:
        opt_g.zero_grad(set_to_none=True)
        # Score the rollout ONCE, then apply the DMD gradient to every block.
        # The teacher (with CFG) and critic forwards run over the whole clip and
        # dominate the iteration cost; taking the gradient on one block would
        # throw away n_blocks-1 of what they already computed. Backprop goes
        # through each block's FINAL denoising step, which is Self-Forcing's
        # gradient truncation and is exact here: recomputing the last step
        # reproduces the rollout's own output, so the sample the score networks
        # see is the sample the gradient is taken at.
        t_d = sample_t()
        z_t_c, _, _ = add_noise(x0_full, t_d)
        t0 = tick('gen_noise', t0)
        with torch.no_grad():
            with lora_enabled(base, False):
                v_real = D.cfg_velocity(base, z_t_c.to(DTYPE), float(t_d), rope_b,
                                        ctx_pos_b, ctx_neg_b, ctx_lens, DTYPE,
                                        args.guidance, TIME_SCALE).float()
            t0 = tick('teacher_cfg', t0)
            with lora_enabled(base, True):
                v_fake = D.bidirectional_velocity(base, z_t_c.to(DTYPE),
                                                  float(t_d), rope_b, ctx_pos_b,
                                                  ctx_lens, DTYPE,
                                                  TIME_SCALE).float()
            t0 = tick('critic_eval', t0)
            grad_all, gstats = D.dmd_gradient(x0_full, t_d, z_t_c, v_real, v_fake)

        prefix_kv = bc.BufferPrefixKV(streamer.cache)
        kis = torch.randperm(n_blocks, generator=rng, device=dev)[
            :min(args.grad_blocks, n_blocks)].tolist()
        gsum = 0.0
        for ki in kis:
            s = n_world + ki * args.block
            x0_hat = streamer.recompute_step(recs[ki][-1], prefix_kv)
            loss_k = D.dmd_surrogate(x0_hat, grad_all[:, :, s:s + args.block],
                                     scale=1.0 / len(kis))
            loss_k.backward()      # per block, so only one graph is ever live
            gsum += float(loss_k.detach())
            del x0_hat, loss_k
        allreduce_(gen_params)
        gn_g = torch.nn.utils.clip_grad_norm_(gen_params, args.clip_grad)
        opt_g.step()
        stats.update(gstats)
        stats['loss_gen'] = gsum
        stats['gn_gen'] = float(gn_g)
        stats['n_blocks'] = n_blocks
        del z_t_c, v_real, v_fake, grad_all
        t0 = tick('gen_update', t0)

    # critic update
    if step % args.critic_every == 0:
        opt_c.zero_grad(set_to_none=True)
        t_c = sample_t()
        with lora_enabled(base, True):
            loss_c = D.critic_loss(base, x0_full.to(DTYPE), t_c, rope_b,
                                   ctx_pos_b, ctx_lens, DTYPE, TIME_SCALE,
                                   grad_checkpoint=True)
        loss_c.backward()
        allreduce_(lora_params)
        gn_c = torch.nn.utils.clip_grad_norm_(lora_params, args.clip_grad)
        opt_c.step()
        stats['loss_critic'] = float(loss_c.detach())
        stats['gn_critic'] = float(gn_c)
        t0 = tick('critic_update', t0)

    # The rollout is already computed, so dumping it costs nothing and gives a
    # visual record of what the student was actually producing at each step.
    if main and step % args.sample_every == 0:
        os.makedirs(os.path.join(args.out, 'samples'), exist_ok=True)
        torch.save({'latents': x0_full[0].to(torch.float16).cpu(),
                    'world_frames': n_world, 'step': step,
                    'prompt': ds.prompts[int(pidx)]},
                   os.path.join(args.out, f'samples/it{step:06d}.pt'))
    del blocks, recs, x0_full

    # EMA
    if step % args.ema_every == 0:
        with torch.no_grad():
            sd = student.state_dict()
            for k, v in ema.items():
                v.mul_(args.ema_decay).add_(sd[k].detach().to('cpu', torch.float32),
                                            alpha=1 - args.ema_decay)

    for k, v in stats.items():
        ema_stats[k] = v if k not in ema_stats else 0.9 * ema_stats[k] + 0.1 * v
    if step % args.log_every == 0:
        el = time.time() - t_start
        log(f'it {step}/{args.iters} '
            f'gen {ema_stats.get("loss_gen", float("nan")):.4f} '
            f'critic {ema_stats.get("loss_critic", float("nan")):.4f} '
            f'|grad| {ema_stats.get("dmd_grad_norm", float("nan")):.1f} '
            f'norm {ema_stats.get("dmd_normalizer", float("nan")):.3f} '
            f'gn_g {ema_stats.get("gn_gen", float("nan")):.2f} | '
            + ' '.join(f'{k}={v:.1f}' for k, v in phase.items())
            + f' | {el/step:.1f}s/it '
            f'eta {(args.iters-step)*el/step/3600:.1f}h '
            f'mem {torch.cuda.max_memory_allocated()/2**30:.1f}G')
        hist.append({'step': step, **{k: float(v) for k, v in ema_stats.items()}})

    if main and (step % args.save_every == 0 or step == args.iters):
        torch.save({'model': student.state_dict(), 'ema': ema, 'step': step,
                    'args': vars(args)}, os.path.join(args.out, 'latest.pt'))
        torch.save({'model': student.state_dict(), 'ema': ema, 'step': step,
                    'args': vars(args)},
                   os.path.join(args.out, f'step{step:06d}.pt'))
        torch.save({'lora': lora_state_dict(base), 'step': step},
                   os.path.join(args.out, 'critic_latest.pt'))
        with open(os.path.join(args.out, 'history.json'), 'w') as f:
            json.dump(hist, f, indent=1)
        log(f'saved step {step}')

log(f'done in {(time.time()-t_start)/3600:.2f} h')
if world_size > 1:
    dist.destroy_process_group()
