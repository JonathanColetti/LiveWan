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

ROLLING K/V (--roll-depth). The above rolls a fresh clip every iteration, so the
student is only ever self-forced 4-6 blocks deep while deployment runs it 100
blocks deep -- a ~20x extrapolation, and measurably the largest remaining
structural gap (sharpness decay 0.84). Longer clips cannot close it: teacher and
critic attention is O(L^2) and the K/V cache doubles, which OOMs a 40 GB card.

`--roll-depth D` closes it without either. The streamer's cache is simply not
reset between iterations: iteration k continues the stream iteration k-1 left
off, generating `--roll-blocks` more blocks against the K/V that is already
there, until the roll reaches D blocks and a fresh clip starts. Cost per
iteration is unchanged -- the same number of new blocks, and a bounded event
window (`--roll-window`, matching deployment's) keeps the cache flat -- but the
context those blocks are conditioned on is now arbitrarily deep, and it is the
student's own output all the way down.

Two things make this correct rather than merely cheap:

  * eviction happens ONLY at an iteration boundary. `BufferPrefixKV` redoes a
    recorded denoising step against `buffer[:prefix]`, which is that step's real
    prefix only while nothing has been evicted since. So the streamer's
    automatic per-unit eviction is off during training and `trim_to_window` is
    called between iterations, where no recorded step is still owed a prefix.
    `prefix_intact` asserts it rather than trusting it.
  * `rope_gap` is drawn once per roll, not once per iteration. It shifts event
    frames in RoPE index space; changing it mid-roll would reinterpret K/V that
    was already committed at the old offset.

The clip handed to the teacher and critic is then the last `--frames` latent
frames of the stream rather than world+blocks. Deep into a roll that content is
entirely student-generated, which is what DMD wants to score anyway.

Launch:
  torchrun --nproc_per_node=4 scripts/train_dmd.py --init checkpoints/stage1/latest.pt
"""
import os, sys, time, math, json, copy, glob, argparse
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
from wanstreamer.stream import FewStepStreamer, few_step_schedule

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='640*368')
ap.add_argument('--data', default=os.path.join(HERE, 'data/teacher'))
ap.add_argument('--prompts', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--teacher-ckpt', default='',
                help='weights for the real score / critic base. Defaults to '
                     '--ckpt, i.e. the student\'s own base. Point it at '
                     'checkpoints/wan21_14b to distil from the 14B teacher, '
                     'which is what the published recipe (arXiv:2606.25473) '
                     'does -- the student stays 1.3B, so inference cost is '
                     'unchanged and only the reference distribution improves. '
                     'Needs --fsdp on 40 GB cards: 14B bf16 is 28 GiB.')
ap.add_argument('--teacher-size', default='1.3B', choices=['1.3B', '14B'])
ap.add_argument('--init', default=os.path.join(HERE, 'checkpoints/stage1/latest.pt'))
ap.add_argument('--init-critic', default='',
                help='resume the LoRA fake score from a critic_latest.pt. '
                     'Continuing a DMD run without it restarts the critic at '
                     'zero, and the generator then spends --critic-warmup '
                     'iterations being scored by a critic that says nothing.')
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
ap.add_argument('--steps-cycle', default='',
                help='e.g. "1,2,3,4": cycle the number of denoising steps per '
                     'iteration instead of fixing it at --steps-per-block. '
                     'DMD only backprops through each block\'s FINAL step, so '
                     'a fixed step count supervises only ONE denoising '
                     'interval -- with 4 steps, only t=0.25->0. Cycling makes '
                     'every interval take a turn as the differentiable one. '
                     'Empty = original fixed-schedule behaviour.')
ap.add_argument('--rope-gap-max', type=int, default=240,
                help='max simulated world->event RoPE distance; 0 disables')
ap.add_argument('--rope-gap-prob', type=float, default=0.7)
ap.add_argument('--roll-depth', type=int, default=0,
                help='blocks of continuous self-forcing before a fresh clip. '
                     '0 = the original behaviour (reset every iteration, so '
                     'depth == one iteration). Raising it is the fix for the '
                     'trained-4-6 / deployed-100 rollout mismatch; it costs '
                     'prompt diversity, since one clip now spans '
                     'roll-depth/roll-blocks iterations.')
ap.add_argument('--roll-blocks', type=int, default=4,
                help='blocks generated per iteration while rolling. Only used '
                     'when --roll-depth > 0; otherwise the clip length decides.')
ap.add_argument('--roll-window', type=int, default=6,
                help='event K/V window in latent frames during a roll, evicted '
                     'at iteration boundaries. Matches the deployment window, '
                     'which is what keeps per-iteration cost flat in depth.')
ap.add_argument('--accum', type=int, default=1,
                help='gradient accumulation micro-steps. Effective batch is '
                     'world_size * accum clips; the DMD run this continues used '
                     '4, against Self-Forcing and Causal-rCM 64. NOTE the '
                     'interaction with --roll-depth: while rolling, the micro-'
                     'steps of one optimizer step continue the SAME stream '
                     'rather than drawing new clips, so they raise depth per '
                     'step but not clip diversity. Combine them only '
                     'deliberately; to isolate batch size, roll-depth 0.')
ap.add_argument('--fsdp', action='store_true',
                help='shard the student with FSDP2 instead of replicating it '
                     'and all-reducing by hand. Required for any base larger '
                     'than 1.3B; see wanstreamer/fsdp.py.')
ap.add_argument('--fsdp-reshard', action='store_true',
                help='reshard parameters after every forward. Saves memory, '
                     'but the rollout runs the student dozens of times per '
                     'iteration and pays 30 all-gathers each time.')
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
ap.add_argument('--resume', action='store_true',
                help='continue an interrupted run from --out: model, EMA, LoRA '
                     'critic, BOTH AdamW states and the step counter. Without '
                     'it, restarting a crashed run reloads weights but throws '
                     'away the optimizer moments and restarts the step count, '
                     'so the generator sits out --critic-warmup again and '
                     'AdamW re-estimates its moments from scratch. On a '
                     'multi-day run that is the difference between losing '
                     'minutes and losing the shape of the optimizer state.')
ap.add_argument('--no-save-opt', action='store_true',
                help='skip writing the per-rank optimizer shards that --resume '
                     'needs. They are ~2x the model size per rank.')
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


TEACHER_DIMS = {'1.3B': dict(dim=1536, ffn_dim=8960, num_heads=12, num_layers=30),
                '14B': dict(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40)}


def load_sharded(d):
    """A repo's DiT weights, whether it ships one safetensors file or six."""
    one = os.path.join(d, 'diffusion_pytorch_model.safetensors')
    if os.path.exists(one):
        return load_file(one)
    sd = {}
    for f in sorted(glob.glob(os.path.join(
            d, 'diffusion_pytorch_model-*-of-*.safetensors'))):
        sd.update(load_file(f))
    if not sd:
        raise SystemExit(f'no diffusion_pytorch_model*.safetensors under {d}')
    return sd


def new_model():
    m = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
    m.load_state_dict(BASE_SD, strict=True)
    return m


def new_teacher():
    """The real score and the critic's base. Same architecture family as the
    student -- head_dim is 128 at both 1.3B and 14B, so the RoPE tables, the
    block-causal forward and the LoRA targets all carry over unchanged."""
    if not args.teacher_ckpt or args.teacher_ckpt == args.ckpt:
        return new_model()
    m = WanModel(freq_dim=256, window_size=(-1, -1), qk_norm=True,
                 cross_attn_norm=True, eps=1e-6, text_len=512,
                 **TEACHER_DIMS[args.teacher_size])
    m.load_state_dict(load_sharded(args.teacher_ckpt), strict=True)
    log(f'teacher: {args.teacher_size} from {args.teacher_ckpt} '
        f'({sum(p.numel() for p in m.parameters())/1e9:.2f}B params)')
    return m


# resuming
# Point --init/--init-critic at this run's own last checkpoint, so everything
# downstream is the ordinary load path and there is no second way to build the
# model. The optimizer moments and the step counter are picked up later, where
# the optimizers exist.
RESUME_OPT = os.path.join(args.out, f'opt_rank{rank:03d}.pt')
if args.resume:
    ck = os.path.join(args.out, 'latest.pt')
    if not os.path.exists(ck):
        raise SystemExit(f'--resume: nothing to resume from at {ck}')
    if not os.path.exists(RESUME_OPT):
        raise SystemExit(
            f'--resume: {RESUME_OPT} is missing. Optimizer shards are written '
            f'per rank, so a resume needs the SAME world size the run was '
            f'saved at (and --no-save-opt must not have been set).')
    args.init = ck
    args.init_critic = os.path.join(args.out, 'critic_latest.pt')
    log(f'resuming from {args.out}')

# the student
student = new_model()
if args.init and os.path.exists(args.init):
    sd = torch.load(args.init, map_location='cpu')
    student.load_state_dict(sd.get('model', sd), strict=True)
    log(f'student initialised from {args.init} (step {sd.get("step", "?")})')
else:
    log('WARNING: no Stage-1 checkpoint; starting DMD from base weights')

if args.fsdp:
    if world_size == 1:
        raise SystemExit('--fsdp needs world_size > 1')
    from wanstreamer.fsdp import shard_model, full_state_dict, set_grad_sync
    # Shard on the host, then move: each rank only ever materialises its own
    # 1/N of the parameters on the GPU.
    shard_model(student, reshard_after_forward=args.fsdp_reshard)
    log(f'FSDP2: student sharded over {world_size} ranks '
        f'(reshard_after_forward={args.fsdp_reshard})')
student = student.to(dev).train()

# frozen teacher + LoRA fake score
# At 1.3B one bf16 base serves both roles: LoRA off == the original
# bidirectional teacher (the real score), LoRA on == the trainable critic (the
# fake score). That is right when teacher and student are the same size.
#
# With a 14B teacher it is wrong, and measurably so. The two roles do different
# jobs: the REAL score defines the target distribution, which is where a bigger
# teacher earns its keep; the FAKE score only has to track the *student's* own
# output distribution, and the student is 1.3B. Sharing one base silently makes
# the critic 14B too. Measured at 69% of a 55.9 s iteration (critic_update
# 29.7 s + critic_eval 8.8 s) for capacity it has no use for. So when the
# teacher is larger than the student, the critic gets its own 1.3B base.
SPLIT_CRITIC = bool(args.teacher_ckpt) and args.teacher_size != '1.3B'

teacher = new_teacher().to(DTYPE).eval().requires_grad_(False)
if SPLIT_CRITIC:
    critic = new_model().to(DTYPE).eval().requires_grad_(False)
    lora_params, n_lora_mods, n_lora = inject_lora(critic, rank=args.lora_rank)
    if args.fsdp:
        from wanstreamer.fsdp import shard_model as _shard
        _shard(teacher, reshard_after_forward=True)
        log(f'FSDP2: 14B teacher sharded over {world_size} ranks (frozen, bf16)')
    log(f'critic: separate 1.3B base (teacher is {args.teacher_size}), '
        f'LoRA rank {args.lora_rank} on {n_lora_mods} modules, '
        f'{n_lora/1e6:.1f}M trainable params')
else:
    critic = teacher
    lora_params, n_lora_mods, n_lora = inject_lora(critic, rank=args.lora_rank)
    log(f'critic: LoRA rank {args.lora_rank} on {n_lora_mods} modules, '
        f'{n_lora/1e6:.1f}M trainable params on the shared frozen bf16 base')
teacher = teacher.to(dev)
critic = critic.to(dev)
# `base` is kept as the name the critic path uses throughout; when they are not
# split it is the same object as `teacher`, exactly as before.
base = critic
if args.init_critic:
    cs = torch.load(args.init_critic, map_location='cpu')
    cs = cs.get('lora', cs)
    got = critic.load_state_dict(cs, strict=False)
    loaded = len(cs) - len(got.unexpected_keys)
    if loaded != len(cs):
        raise SystemExit(f'critic resume matched only {loaded}/{len(cs)} '
                         f'tensors from {args.init_critic}')
    log(f'critic resumed from {args.init_critic} ({loaded} tensors, '
        f'step {torch.load(args.init_critic, map_location="cpu").get("step","?")})')

opt_g = torch.optim.AdamW([p for p in student.parameters()], lr=args.lr_gen,
                          betas=(0.0, 0.99), weight_decay=0.0, eps=1e-8,
                          foreach=True)
opt_c = torch.optim.AdamW(lora_params, lr=args.lr_critic, betas=(0.0, 0.99),
                          weight_decay=0.0, eps=1e-8, foreach=True)

# Filled in below when resuming; `start_step` is the last COMPLETED step, so the
# loop begins at start_step + 1 and a resumed run reaches --iters in total
# rather than running --iters more.
start_step = 0
resume_blob = None
if args.resume:
    # Loaded straight onto this rank's device. Under FSDP the moments are
    # DTensors over the same 1-D mesh the run was saved on, which is why a
    # resume is pinned to the world size that wrote them.
    resume_blob = torch.load(RESUME_OPT, map_location=dev, weights_only=False)
    opt_g.load_state_dict(resume_blob['opt_g'])
    opt_c.load_state_dict(resume_blob['opt_c'])
    start_step = int(resume_blob['step'])
    log(f'optimizer state restored from {RESUME_OPT}; continuing at step '
        f'{start_step + 1}/{args.iters}')

WORLD_MAX = max([int(x) for x in args.world_choices.split(',')]
                if args.world_choices else [args.world])
ROLLING = args.roll_depth > 0

# A roll walks the stream forward, so the temporal RoPE index it reaches is the
# world plus the whole roll, plus whatever gap augmentation is stacked on top.
# Deployment reaches 9 + 100*3 = 309, so this is also the axis the table has to
# cover for the deployed configuration to be in distribution at all.
ROPE_MAX = (WORLD_MAX + (args.roll_depth * args.block if ROLLING else args.frames)
            + args.rope_gap_max + 8)
# The cache holds the pinned world, the retained event window, and the blocks
# this iteration is about to add before the next boundary trims it back.
CACHE_FRAMES = (WORLD_MAX + args.roll_window + args.roll_blocks * args.block + 2
                if ROLLING else args.frames + 1)

rope = make_rope_table(student, hp, wp, ROPE_MAX, dev)
# head_dim is 128 at 1.3B, 5B and 14B, so the RoPE table is identical for
# the teacher and the critic even when they are different sizes.
rope_b = make_rope_table(critic, hp, wp, args.frames + 8, dev)

streamer = FewStepStreamer(student, width=size[0], height=size[1],
                           max_frames=ROPE_MAX,
                           device=dev, dtype=DTYPE,
                           # None: eviction is driven by hand from the training
                           # loop at iteration boundaries, never mid-rollout --
                           # see the module docstring and stream.trim_to_window.
                           window_frames=None, time_scale=TIME_SCALE,
                           cache_frames=CACHE_FRAMES, block_frames=args.block,
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
    f'| world_size {world_size} | accum {args.accum} '
    f'| effective batch {world_size * args.accum}')
STEPS_CYCLE = [int(x) for x in args.steps_cycle.split(',')] if args.steps_cycle \
    else []
if STEPS_CYCLE:
    log(f'step cycling: {STEPS_CYCLE} -- every denoising interval takes a turn '
        f'as the differentiable one')
if ROLLING:
    log(f'rolling K/V: depth {args.roll_depth} blocks '
        f'({args.roll_depth * args.block} latent frames, '
        f'{args.roll_depth * args.block * 4 / 25:.1f}s of video) '
        f'| {args.roll_blocks} blocks/iter -> '
        f'{max(1, args.roll_depth // args.roll_blocks)} iters per roll '
        f'| window {args.roll_window} | cache {CACHE_FRAMES} frames '
        f'({streamer.cache.memory_bytes()/2**30:.1f} GiB) | rope table {ROPE_MAX}')

os.makedirs(args.out, exist_ok=True)
# EMA lives on the host: a second 5.7 GB fp32 copy on-device would not fit
# alongside the student's AdamW state, the frozen base and the K/V cache.
# Updated every `--ema-every` steps so the 5.7 GB device->host copy is amortised.
# Under FSDP it stays on-device instead: each rank holds 1/N of it, which does
# fit, and the EMA recursion is elementwise so the EMA of a shard is the shard
# of the EMA. It is gathered only when a checkpoint is written.
ema = {k: (v.detach().clone().float() if args.fsdp
           else v.detach().to('cpu', torch.float32).clone())
       for k, v in student.state_dict().items() if v.is_floating_point()}
if resume_blob is not None:
    # The EMA in latest.pt is gathered and full; the per-rank shard file has it
    # in the layout this process actually accumulates into, so restore from
    # there and keep the two representations from having to agree.
    for k, v in resume_blob['ema'].items():
        ema[k].copy_(v)
    log(f'EMA restored ({len(ema)} tensors)')
    del resume_blob


def gathered(sd):
    """Full unsharded tensors on the host. Collective -- every rank must call
    it, even the ones that will not write the file."""
    if not args.fsdp:
        return sd
    return {k: (v.full_tensor() if hasattr(v, 'full_tensor') else v).cpu()
            for k, v in sd.items()}


def allreduce_(params, sharded=False):
    """Average gradients across ranks by hand.

    `sharded=True` means FSDP already reduce-scattered them inside the backward,
    so doing it again would both double-count and operate on DTensors. The LoRA
    critic is NOT sharded even under --fsdp -- it is 32 M parameters on a frozen
    bf16 base, so it stays replicated and still needs this.
    """
    if world_size == 1 or (sharded and args.fsdp):
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
if args.resume and main:
    # history.json is rewritten whole on every save, so carry the earlier curve
    # forward or the resumed run silently truncates it to its own segment.
    hpath = os.path.join(args.out, 'history.json')
    if os.path.exists(hpath):
        with open(hpath) as f:
            hist = [h for h in json.load(f) if h.get('step', 0) <= start_step]
        log(f'history: kept {len(hist)} rows up to step {start_step}')
t_start = time.time()
ema_stats = {}
phase = {}
# Offset by the resumed step so a restarted run does not replay the same
# world/gap/prompt draws it already trained on.
rng = torch.Generator(device=dev).manual_seed(777 + rank + 100003 * start_step)


def tick(name, t0):
    torch.cuda.synchronize()
    t = time.time()
    phase[name] = 0.9 * phase.get(name, t - t0) + 0.1 * (t - t0)
    return t

with torch.no_grad():
    ctx_neg_t = teacher.text_embedding(neg_t5.to(DTYPE))


def next_batch():
    global it_dl
    try:
        return next(it_dl)
    except StopIteration:
        if sampler:
            sampler.set_epoch(step)
        it_dl = iter(dl)
        return next(it_dl)


class Roll:
    """One continuous self-forcing rollout: the streamer's K/V, plus the clean
    stream latents the teacher and critic will be shown.

    `depth` starts past the cap so the first iteration always begins a roll.
    """
    depth = 1 << 30
    lat = None            # [1, 16, F, H, W] clean stream latents, most recent first F
    n_world = 0
    ctx_pos_t = None      # text embedding in the TEACHER's width
    ctx_pos_c = None      # ... and in the CRITIC's, identical when not split
    ctx_lens = None
    pidx = 0


roll = Roll()


def start_roll():
    """Fresh clip, fresh world, fresh RoPE gap. The gap is drawn HERE and not
    per iteration: it offsets event frames in RoPE index space, and K/V already
    committed under one offset cannot be reinterpreted under another."""
    lat, ctx, pidx = next_batch()
    z_teacher = lat.to(dev, non_blocking=True)
    ctx_t5 = ctx.to(dev, non_blocking=True)
    with torch.no_grad():
        ctx_s = student.text_embedding(ctx_t5)
        roll.ctx_pos_t = teacher.text_embedding(ctx_t5.to(DTYPE))
        roll.ctx_pos_c = roll.ctx_pos_t if critic is teacher else \
            critic.text_embedding(ctx_t5.to(DTYPE))
    roll.ctx_lens = torch.tensor([ctx_t5.shape[1]], device=dev, dtype=torch.long)
    roll.pidx = int(pidx)
    streamer.set_text(ctx_s, roll.ctx_lens)
    roll.n_world = WORLDS[int(torch.randint(len(WORLDS), (1,), device=dev,
                                            generator=rng).item())]
    streamer.rope_gap = 0
    if args.rope_gap_max and torch.rand(1, device=dev, generator=rng).item() \
            < args.rope_gap_prob:
        streamer.rope_gap = int(torch.randint(1, args.rope_gap_max + 1, (1,),
                                              device=dev, generator=rng).item())
    with torch.no_grad():
        streamer.set_world(z_teacher[:, :, :roll.n_world])
    roll.lat = z_teacher[:, :, :roll.n_world].clone()
    roll.depth = 0


def advance_roll():
    """Generate this iteration's blocks. Returns (recs, x0_full, base_off).

    `base_off` is where the NEW blocks start inside `x0_full` -- the only frames
    the generator gradient may be applied to, since they are the only ones with
    recorded denoising steps.
    """
    if not ROLLING:
        start_roll()
        n_new = (args.frames - roll.n_world) // args.block
    else:
        if roll.depth >= args.roll_depth:
            start_roll()
        else:
            # The one point in the cycle where evicting is safe: the previous
            # iteration's recorded steps have all been consumed.
            streamer.trim_to_window(args.roll_window)
        n_new = args.roll_blocks

    if STEPS_CYCLE:
        n_steps = STEPS_CYCLE[(step - 1) % len(STEPS_CYCLE)]
        streamer.schedule = few_step_schedule(n_steps)
    with torch.no_grad():
        blocks, recs = streamer.rollout_record(n_new, generator=rng)
    roll.depth += n_new
    new = torch.cat([b.detach() for b in blocks], dim=2)
    roll.lat = torch.cat([roll.lat, new], dim=2)[:, :, -args.frames:]
    x0_full = roll.lat
    return recs, x0_full, x0_full.shape[2] - n_new * args.block, n_new


def micro_step(scale, stats):
    """One rollout, one DMD generator gradient, one critic gradient.

    `scale` is 1/accum: both losses are scaled here rather than the gradients
    afterwards, so accumulation is exact for the clipping that follows.
    """
    global phase
    t0 = time.time()
    recs, x0_full, base_off, n_new = advance_roll()
    ctx_pos_t, ctx_pos_c = roll.ctx_pos_t, roll.ctx_pos_c
    ctx_lens = roll.ctx_lens
    t0 = tick('rollout', t0)

    # generator DMD update
    if step > args.critic_warmup and step % args.gen_every == 0:
        t_d = sample_t()
        z_t_c, _, _ = add_noise(x0_full, t_d)
        t0 = tick('gen_noise', t0)
        with torch.no_grad():
            with lora_enabled(teacher, False):
                v_real = D.cfg_velocity(teacher, z_t_c.to(DTYPE), float(t_d),
                                        rope_b, ctx_pos_t, ctx_neg_t, ctx_lens,
                                        DTYPE, args.guidance, TIME_SCALE).float()
            t0 = tick('teacher_cfg', t0)
            with lora_enabled(critic, True):
                v_fake = D.bidirectional_velocity(critic, z_t_c.to(DTYPE),
                                                  float(t_d), rope_b, ctx_pos_c,
                                                  ctx_lens, DTYPE,
                                                  TIME_SCALE).float()
            t0 = tick('critic_eval', t0)
            grad_all, gstats = D.dmd_gradient(x0_full, t_d, z_t_c, v_real, v_fake)

        prefix_kv = bc.BufferPrefixKV(streamer.cache)
        kis = torch.randperm(n_new, generator=rng, device=dev)[
            :min(args.grad_blocks, n_new)].tolist()
        gsum = 0.0
        for ki in kis:
            rec = recs[ki][-1]
            # Refuse to backprop through a step whose prefix the cache has
            # since moved out from under. It would run and produce a number.
            if not streamer.prefix_intact(rec):
                raise RuntimeError(
                    f'block {ki} was recorded before an eviction; its prefix '
                    f'is no longer buffer[:{rec["prefix"]}]. Eviction must '
                    f'only happen at iteration boundaries.')
            s = base_off + ki * args.block
            x0_hat = streamer.recompute_step(rec, prefix_kv)
            loss_k = D.dmd_surrogate(x0_hat, grad_all[:, :, s:s + args.block],
                                     scale=scale / len(kis))
            loss_k.backward()      # per block, so only one graph is ever live
            gsum += float(loss_k.detach())
            del x0_hat, loss_k
        stats.update(gstats)
        # Accumulate rather than overwrite: `gsum` is already scaled by 1/accum,
        # so summing over the micro-steps keeps the logged loss comparable
        # across --accum settings instead of shrinking with it.
        stats['loss_gen'] = stats.get('loss_gen', 0.0) + gsum
        stats['n_blocks'] = n_new
        stats['roll_depth'] = roll.depth
        del z_t_c, v_real, v_fake, grad_all
        t0 = tick('gen_update', t0)

    # critic update
    if step % args.critic_every == 0:
        t_c = sample_t()
        with lora_enabled(critic, True):
            loss_c = D.critic_loss(critic, x0_full.to(DTYPE), t_c, rope_b,
                                   ctx_pos_c, ctx_lens, DTYPE, TIME_SCALE,
                                   grad_checkpoint=True)
        (loss_c * scale).backward()
        stats['loss_critic'] = stats.get('loss_critic', 0.0) + \
            float(loss_c.detach()) * scale
        t0 = tick('critic_update', t0)

    return x0_full


for step in range(start_step + 1, args.iters + 1):
    stats = {}
    opt_g.zero_grad(set_to_none=True)
    opt_c.zero_grad(set_to_none=True)
    for mi in range(args.accum):
        if args.fsdp:
            set_grad_sync(student, mi == args.accum - 1)
        x0_full = micro_step(1.0 / args.accum, stats)

    if step > args.critic_warmup and step % args.gen_every == 0:
        allreduce_(gen_params, sharded=True)
        gn_g = torch.nn.utils.clip_grad_norm_(gen_params, args.clip_grad)
        opt_g.step()
        stats['gn_gen'] = float(gn_g)
    if step % args.critic_every == 0:
        allreduce_(lora_params)
        gn_c = torch.nn.utils.clip_grad_norm_(lora_params, args.clip_grad)
        opt_c.step()
        stats['gn_critic'] = float(gn_c)

    # The rollout is already computed, so dumping it costs nothing and gives a
    # visual record of what the student was actually producing at each step.
    if main and step % args.sample_every == 0:
        os.makedirs(os.path.join(args.out, 'samples'), exist_ok=True)
        torch.save({'latents': x0_full[0].to(torch.float16).cpu(),
                    'world_frames': roll.n_world, 'step': step,
                    'roll_depth': roll.depth,
                    'prompt': ds.prompts[roll.pidx]},
                   os.path.join(args.out, f'samples/it{step:06d}.pt'))
    del x0_full

    # EMA
    if step % args.ema_every == 0:
        with torch.no_grad():
            sd = student.state_dict()
            for k, v in ema.items():
                src = sd[k].detach()
                v.mul_(args.ema_decay).add_(
                    src if args.fsdp else src.to('cpu', torch.float32),
                    alpha=1 - args.ema_decay)

    for k, v in stats.items():
        ema_stats[k] = v if k not in ema_stats else 0.9 * ema_stats[k] + 0.1 * v
    if step % args.log_every == 0:
        el = time.time() - t_start
        # Rate over the steps THIS process ran, not the absolute step number --
        # after a resume those differ and the ETA would be nonsense.
        done_here = step - start_step
        log(f'it {step}/{args.iters} '
            f'gen {ema_stats.get("loss_gen", float("nan")):.4f} '
            f'critic {ema_stats.get("loss_critic", float("nan")):.4f} '
            f'|grad| {ema_stats.get("dmd_grad_norm", float("nan")):.1f} '
            f'norm {ema_stats.get("dmd_normalizer", float("nan")):.3f} '
            f'gn_g {ema_stats.get("gn_gen", float("nan")):.2f} '
            f'depth {roll.depth} | '
            + ' '.join(f'{k}={v:.1f}' for k, v in phase.items())
            + f' | {el/done_here:.1f}s/it '
            f'eta {(args.iters-step)*el/done_here/3600:.1f}h '
            f'mem {torch.cuda.max_memory_allocated()/2**30:.1f}G')
        hist.append({'step': step, **{k: float(v) for k, v in ema_stats.items()}})

    if step % args.save_every == 0 or step == args.iters:
        # Both gathers are collective: every rank has to reach them, only rank
        # 0 writes. Doing this under `if main` would hang the other three.
        msd = full_state_dict(student) if args.fsdp else student.state_dict()
        esd = gathered(ema)
        lsd = lora_state_dict(critic)
        if main:
            blob = {'model': msd, 'ema': esd, 'step': step, 'args': vars(args)}
            torch.save(blob, os.path.join(args.out, 'latest.pt'))
            torch.save(blob, os.path.join(args.out, f'step{step:06d}.pt'))
            torch.save({'lora': lsd, 'step': step},
                       os.path.join(args.out, 'critic_latest.pt'))
            with open(os.path.join(args.out, 'history.json'), 'w') as f:
                json.dump(hist, f, indent=1)
            log(f'saved step {step}')
        del msd, esd, lsd
        if not args.no_save_opt:
            # Per rank, ungathered, because under FSDP the AdamW moments are
            # DTensors on this run's mesh and gathering 2x5.7 GB of them to one
            # host every --save-every is neither cheap nor necessary: a resume
            # runs at the same world size. Written to a temp name and renamed,
            # so a crash mid-write cannot leave a half-file that --resume would
            # happily load.
            tmp = RESUME_OPT + '.tmp'
            torch.save({'opt_g': opt_g.state_dict(), 'opt_c': opt_c.state_dict(),
                        'ema': ema, 'step': step, 'world_size': world_size}, tmp)
            os.replace(tmp, RESUME_OPT)
            if main:
                log(f'saved optimizer shards (rank-local) for --resume')

log(f'done in {(time.time()-t_start)/3600:.2f} h')
if world_size > 1:
    dist.destroy_process_group()
