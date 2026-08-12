"""Correctness gate for the DDP -> FSDP port. Run before trusting any FSDP run.

Sharding can land in the wrong place in this codebase's forward, which is the
entire reason this file exists. `blockcausal._run` does not call a
WanAttentionBlock; it hands the block to `_layer`, which reaches into
`blk.self_attn.q` and friends. FSDP2 installs its all-gather in a pre-forward
hook on the module it shards, so sharding the block leaves that hook unfired --
the same shape of bug DDP already has here (SCALING_PROMPT §8). Check 3 pins
down what that actually does: DTensor rejects the mixed operand and it raises,
rather than quietly training on shards.

So every check below is paired with a deliberately broken control, exactly as
scripts/verify_blockcausal.py does it. A check that cannot fail proves nothing.

  1. wrapping alone is a no-op          CausalBlock/CausalHead == plain path
  2. sharded forward == single-GPU forward
  3. CONTROL: shard without the wrapper does NOT reproduce it
  4. sharded gradients == single-GPU gradients
  5. full_state_dict round-trips into a plain WanModel on one rank

Launch:
  torchrun --nproc_per_node=4 scripts/verify_fsdp.py          # tiny model, seconds
  torchrun --nproc_per_node=4 scripts/verify_fsdp.py --full   # the real 1.3B
"""
import os, sys, argparse, traceback

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel

from wanstreamer import blockcausal as bc
from wanstreamer.core import make_rope_table
from wanstreamer.fsdp import attach_causal_modules, shard_model, full_state_dict

ap = argparse.ArgumentParser()
ap.add_argument('--full', action='store_true', help='use the real 1.3B config')
ap.add_argument('--frames', type=int, default=3)
ap.add_argument('--tol', type=float, default=2e-3)
args = ap.parse_args()

rank = int(os.environ.get('RANK', 0))
world = int(os.environ.get('WORLD_SIZE', 1))
local = int(os.environ.get('LOCAL_RANK', 0))
dist.init_process_group('nccl')
torch.cuda.set_device(local)
dev = torch.device(f'cuda:{local}')
main = rank == 0
FAILED = []


def log(m):
    if main:
        print(m, flush=True)


def check(name, ok, detail=''):
    if main:
        print(f'  [{"PASS" if ok else "FAIL"}] {name:52s} {detail}', flush=True)
    if not ok:
        FAILED.append(name)


def rel(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


if args.full:
    cfg = WAN_CONFIGS['t2v-1.3B']
    MK = dict(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
              num_heads=cfg.num_heads, num_layers=cfg.num_layers,
              window_size=cfg.window_size)
    HP, WP, TEXT = 23, 40, 512
else:
    MK = dict(dim=256, ffn_dim=512, freq_dim=256, num_heads=4, num_layers=4,
              window_size=(-1, -1))
    HP, WP, TEXT = 6, 8, 32


def build():
    """Identical weights on every rank -- the reference must be rank-invariant.

    `WanModel.init_weights` does `nn.init.zeros_(self.head.head.weight)`, so a
    freshly built model returns EXACTLY zero from every forward, and every
    gradient behind the head is zero too. Left alone, this file would compare
    zeros against zeros and pass unconditionally -- the precise failure it
    exists to prevent, one level up. Give the head real weights.
    """
    torch.manual_seed(1234)
    m = WanModel(qk_norm=True, cross_attn_norm=True, eps=1e-6,
                 text_len=TEXT, **MK)
    torch.nn.init.normal_(m.head.head.weight, std=0.02)
    return m


F, C = args.frames, 16
H, W = HP * 2, WP * 2
torch.manual_seed(99)
z = torch.randn(1, C, F, H, W, device=dev)
ctx = torch.randn(1, TEXT, MK['dim'], device=dev)
tgt = torch.randn(1, C, F, H, W, device=dev)

log(f'=== FSDP port gate | world_size {world} | '
    f'{"1.3B" if args.full else "tiny"} | {F} frames {H}x{W} ===')

# ------------------------------------------------------------------ reference
ref_model = build().to(dev).float()
rope = make_rope_table(ref_model, HP, WP, F + 8, dev)


def fwd(model, grad=False):
    with torch.set_grad_enabled(grad):
        return bc.block_forward(model, z, 0.3, 0, rope, ctx, None, kv=None,
                                dtype=torch.bfloat16, time_scale=1000.0)


ref_out = fwd(ref_model).float()
loss = torch.nn.functional.mse_loss(fwd(ref_model, grad=True).float(), tgt)
loss.backward()
ref_grads = {n: p.grad.detach().float().clone()
             for n, p in ref_model.named_parameters() if p.grad is not None}
ref_model.zero_grad(set_to_none=True)

# Non-degeneracy. Everything below compares the sharded run against `ref_out`
# and `ref_grads`, and `rel_err 0` is the expected PASS -- so if the reference
# were all zeros every check would pass while testing nothing. That is not
# hypothetical: WanModel.init_weights zero-initialises head.head.weight, and
# this file did exactly that until `build` was fixed to give the head weights.
log('\n--- 0. the reference is not degenerate ---')
n_zero_grads = sum(1 for g in ref_grads.values() if float(g.abs().max()) == 0.0)
check('reference output is non-zero', float(ref_out.abs().max()) > 1e-6,
      f'|out|_max {float(ref_out.abs().max()):.4f}')
check('reference gradients are non-zero',
      n_zero_grads == 0 and len(ref_grads) > 0,
      f'{len(ref_grads)-n_zero_grads}/{len(ref_grads)} params have non-zero grad')

# ----------------------------------------------- 1. wrapping alone is a no-op
log('\n--- 1. wrapping alone changes nothing (no FSDP yet) ---')
wrapped = attach_causal_modules(build().to(dev).float())
e = rel(fwd(wrapped), ref_out)
check('CausalBlock/CausalHead == plain forward', e < 1e-5, f'rel_err {e:.2e}')
del wrapped

# ------------------------------------------------------- 2/4. sharded forward
log('\n--- 2/4. FSDP2-sharded forward and gradients ---')
sharded = build()
shard_model(sharded, reshard_after_forward=True)
sharded = sharded.to(dev)
n_local = sum(p.to_local().numel() if hasattr(p, 'to_local') else p.numel()
              for p in sharded.parameters())
n_total = sum(p.numel() for p in sharded.parameters())
log(f'    {n_total/1e6:.1f}M params -> {n_local/1e6:.1f}M local on rank {rank} '
    f'({n_total/max(n_local,1):.2f}x reduction)')

sh_out = fwd(sharded).float()
e = rel(sh_out, ref_out)
check('sharded forward == single-GPU forward', e < args.tol, f'rel_err {e:.5f}')

loss = torch.nn.functional.mse_loss(fwd(sharded, grad=True).float(), tgt)
loss.backward()


def canonical(n):
    """Shard-unit name -> the WanModel name the reference gradients use."""
    if n.startswith('causal_layers.'):
        i, rest = n[len('causal_layers.'):].split('.blk.', 1)
        return f'blocks.{i}.{rest}'
    if n.startswith('causal_head.head.'):
        return 'head.' + n[len('causal_head.head.'):]
    return n


worst, worst_n, compared = 0.0, '', 0
for n, p in sharded.named_parameters():
    if p.grad is None:
        continue
    cn = canonical(n)
    if cn not in ref_grads:
        continue
    g = p.grad
    if hasattr(g, 'full_tensor'):
        g = g.full_tensor()
    compared += 1
    r = rel(g, ref_grads[cn])
    if r > worst:
        worst, worst_n = r, cn
# A comparison that silently matched nothing is the same bug this file exists
# to catch, one level up: it would pass forever. Require full coverage.
check('sharded gradients == single-GPU gradients',
      compared == len(ref_grads) and worst < 5e-2,
      f'{compared}/{len(ref_grads)} params, worst rel_err {worst:.5f} ({worst_n})')

# ------------------------------- 5. checkpoint round-trip into a plain WanModel
log('\n--- 5. full_state_dict loads into a plain WanModel ---')
fsd = full_state_dict(sharded)
ok, detail = False, ''
if main:
    plain = build()
    try:
        res = plain.load_state_dict(fsd, strict=True)
        plain = plain.to(dev).float()
        e = rel(fwd(plain), ref_out)
        ok = e < 1e-5
        detail = f'{len(fsd)} tensors, rel_err {e:.2e}'
        del plain
    except Exception as ex:
        detail = f'{type(ex).__name__}: {ex}'
flag = torch.tensor([1.0 if ok else 0.0], device=dev)
dist.broadcast(flag, 0)
check('full_state_dict -> WanModel round-trip', bool(flag.item()), detail)
del sharded, fsd
torch.cuda.empty_cache()

# ---------------------------------------------------------- 3. BROKEN CONTROL
log('\n--- 3. CONTROL: sharding the block WITHOUT entering it ---')
log('    (fully_shard on WanAttentionBlock; _layer reaches past the hook)')
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
_mesh = init_device_mesh('cuda', (world,), mesh_dim_names=('dp',))
broken = build()
# Only the blocks, and no root shard -- so this isolates the one failure being
# controlled for (the pre-forward hook never fires) rather than tripping over
# mesh composition on the way there.
for blk in broken.blocks:
    fully_shard(blk, mesh=_mesh)
broken = broken.to(dev)
detected, detail = False, ''
try:
    out = fwd(broken).float()
    e = rel(out, ref_out)
    detected = e > 1e-2
    detail = f'rel_err {e:.5f} (differs => the gate can see this)'
except Exception as ex:
    detected = True
    detail = f'raises {type(ex).__name__}: {str(ex).splitlines()[0][:90]}'
check('CONTROL mis-sharded forward != reference', detected, detail)

log('\n' + ('ALL CHECKS PASSED' if not FAILED
            else f'{len(FAILED)} FAILED: {FAILED}'))
dist.barrier()
dist.destroy_process_group()
if FAILED and main:
    sys.exit(1)
