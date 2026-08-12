"""Correctness gate for the rolling-K/V self-forcing rollout.

The mechanism is cheap to state and easy to get silently wrong: keep the K/V
cache between training iterations so the student's self-forcing depth grows past
the 4-6 blocks a single clip allows, toward the 100 blocks deployment actually
runs. The failure mode is not a crash. It is a gradient taken against a prefix
that is no longer the prefix the block was generated with, which trains fine and
produces a worse model.

  1. depth actually grows           n_frames climbs across iterations
  2. memory does NOT               cache stays inside its window budget
  3. prefix intact within an iteration
  4. CONTROL: evicting mid-rollout marks it not-intact
  5. recompute_step is EXACT       redoing a block's final denoising step under
                                   gradient reproduces the rollout's own output
                                   (this is what makes the DMD gradient land on
                                   the sample the score networks were shown)
  6. CONTROL: recomputing against the WRONG prefix does not reproduce it

Runs on a small model in seconds -- none of this depends on width or depth.

  python scripts/verify_rolling.py
"""
import os, sys, argparse

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.modules.model import WanModel

from wanstreamer import blockcausal as bc
from wanstreamer.core import latent_geometry
from wanstreamer.rope import RopeTable
from wanstreamer.stream import FewStepStreamer

ap = argparse.ArgumentParser()
ap.add_argument('--iters', type=int, default=8)
ap.add_argument('--roll-blocks', type=int, default=4)
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--window', type=int, default=6)
ap.add_argument('--world', type=int, default=9)
args = ap.parse_args()

dev = torch.device('cuda')
FAILED = []


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name:54s} {detail}', flush=True)
    if not ok:
        FAILED.append(name)


torch.manual_seed(7)
DIM, HEADS, LAYERS = 256, 4, 4
W, H = 128, 96                      # small frame; geometry is what matters
h_lat, w_lat, hp, wp = latent_geometry(W, H)
S = hp * wp

model = WanModel(dim=DIM, ffn_dim=512, freq_dim=256, num_heads=HEADS,
                 num_layers=LAYERS, window_size=(-1, -1), qk_norm=True,
                 cross_attn_norm=True, eps=1e-6, text_len=32).to(dev).eval()
# `WanModel.init_weights` zero-initialises head.head.weight, so a fresh model
# predicts exactly zero velocity and `x0 = z - t*v` collapses to `z` -- which
# is independent of the K/V prefix, making checks 5 and 6 below pass and fail
# respectively for reasons that have nothing to do with the cache. Give the
# head real weights so the output actually depends on the context.
torch.nn.init.normal_(model.head.head.weight, std=0.02)
model.requires_grad_(True)

ROPE_MAX = args.world + args.iters * args.roll_blocks * args.block + 260
CACHE_F = args.world + args.window + args.roll_blocks * args.block + 2
rope = RopeTable(model.freqs.to(dev), hp, wp, ROPE_MAX, dev)

st = FewStepStreamer(model, width=W, height=H, max_frames=ROPE_MAX, device=dev,
                     dtype=torch.bfloat16, window_frames=None,
                     cache_frames=CACHE_F, block_frames=args.block,
                     num_steps=2, rope=rope)
st.set_text(torch.randn(1, 32, DIM, device=dev))

print(f'=== rolling K/V gate | {LAYERS} layers, {S} tok/frame | world '
      f'{args.world} window {args.window} block {args.block} '
      f'x {args.roll_blocks}/iter, {args.iters} iters ===')

g = torch.Generator(device=dev).manual_seed(1)
with torch.no_grad():
    st.set_world(torch.randn(1, 16, args.world, h_lat, w_lat, device=dev))

budget_tokens = (args.world + args.window + args.roll_blocks * args.block) * S
depths, tokens, intact_all = [], [], True
last_recs = None
for it in range(args.iters):
    if it:
        st.trim_to_window(args.window)          # the iteration boundary
    with torch.no_grad():
        blocks, recs = st.rollout_record(args.roll_blocks, generator=g)
    depths.append(st.n_frames)
    tokens.append(st.cache.num_tokens)
    intact_all &= all(st.prefix_intact(r[-1]) for r in recs)
    last_recs, last_blocks = recs, blocks

# --------------------------------------------------------------- 1. depth grows
grew = depths == sorted(depths) and depths[-1] > depths[0]
expect = args.world + args.iters * args.roll_blocks * args.block
check('1. rollout depth grows across iterations', grew and depths[-1] == expect,
      f'n_frames {depths[0]} -> {depths[-1]} (expected {expect}), '
      f'{depths[-1]/ (args.world + args.roll_blocks*args.block):.1f}x one iteration')

# ------------------------------------------------------------ 2. memory bounded
check('2. cache stays inside the window budget', max(tokens) <= budget_tokens,
      f'max {max(tokens)} tokens vs budget {budget_tokens} '
      f'({max(tokens)/S:.0f} vs {budget_tokens/S:.0f} latent frames)')

# ------------------------------------------------------- 3/4. prefix validity
check('3. every block prefix intact within an iteration', intact_all,
      f'{args.roll_blocks} blocks x {args.iters} iters')

before = st.cache.evictions
st.trim_to_window(1)                       # force an eviction after recording
detected = not any(st.prefix_intact(r[-1]) for r in last_recs)
check('4. CONTROL mid-rollout eviction invalidates the prefix', detected,
      f'evictions {before} -> {st.cache.evictions}, prefix_intact now False')

# ------------------------------------------- 5/6. recompute_step exactness
# Redo the last iteration on a clean boundary so the prefix is valid again.
st.trim_to_window(args.window)
with torch.no_grad():
    blocks, recs = st.rollout_record(args.roll_blocks, generator=g)
prefix_kv = bc.BufferPrefixKV(st.cache)
ki = args.roll_blocks - 1
rec = recs[ki][-1]
x0_hat = st.recompute_step(rec, prefix_kv)
e = float((x0_hat - blocks[ki]).norm() / blocks[ki].norm())
check('5. recompute_step reproduces the rollout output', e < 2e-2,
      f'rel_err {e:.5f}')

# Both controls are judged RELATIVE to the correct-prefix error, not against an
# absolute tolerance. This model is randomly initialised, so its attention is
# only weakly context-sensitive and an absolute threshold would measure the
# initialisation rather than the cache; the ratio is what carries the signal.
def wrong_prefix(upto):
    bad = dict(rec)
    bad['prefix'] = upto
    with torch.no_grad():
        xb = st.recompute_step(bad, prefix_kv)
    return float((xb - blocks[ki]).norm() / blocks[ki].norm())


e_none = wrong_prefix(0)                       # no prefix at all
e_short = wrong_prefix(max(0, rec['prefix'] - 3 * S))   # 3 latent frames short
check('6. CONTROL no prefix does NOT reproduce it', e_none > 5 * e,
      f'rel_err {e_none:.5f} = {e_none/e:.1f}x the correct {e:.5f}')
check('7. CONTROL 3-frame-short prefix is detectably different',
      e_short > 1.5 * e, f'rel_err {e_short:.5f} = {e_short/e:.1f}x correct')

print('\n' + ('ALL CHECKS PASSED' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'))
sys.exit(1 if FAILED else 0)
