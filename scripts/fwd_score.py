"""Frechet distance to held-out teacher clips, in the teacher's own features.

A metric that goes UP when the picture gets worse cannot be the thing a sweep is ranked on.

FVD would be the standard answer, but it wants an I3D or VideoMAE checkpoint
that is not on this machine, and downloading a third network to judge a video
model when a perfectly good one is already loaded is silly. So the features come
from the frozen teacher itself:

  * noise the clip to a fixed, low t (the same draw for every clip being
    compared), run the bidirectional teacher over it, and take the self-attention
    KEYS at one middle layer -- `block_forward(..., collect=[])` already hands
    those out, so no hook is needed and nothing in the forward path changes;
  * mean-pool over spatial tokens, giving one 1536-d vector per latent frame;
  * fit a Gaussian to the reference set (real teacher clips) and to the clip
    under test, and report the Frechet distance between them.

Read it as a distance, so lower is better and 0 is unreachable (the reference
set's own split-half distance is printed as the floor).


  # once: cache reference statistics from the teacher clips
  python scripts/fwd_score.py --build-ref --out data/fwd_ref.npz

  # then: score any run's latents against it
  python scripts/fwd_score.py out/FINAL/latents.pt out/base_ln00/latents.pt \
      --ref data/fwd_ref.npz
"""
import os, sys, glob, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from safetensors.torch import load_file

from wanstreamer import blockcausal as bc
from wanstreamer.core import make_rope_table

ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs='*', help='.pt files with latents (+prompt_idx)')
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--prompts', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--teacher', default=os.path.join(HERE, 'data/teacher'))
ap.add_argument('--build-ref', action='store_true')
ap.add_argument('--ref', default=os.path.join(HERE, 'data/fwd_ref.npz'))
ap.add_argument('--ref-clips', type=int, default=320,
                help='teacher clips in the reference set; the rest are held out '
                     'to measure the floor')
ap.add_argument('--layer', type=int, default=20,
                help='which of the 30 blocks to read features from. Middle '
                     'layers carry content; the last ones carry the velocity '
                     'prediction and the first ones barely differ from pixels.')
ap.add_argument('--t', type=float, default=0.25,
                help='noise level the features are read at, fixed across every '
                     'clip so the comparison is like-for-like')
ap.add_argument('--chunk', type=int, default=21,
                help='window length; the teacher is trained at 21 frames')
ap.add_argument('--skip-world', type=int, default=-1,
                help='drop this many leading latent frames (the world) before '
                     'scoring; -1 uses the value stored in the file')
ap.add_argument('--dims', type=int, default=96,
                help='PCA dimensions the Frechet distance is computed in. The '
                     'raw features are 1536-d, but a 100-unit run only yields '
                     '~300 feature frames -- fewer samples than dimensions, '
                     'which makes the sample covariance rank-deficient and the '
                     'distance meaningless in absolute terms. The projection '
                     'is fitted ONCE on the reference set and reused, so it is '
                     'the same basis for every run being compared. 0 disables.')
ap.add_argument('--out', default='')
args = ap.parse_args()

dev = torch.device('cuda')
cfg = WAN_CONFIGS['t2v-1.3B']
DTYPE = torch.bfloat16
blob = torch.load(args.prompts, map_location='cpu')

model = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
model.load_state_dict(
    load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'), strict=True)
model = model.to(dev).to(DTYPE).eval().requires_grad_(False)

_ropes = {}


def rope_for(hp, wp):
    if (hp, wp) not in _ropes:
        _ropes[(hp, wp)] = make_rope_table(model, hp, wp, args.chunk + 4, dev)
    return _ropes[(hp, wp)]


@torch.no_grad()
def features(x0, ctx_emb, ctx_lens):
    """[1, C, F, H, W] clean latents -> [F, dim] float32 features.

    The noise draw is seeded off nothing but the frame count, so two clips of
    the same length are compared under the *same* eps -- otherwise the Frechet
    distance would partly measure noise, not content.
    """
    F = x0.shape[2]
    hp, wp = x0.shape[3] // 2, x0.shape[4] // 2
    rope = rope_for(hp, wp)
    g = torch.Generator(device=dev).manual_seed(20260731 + F)
    eps = torch.randn(x0.shape, device=dev, dtype=torch.float32, generator=g)
    z_t = ((1 - args.t) * x0 + args.t * eps).to(DTYPE)
    collect = []
    bc.block_forward(model, z_t, args.t, 0, rope, ctx_emb, ctx_lens, kv=None,
                     collect=collect, dtype=DTYPE)
    k = collect[args.layer][0]                    # [1, F*S, heads, head_dim]
    S = rope.seq
    k = k.reshape(1, F, S, -1).mean(dim=2)[0]     # pool spatial -> [F, dim]
    return k.float().cpu().numpy()


def ctx_for(pi):
    ctx = blob['pos'][pi].to(dev, DTYPE).unsqueeze(0)
    with torch.amp.autocast('cuda', enabled=False):
        emb = model.text_embedding(ctx)
    return emb, torch.tensor([ctx.shape[1]], device=dev, dtype=torch.long)


def windows(F, chunk):
    """Non-overlapping windows, plus a final partial one if it is long enough."""
    out = [(s, min(s + chunk, F)) for s in range(0, F, chunk)]
    return [(s, e) for s, e in out if e - s >= 8]


def clip_features(x0, pi, skip=0):
    emb, lens = ctx_for(pi)
    x0 = x0[:, :, skip:]
    feats = [features(x0[:, :, s:e], emb, lens) for s, e in
             windows(x0.shape[2], args.chunk)]
    return np.concatenate(feats, 0) if feats else np.zeros((0, cfg.dim))


def frechet(a, b):
    """Frechet distance between Gaussians fitted to feature sets a and b."""
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)
    diff = mu1 - mu2
    # Tr((S1 S2)^1/2) via the symmetric form, which stays real and PSD.
    s1h = linalg.sqrtm(s1 + np.eye(len(s1)) * 1e-6).real
    m = s1h @ s2 @ s1h
    ev = np.clip(linalg.eigvalsh(m), 0, None)
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(ev).sum())


# ------------------------------------------------------------ build reference
if args.build_ref:
    files = sorted(glob.glob(os.path.join(args.teacher, '*.pt')))
    n_ref = min(args.ref_clips, len(files))
    ref, hold = [], []
    for i, f in enumerate(files):
        d = torch.load(f, map_location='cpu', weights_only=False)
        fe = clip_features(d['latents'].float().to(dev).unsqueeze(0),
                           int(d['prompt_idx']))
        (ref if i < n_ref else hold).append(fe)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(files)} clips', flush=True)
    ref = np.concatenate(ref, 0)
    hold = np.concatenate(hold, 0) if hold else ref[len(ref) // 2:]
    out = args.out or args.ref
    np.savez_compressed(out, ref=ref.astype(np.float32),
                        hold=hold.astype(np.float32), layer=args.layer, t=args.t)
    floor = frechet(ref, hold)
    print(f'\nwrote {out}: ref {ref.shape} from {n_ref} clips, '
          f'held-out {hold.shape} from {len(files)-n_ref}')
    print(f'FLOOR (real vs real, held out): FWD {floor:.3f}  '
          f'-- no generated clip should be expected to beat this')
    sys.exit(0)

# ------------------------------------------------------------------- score
z = np.load(args.ref)
ref, hold = z['ref'], z['hold']
if int(z['layer']) != args.layer or abs(float(z['t']) - args.t) > 1e-9:
    raise SystemExit(f'--ref was built at layer {int(z["layer"])} t {float(z["t"])}, '
                     f'but this run asks for layer {args.layer} t {args.t}')

PROJ = None
if args.dims:
    # Fitted on the reference only, then frozen. Every clip -- held-out real and
    # generated alike -- is projected through the same basis, so the comparison
    # stays like-for-like; and the reference has 6720 samples, so the basis
    # itself is well determined even though an individual run is not.
    mu0 = ref.mean(0)
    _, _, vt = np.linalg.svd(ref - mu0, full_matrices=False)
    PROJ = vt[:args.dims].T

    def project(a):
        return (a - mu0) @ PROJ

    ref, hold = project(ref), project(hold)
    print(f'PCA: 1536 -> {args.dims} dims, basis fitted on the reference set')

floor = frechet(ref, hold)
print(f'reference: {ref.shape[0]} feature frames | floor (real vs held-out real) '
      f'FWD {floor:.3f}\n')

files = []
for pat in args.inputs:
    files.extend(sorted(glob.glob(pat)) if any(c in pat for c in '*?[') else [pat])

rows = []
for f in files:
    d = torch.load(f, map_location='cpu')
    x0 = d['latents'].float().to(dev).unsqueeze(0)
    pi = d.get('prompt_idx', 0)
    if 'prompt' in d and isinstance(d['prompt'], str) and d['prompt'] in blob['prompts']:
        pi = blob['prompts'].index(d['prompt'])
    skip = d.get('world_frames', 0) if args.skip_world < 0 else args.skip_world
    fe = clip_features(x0, pi, skip=int(skip))
    if PROJ is not None:
        fe = project(fe)
    fwd = frechet(ref, fe)
    # Split the generated segment in half: a rising FWD is drift the aggregate
    # number hides. Only meaningful with more samples than dimensions, so it is
    # reported as NaN rather than as a confidently wrong number.
    h = len(fe) // 2
    need = 2 * (args.dims or fe.shape[1])
    fwd_a = frechet(ref, fe[:h]) if h > need else float('nan')
    fwd_b = frechet(ref, fe[h:]) if h > need else float('nan')
    name = os.path.basename(os.path.dirname(f)) or os.path.basename(f)
    rows.append({'run': name, 'file': f, 'frames': int(fe.shape[0]),
                 'fwd': fwd, 'fwd_first_half': fwd_a, 'fwd_second_half': fwd_b,
                 'floor': floor})
    print(f'{name:24s} FWD {fwd:8.3f}  ({fwd/floor:5.2f}x floor)   '
          f'first half {fwd_a:8.3f} -> second half {fwd_b:8.3f}', flush=True)

if args.out:
    with open(args.out, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print(f'\nwrote {args.out}')
