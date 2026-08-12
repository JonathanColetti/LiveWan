"""Teacher score quality metric: how in-distribution is a generated clip?

The prior work's Laplacian sharpness statistic was fooled three times (periodic
artifacts read as detail, background grain read as detail, a soft denominator
inflating a ratio), so a second, independent measure is needed -- one that
cannot be gamed by adding high-frequency energy.

This one is the teacher's own denoising-score-matching loss on the clip:

    err(t) = E_eps || v_teacher((1-t)x0 + t*eps, t) - (eps - x0) ||^2
             / E_eps || eps - x0 ||^2

which is (up to a t-dependent weight) the diffusion ELBO gap -- a monotone proxy
for how likely the clip is under the model that defines "good" here. Adding
grain or blockiness *raises* it. It is normalised by the trivial predictor's
error, so 1.0 means "the teacher can say nothing about this clip" and lower is
better.

Reported per region: the world (teacher-generated, an in-distribution control
inside the same clip) and the streamed continuation. The world's number is the
floor the student is being measured against, on the same content and the same
noise draws (which is exactly the comparison the sharpness ratio failed to be).
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

from wanstreamer.core import make_rope_table, latent_geometry
from wanstreamer.dmd import bidirectional_velocity

ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs='+', help='.pt files with latents + prompt_idx')
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--prompts', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--world-frames', type=int, default=9)
ap.add_argument('--ts', default='0.2,0.4,0.6,0.8')
ap.add_argument('--repeats', type=int, default=2)
ap.add_argument('--chunk', type=int, default=21,
                help='score in windows of this many latent frames (the teacher '
                     'is trained at 21; longer inputs are out of distribution)')
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
ts = [float(x) for x in args.ts.split(',')]


@torch.no_grad()
def score_window(x0, ctx_emb, ctx_lens, rope, seed0):
    """Normalised teacher flow error per latent frame, averaged over t."""
    per_frame = torch.zeros(x0.shape[2], device=dev)
    for ti, t in enumerate(ts):
        for r in range(args.repeats):
            g = torch.Generator(device=dev).manual_seed(seed0 + 1000 * ti + r)
            eps = torch.randn(x0.shape, device=dev, dtype=torch.float32,
                              generator=g)
            z_t = (1 - t) * x0 + t * eps
            target = eps - x0
            v = bidirectional_velocity(model, z_t.to(DTYPE), t, rope, ctx_emb,
                                       ctx_lens, DTYPE).float()
            num = ((v - target) ** 2).mean(dim=(0, 1, 3, 4))
            den = (target ** 2).mean(dim=(0, 1, 3, 4)).clamp_min(1e-8)
            per_frame += num / den
    return (per_frame / (len(ts) * args.repeats)).cpu().numpy()


files = []
for pat in args.inputs:
    files.extend(sorted(glob.glob(pat)) if any(c in pat for c in '*?[') else [pat])

rows = []
for f in files:
    d = torch.load(f, map_location='cpu')
    x0 = d['latents'].float().to(dev).unsqueeze(0)
    pi = d.get('prompt_idx', d.get('prompt_index', 0))
    if 'prompt' in d and isinstance(d['prompt'], str) and d['prompt'] in blob['prompts']:
        pi = blob['prompts'].index(d['prompt'])
    ctx = blob['pos'][pi].to(dev, DTYPE).unsqueeze(0)
    with torch.amp.autocast('cuda', enabled=False):
        ctx_emb = model.text_embedding(ctx)
    ctx_lens = torch.tensor([ctx.shape[1]], device=dev, dtype=torch.long)
    F = x0.shape[2]
    hp, wp = x0.shape[3] // 2, x0.shape[4] // 2
    rope = make_rope_table(model, hp, wp, args.chunk + 4, dev)

    # Score in teacher-sized windows so the teacher is never asked about a clip
    # length it was not trained on.
    errs = np.zeros(F)
    counts = np.zeros(F)
    for s in range(0, max(1, F - args.chunk + 1), max(1, args.chunk // 2)):
        e = min(s + args.chunk, F)
        if e - s < 4:
            break
        w = score_window(x0[:, :, s:e], ctx_emb, ctx_lens, rope, seed0=hash(f) % 10000)
        errs[s:e] += w
        counts[s:e] += 1
    counts[counts == 0] = 1
    errs = errs / counts

    wf = d.get('world_frames', args.world_frames)
    row = {'file': os.path.basename(f), 'frames': int(F), 'world_frames': int(wf),
           'err_world': float(errs[:wf].mean()) if wf > 0 else None,
           'err_generated': float(errs[wf:].mean()) if F > wf else None,
           'err_first_third': float(errs[wf:wf + max(1, (F - wf) // 3)].mean())
           if F > wf else None,
           'err_last_third': float(errs[-max(1, (F - wf) // 3):].mean())
           if F > wf else None,
           'per_frame': [round(float(x), 4) for x in errs]}
    rows.append(row)
    print(f'{row["file"]:38s} world {row["err_world"] if row["err_world"] is None else round(row["err_world"],4)}'
          f'  gen {round(row["err_generated"],4) if row["err_generated"] else "-"}'
          f'  (first3rd {round(row["err_first_third"],4) if row["err_first_third"] else "-"}'
          f' -> last3rd {round(row["err_last_third"],4) if row["err_last_third"] else "-"})',
          flush=True)

if len(rows) > 1:
    gw = [r['err_world'] for r in rows if r['err_world'] is not None]
    gg = [r['err_generated'] for r in rows if r['err_generated'] is not None]
    print(f'\nMEAN over {len(rows)} clips: world {np.mean(gw):.4f}  '
          f'generated {np.mean(gg):.4f}  ratio {np.mean(gg)/np.mean(gw):.3f}')
if args.out:
    with open(args.out, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print(f'wrote {args.out}')
