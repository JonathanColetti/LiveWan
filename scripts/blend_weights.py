"""Linearly interpolate two checkpoints of the same model.

Stage 1 is stable but was over smoothed

the DMD student is sharp and moving but
drifts over a long rollout. They are the same network, 1 fine tuned from the
other, so the segment between them is a 1-parameter family trading exactly
those two properties off and picking a point on it is far cheaper than
retraining. This is the standard linear-mode-connectivity argument, and it only
applies because one is a continuation of the other.
"""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True, help='checkpoint A (e.g. stage 1)')
ap.add_argument('--b', required=True, help='checkpoint B (e.g. DMD)')
ap.add_argument('--alpha', type=float, required=True,
                help='weight on B; 0 = pure A, 1 = pure B')
ap.add_argument('--key-a', default='model')
ap.add_argument('--key-b', default='model')
ap.add_argument('--out', required=True)
args = ap.parse_args()

sa = torch.load(args.a, map_location='cpu')
sb = torch.load(args.b, map_location='cpu')
a = sa.get(args.key_a, sa)
b = sb.get(args.key_b, sb)

out, skipped = {}, 0
for k, va in a.items():
    vb = b.get(k)
    if vb is None or not va.is_floating_point():
        out[k] = va
        skipped += 1
        continue
    out[k] = (1 - args.alpha) * va.float() + args.alpha * vb.float()
    out[k] = out[k].to(va.dtype)
torch.save({'model': out, 'blend': {'a': args.a, 'b': args.b,
                                    'alpha': args.alpha}}, args.out)
print(f'blended {len(out)-skipped} tensors at alpha={args.alpha} '
      f'({skipped} copied verbatim) -> {args.out}')
