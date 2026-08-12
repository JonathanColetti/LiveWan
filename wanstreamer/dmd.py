"""Distribution Matching Distillation for the blockcausal student.

The generator loss is the DMD2 form. Given a generated clean latent `x0`, noise
it to some level t and ask two score networks where they think it came from:

    x0_real = teacher(x_t, t)      with CFG   -- the distribution we want
    x0_fake = critic(x_t, t)                  -- the distribution we have
    grad    = (x0_fake - x0_real) / normalizer
    L_gen   = 0.5 * || x0 - stopgrad(x0 - grad) ||^2

so d L_gen / d x0 == grad, which is the KL gradient between the two
distributions. The normalizer is DMD2 scale invariant one. Which is mean |x0 - x0_real|
. Without which the loss magnitude swings by orders of magnitude across t and
the generator LR cannot be set at all.

The critic is trained in the ordinary way, a flow-matching loss on the student's
own samples, so it tracks a moving target. That is the whole reason DMD needs
two time scales: the critic must stay ahead of the generator.

Why the teacher is allowed to be bidirectional: it only ever scores a finished
clip, never generates one. Causality is a constraint on the students sampling
procedure, not on the reference distribution.
"""
import torch
import torch.nn.functional as tnnF

from . import blockcausal as bc
from .data import add_noise


def bidirectional_velocity(model, z, t, rope, ctx, ctx_lens, dtype,
                           time_scale=1000.0, grad_checkpoint=False):
    """Full bidirectional forward over a whole clip.

    `block_forward` with no K/V prefix and the entire clip as one block *is* the
    bidirectional forward -- attention runs over the clip and nothing else -- so
    teacher, critic and student all go through one code path and one RoPE
    implementation.
    """
    return bc.block_forward(model, z, t, 0, rope, ctx, ctx_lens, kv=None,
                            dtype=dtype, time_scale=time_scale,
                            grad_checkpoint=grad_checkpoint)


def cfg_velocity(model, z, t, rope, ctx_pos, ctx_neg, ctx_lens, dtype,
                 guidance, time_scale=1000.0):
    """Teacher velocity with classifier-free guidance, cond and uncond batched."""
    if not guidance or guidance == 1.0:
        return bidirectional_velocity(model, z, t, rope, ctx_pos, ctx_lens,
                                      dtype, time_scale)
    z2 = torch.cat([z, z], 0)
    ctx2 = torch.cat([ctx_pos, ctx_neg], 0)
    lens2 = ctx_lens.expand(2)
    v = bidirectional_velocity(model, z2, t, rope, ctx2, lens2, dtype, time_scale)
    vc, vu = v[0:1], v[1:2]
    return vu + guidance * (vc - vu)


@torch.no_grad()
def dmd_gradient(x0, t, z_t, v_real, v_fake):
    """DMD2's KL gradient wrt the generated clip, for the whole clip at once.

    One normalizer for the clip rather than one per block: per-block
    normalisation would rescale each block's gradient by its own error and so
    quietly up-weight whichever block is currently worst.
    """
    while t.dim() < z_t.dim():
        t = t.unsqueeze(-1)
    x0_real = (z_t - t * v_real).float()
    x0_fake = (z_t - t * v_fake).float()
    normalizer = (x0.float() - x0_real).abs().mean().clamp_min(1e-4)
    grad = torch.nan_to_num((x0_fake - x0_real) / normalizer)
    return grad, {'dmd_grad_norm': float(grad.norm()),
                  'dmd_normalizer': float(normalizer),
                  'dmd_real_fake_gap': float((x0_real - x0_fake).abs().mean())}


def dmd_surrogate(x0_hat, grad, scale=1.0):
    """Scalar whose gradient wrt x0_hat is exactly `grad` (times scale)."""
    return scale * 0.5 * tnnF.mse_loss(x0_hat.float(),
                                       (x0_hat.detach() - grad).float())


def critic_loss(model, x0, t, rope, ctx, ctx_lens, dtype, time_scale=1000.0,
                grad_checkpoint=True, noise=None):
    """Flow-matching loss for the fake-score network on student samples."""
    z_t, target, _ = add_noise(x0, t, noise)
    v = bidirectional_velocity(model, z_t, t, rope, ctx, ctx_lens, dtype,
                               time_scale, grad_checkpoint=grad_checkpoint)
    return tnnF.mse_loss(v.float(), target.float())
