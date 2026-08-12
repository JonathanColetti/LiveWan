"""Few-step block-causal streaming: the deployment loop, and the loop the
self-forcing trainer rolls out through.

One class serves both, because that identity *is* the method. Self-forcing only
means anything if the context the student is trained on is the context it will
actually receive -- its own previous output, encoded into the same K/V cache, at
the same noise levels, with the same number of steps.

Per streaming unit (a block of `block_frames` latent frames):

    z = noise
    for t, t_next in schedule:            # e.g. 1.0 -> .75 -> .5 -> .25 -> 0
        x0 = z - t * G(z, t | clean K/V of world + committed events)
        z  = (1 - t_next) * x0 + t_next * eps            (fresh eps; last step
                                                          leaves x0 as-is)
    G(x0, t=0)  -> write this block's clean K/V, commit

The re-run at t=0 is what keeps the cache clean: every committed key/value in
the prefix was produced from a *clean* latent, which is the condition Stage 1
trained under. It costs one extra forward per unit and is worth it.

The world's K/V is pinned; eviction under a bounded window drops the oldest
events only, so memory and per-unit cost are flat in stream length while scene
and subject identity persist (v0.3's "world + event stream").
"""
import time

import torch

from . import blockcausal as bc
from .core import make_rope_table, make_cache, latent_geometry
from .graphrunner import GraphedForward, SteadyStateGraphs


def few_step_schedule(num_steps, t_max=1.0, shift=1.0):
    """Flow fractions, highest first.

    shift == 1 gives the uniform spacing DMD-distilled few-step video models use
    (CausVid / Self-Forcing: 1.0, .75, .5, .25). shift > 1 reproduces Wan's own
    inference schedule, which is the right one for evaluating a many-step,
    not-yet-distilled model.
    """
    ts = [t_max * (num_steps - i) / num_steps for i in range(num_steps)]
    if shift == 1.0:
        return ts
    return [shift * t / (1 + (shift - 1) * t) for t in ts]


class FewStepStreamer:
    def __init__(self, model, width=640, height=368, max_frames=256,
                 device='cuda', dtype=torch.bfloat16, window_frames=None,
                 time_scale=1000.0, cache_frames=None, block_frames=3,
                 num_steps=4, rope=None, shift=1.0, sampler='renoise',
                 cuda_graphs=False, noisy_context=False):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.time_scale = time_scale
        self.block_frames = block_frames
        self.schedule = few_step_schedule(num_steps, shift=shift)
        self.sampler = sampler
        self.h_lat, self.w_lat, self.hp, self.wp = latent_geometry(width, height)
        self.S = self.hp * self.wp
        self.max_frames = max_frames
        self.window_frames = window_frames
        self.rope = rope if rope is not None else make_rope_table(
            model, self.hp, self.wp, max_frames, device)
        self.cache = make_cache(model, self.S, cache_frames or max_frames,
                                device, dtype)
        self.kv = bc.BufferKV(self.cache)
        self.n_world = 0
        self.n_frames = 0
        # Training-only: pushes the event window this far from the pinned world
        # in RoPE index space, simulating a long stream. At deployment the gap
        # arises naturally from evicted events, so this stays 0.
        self.rope_gap = 0
        # per-channel moment matching towards the world (0 = off); see _renorm
        self.latent_norm = 0.0
        self.ref_stats = None
        self.graphs = SteadyStateGraphs() if cuda_graphs else None
        # "Noisy context": commit the K/V the LAST denoising step already wrote
        # instead of re-running the finished block at t=0. That drops the cost
        # of a unit from N+1 forwards to N -- a third of the latency at N=2,
        # which is the single largest inference-cost lever left in this loop.
        # It is NOT a free inference-time swap: every committed key then comes
        # from a latent at the schedule's lowest noise level rather than a clean
        # one, and Stage 1 and DMD both trained against a clean prefix. Turning
        # it on without retraining under the same convention changes the
        # conditioning the student was distilled for. Exposed here so the
        # latency claim can be measured before that training is paid for.
        self.noisy_context = noisy_context

    # ------------------------------------------------------------------ context
    def set_text(self, ctx_emb, ctx_lens=None):
        self.ctx = ctx_emb
        # Resolve "is a cross-attention mask needed?" ONCE, here, and store None
        # if it is not. Deciding it inside the forward costs a `.item()` on a
        # device tensor -- a host sync, which is illegal during CUDA graph
        # capture. Contexts are padded to the full text length (upstream's own
        # WanModel.forward passes context_lens=None for exactly this reason), so
        # None is not an approximation, it is the same computation.
        full = ctx_emb.shape[1]
        if ctx_lens is not None and int(torch.as_tensor(ctx_lens).min()) < full:
            self.ctx_lens = ctx_lens
        else:
            self.ctx_lens = None

    def _eager(self, z, tbl, e, e0):
        return bc.block_forward(self.model, z, 0.0, 0, self.rope, self.ctx,
                                self.ctx_lens, kv=self.kv, dtype=self.dtype,
                                time_scale=self.time_scale, tbl=tbl, emb=(e, e0))

    def _fwd(self, z, t, t_start):
        tbl = self.rope.span(t_start, z.shape[2])
        e, e0 = bc.time_embed(self.model, t, self.device, self.time_scale)
        if self.graphs is not None and not torch.is_grad_enabled():
            # Key on the cache length: that is the only thing that changes the
            # shapes, and a bounded window makes it constant once saturated.
            g = self.graphs.get(
                (self.cache.length, z.shape[2]),
                lambda: GraphedForward(self._eager, tuple(z.shape),
                                       tuple(tbl.shape), tuple(e.shape),
                                       tuple(e0.shape), self.device))
            if g is not None:
                return g(z, tbl, e, e0)
        return self._eager(z, tbl, e, e0)

    def _commit(self, n):
        self.cache.commit(n * self.S)
        self.n_frames += n
        if self.window_frames is not None:
            self.trim_to_window(self.window_frames)

    def trim_to_window(self, window_frames):
        """Evict the oldest events so only `window_frames` of them remain.

        Deployment calls this from `_commit` after every unit, which is what
        makes per-unit cost flat in stream length. The self-forcing trainer
        cannot: it records denoising steps during a rollout and redoes them
        under gradient afterwards against `buffer[:prefix]`, and an eviction in
        between silently changes what those tokens are (see
        StreamingKVCache.evictions). So the trainer leaves `window_frames` None,
        rolls one iteration's blocks with the buffer growing monotonically, and
        calls this itself at the iteration boundary -- the one point where no
        recorded step is still owed its prefix.

        The world stays pinned either way; only events are dropped.
        """
        protect = self.n_world * self.S
        budget = (self.n_world + window_frames) * self.S
        excess = self.cache.num_tokens - budget
        if excess > 0:
            self.cache.evict_front(excess, protect=protect)

    # -------------------------------------------------------------------- world
    @torch.no_grad()
    def set_world(self, world_latents):
        """Prime the cache from clean world latents, attended bidirectionally."""
        self.cache.reset()
        self.n_frames = 0
        z = world_latents.to(self.device).unsqueeze(0) \
            if world_latents.dim() == 4 else world_latents.to(self.device)
        F = z.shape[2]
        self._fwd(z, 0.0, 0)
        self._commit(F)
        self.n_world = F
        # Per-channel moments of the world: the in-distribution reference for
        # `latent_norm`. Computed over (F, H, W) per channel.
        w = z[0].float()
        self.ref_stats = (w.mean(dim=(1, 2, 3), keepdim=True).unsqueeze(0),
                          w.std(dim=(1, 2, 3), keepdim=True).unsqueeze(0))
        return F

    def _renorm(self, z, strength):
        """Pull a finished block's PER-CHANNEL moments back towards the world's.

        Long autoregressive rollouts drift, and here the drift is measurably a
        per-channel one: the decoded stream develops colour casts and saturated
        blotches long before it loses structure. A distilled student inherits
        this from the CFG-guided real score it was matched to -- CFG moves mass
        towards higher-contrast, more saturated samples, and in a feedback loop
        (output becomes context becomes output) that bias compounds.

        Matching the first two moments per channel against the world is the
        cheapest correction that targets exactly that, and it is applied BEFORE
        the block's K/V is recomputed and committed, so the cached history stays
        in distribution and the correction cannot itself accumulate. It cannot
        fix a wrong *direction*, only a wrong per-channel scale and offset.
        """
        if strength <= 0 or getattr(self, 'ref_stats', None) is None:
            return z
        mu, sd = self.ref_stats
        zm = z.mean(dim=(2, 3, 4), keepdim=True)
        zs = z.std(dim=(2, 3, 4), keepdim=True)
        z_n = (z - zm) / (zs + 1e-5) * sd.to(z.dtype) + mu.to(z.dtype)
        return (1.0 - strength) * z + strength * z_n

    # ------------------------------------------------------------------- events
    def generate_block(self, generator=None, record=None, n_frames=None):
        """Emit one streaming unit. Returns clean latents [1, C, b, H, W]."""
        b = n_frames or self.block_frames
        t0 = bc.event_rope_index(self.n_frames, self.n_world, self.rope_gap)
        z = torch.randn(1, 16, b, self.h_lat, self.w_lat, device=self.device,
                        dtype=torch.float32, generator=generator)
        sched = self.schedule
        for i, t in enumerate(sched):
            if record is not None:
                record.append({'block_start': t0, 'step': i, 't': t,
                               'z': z.detach()})
            v = self._fwd(z, t, t0).float()
            x0 = z - t * v
            tn = sched[i + 1] if i + 1 < len(sched) else 0.0
            if tn == 0.0:
                z = x0
            elif self.sampler == 'renoise':
                # DMD-distilled few-step sampler: re-noise the x0 prediction to
                # the next level with FRESH noise. This is the sampler the
                # student is distilled under, so it is also the one it must be
                # rolled out with.
                eps = torch.randn(z.shape, device=z.device, dtype=z.dtype,
                                  generator=generator)
                z = (1 - tn) * x0 + tn * eps
            else:
                z = z + (tn - t) * v          # deterministic Euler, for the
                #                               undistilled many-step model
        z = self._renorm(z, self.latent_norm)
        if not self.noisy_context:
            self._fwd(z, 0.0, t0)      # clean K/V for the committed block
        # else: the scratch region still holds the K/V written by the final
        # denoising forward, so committing promotes that instead and the extra
        # pass is skipped entirely. (`_renorm` is then not reflected in the
        # cache, which is another reason the two options are not interchangeable
        # without retraining.)
        self._commit(b)
        return z

    @torch.no_grad()
    def stream(self, num_blocks, seed=0, log_every=0, n_frames=None):
        g = torch.Generator(device=self.device).manual_seed(seed)
        lats, times = [], []
        for u in range(num_blocks):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            z = self.generate_block(generator=g, n_frames=n_frames)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            lats.append(z[0])
            times.append(dt)
            if log_every and (u + 1) % log_every == 0:
                b = n_frames or self.block_frames
                print(f'  unit {u+1}/{num_blocks}: {dt*1000:7.1f} ms '
                      f'({dt/b*1000:6.1f} ms/latent frame, '
                      f'cache {self.cache.num_tokens} tok)', flush=True)
        return torch.cat(lats, dim=1), times

    # --------------------------------------------------- self-forcing rollout
    @torch.no_grad()
    def rollout_record(self, num_blocks, generator=None):
        """Roll out `num_blocks` units and keep what the DMD step needs.

        Returns (x0_blocks, records). `records[i]` holds, for every denoising
        step of block i, the exact input latent and timestep, plus the cache
        length at that point -- enough to recompute any single step under
        gradient without re-running the rollout.
        """
        out, recs = [], []
        for _ in range(num_blocks):
            rec = []
            prefix = self.cache.num_tokens
            ev = self.cache.evictions
            z = self.generate_block(generator=generator, record=rec)
            for r in rec:
                r['prefix'] = prefix
                r['evictions'] = ev
            out.append(z)
            recs.append(rec)
        return out, recs

    def prefix_intact(self, rec_step):
        """Is this recorded step's prefix still literally `buffer[:prefix]`?

        False once anything has been evicted since it was recorded -- the tokens
        at those indices are now different tokens. The trainer must not take a
        gradient through a step for which this is False; the forward would
        succeed and quietly train against the wrong context.
        """
        return rec_step.get('evictions', 0) == self.cache.evictions

    def recompute_step(self, rec_step, prefix_kv):
        """Differentiably redo one recorded denoising step against a detached
        prefix. Returns that step's x0 prediction."""
        z, t = rec_step['z'], rec_step['t']
        v = bc.block_forward(self.model, z, t, rec_step['block_start'],
                             self.rope, self.ctx, self.ctx_lens, kv=prefix_kv,
                             dtype=self.dtype, time_scale=self.time_scale,
                             prefix_upto=rec_step['prefix'],
                             grad_checkpoint=True)
        return z - t * v.float()
