"""Streaming generator: persistent world W + autoregressive event stream.

Implements the modelling contract the two papers share, at 1.3B scale:

    p(e_1..K | W, x_1..K) = prod_k p(e_k | W, x_<=k, e_<k)

  * WORLD W -- the established scene/character. Primed ONCE as a single
    bidirectional block, then its K/V is frozen in the cache (v0.2's "KV
    construction").
  * EVENT STREAM -- emitted one chunk at a time. A chunk attends bidirectionally
    within itself and freely over the whole world plus all prior events, and to
    nothing else.

BLOCK-CAUSALITY IS STRUCTURAL. The cache only ever holds world + committed
events, so full attention over [cache, chunk] already gives exactly "world
bidirectional, events causal". No attention mask is required -- verified in
tests/test_attention_fallback.py case 5, and a mask would wrongly serialise
tokens *within* a chunk.

CHUNK SIZE. final.pt was fine-tuned with --num-frames 4, so it only ever saw
4-latent-frame clips. Generating one latent frame in isolation is out of
distribution; `chunk_frames=4` matches training. One latent frame = 4 pixel
frames = 160 ms at 25 FPS (Wan VAE temporal stride 4), so a 4-frame chunk covers
640 ms of video and must be produced in under 640 ms to sustain 25 FPS.

Timesteps are fed to the model in the [0,1] convention final.pt was fine-tuned on
(train_streaming.py:152 draws torch.rand()); see PROGRESS.md §5.
"""
import time

import torch

from wan.modules.model import sinusoidal_embedding_1d
from wan.modules.attention import flash_attention
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)

from .core import (ModulationCache, make_rope_table, make_cache, latent_geometry,
                   timestep_to_train_scale)
from .rope import apply_rope


class StreamingGenerator:
    def __init__(self, model, ctx_emb, width=640, height=368, max_frames=64,
                 device='cuda', dtype=torch.bfloat16, window_frames=None,
                 time_scale=1.0, cache_frames=None):
        """window_frames: if set, evict oldest frames so the cache holds at most
        this many (bounded memory + bounded per-chunk cost). None = unbounded.

        time_scale converts the internal flow fraction t in [0,1] into the value
        the WEIGHTS expect at their time embedding:

            final.pt        -> 1.0     (fine-tuned on torch.rand(), i.e. [0,1])
            original Wan    -> 1000.0  (trained on scheduler timesteps [0,1000])

        This must match the checkpoint. Getting it backwards is not a small
        degradation, it is total: measured normalised velocity error 0.064 vs
        0.732 for the original weights under the two conventions (PROGRESS.md §5).
        Everything internal to this class speaks the [0,1] fraction.
        """
        self.model = model
        self.device = device
        self.dtype = dtype
        self.time_scale = float(time_scale)
        self.h_lat, self.w_lat, self.hp, self.wp = latent_geometry(width, height)
        self.tokens_per_frame = self.hp * self.wp
        self.max_frames = max_frames
        self.window_frames = window_frames
        # RoPE must span the whole stream (absolute temporal indices keep growing),
        # but the K/V cache only ever holds world + window, so sizing it by
        # max_frames wastes memory quadratically in stream length: 30 layers x
        # 6 KB/token means an unbounded 173 frame buffer at 1560 tok/frame is
        # ~48 GiB, versus ~6 GiB for a 21 frame working set.
        self.rope = make_rope_table(model, self.hp, self.wp, max_frames, device)
        self.cache = make_cache(model, self.tokens_per_frame,
                                cache_frames or max_frames, device, dtype)
        self.ctx = ctx_emb
        self.ctx_lens = torch.tensor([ctx_emb.shape[1]], device=device, dtype=torch.long)
        self.n_world = 0
        self.n_frames = 0        # total committed latent frames
        self.ref_stats = None    # per-channel (mean, std) of the world latents

    def _time_embed(self, t_frac):
        """t_frac is the flow fraction in [0,1]; time_scale maps it to the
        checkpoint's own convention."""
        tv = (torch.ones(1, device=self.device, dtype=torch.float32)
              * float(t_frac) * self.time_scale)
        with torch.amp.autocast('cuda', enabled=False):
            e = self.model.time_embedding(
                sinusoidal_embedding_1d(self.model.freq_dim, tv).float())
            e0 = self.model.time_projection(e).unflatten(1, (6, self.model.dim))
        return e, e0

    @torch.no_grad()
    def chunk_forward(self, z, t_frac, t_start, use_cache=True, ctx=None):
        """Run a chunk of latent frames against the cached past.

        z: [1, C, N, H_lat, W_lat]. t_frac is the flow fraction in [0,1].
        Writes the chunk's K/V into the cache scratch region (uncommitted).
        Returns predicted velocity [C, N, H_lat, W_lat].
        Attention is unmasked over [cache, chunk] == block-causal.
        """
        model = self.model
        ctx = self.ctx if ctx is None else ctx
        N = z.shape[2]
        S = N * self.tokens_per_frame
        e, e0 = self._time_embed(t_frac)
        mod = ModulationCache(model.blocks, e0)

        with torch.amp.autocast('cuda', dtype=self.dtype):
            x = model.patch_embedding(z.to(self.dtype))
            grid = torch.stack([torch.tensor(x.shape[2:], dtype=torch.long,
                                             device=self.device)])
            x = x.flatten(2).transpose(1, 2)
            tbl = self.rope.span(t_start, N)

            for li, blk in enumerate(model.blocks):
                ec = mod[li]
                sa_in = blk.norm1(x).float() * (1 + ec[1]) + ec[0]
                n = blk.num_heads
                d = blk.dim // n
                sa = blk.self_attn
                q = sa.norm_q(sa.q(sa_in)).view(1, S, n, d)
                k = sa.norm_k(sa.k(sa_in)).view(1, S, n, d)
                v = sa.v(sa_in).view(1, S, n, d)
                q = apply_rope(q, tbl).to(self.dtype)
                k = apply_rope(k, tbl).to(self.dtype)
                v = v.to(self.dtype)

                if use_cache:
                    self.cache.write(li, k, v)
                    ck, cv = self.cache.context(li, S)
                else:
                    ck, cv = k, v
                y = flash_attention(q=q, k=ck, v=cv, window_size=(-1, -1),
                                    causal=False)
                y = sa.o(y.flatten(2))
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    x = x + y * ec[2]
                x = x + blk.cross_attn(blk.norm3(x), ctx, self.ctx_lens)
                yf = blk.ffn(blk.norm2(x).float() * (1 + ec[4]) + ec[3])
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    x = x + yf * ec[5]

            return model.unpatchify(model.head(x, e), grid)[0]

    def _commit(self, n_frames):
        self.cache.commit(n_frames * self.tokens_per_frame)
        self.n_frames += n_frames
        if self.window_frames is not None:
            # Persistent world + sliding event window: the world block is pinned,
            # so eviction drops the OLDEST EVENTS. Bounds both memory and the
            # per chunk attention cost, which is what makes throughput flat in
            # stream length rather than degrading.
            protect = self.n_world * self.tokens_per_frame
            budget = (self.n_world + self.window_frames) * self.tokens_per_frame
            excess = self.cache.num_tokens - budget
            if excess > 0:
                self.cache.evict_front(excess, protect=protect)

    @torch.no_grad()
    def set_world(self, world_latents, t_frac=0.0):
        """Prime the cache from clean world latents, attended bidirectionally."""
        self.cache.reset()
        self.n_frames = 0
        z = world_latents.unsqueeze(0).to(self.device, self.dtype)
        F = z.shape[2]
        assert F <= self.max_frames
        self.chunk_forward(z, t_frac, t_start=0, use_cache=True)
        self._commit(F)
        self.n_world = F
        # Per channel moments of the world act as the in dist reference
        # for latent_norm (see _renorm). Computed over (F, H, W) per channel.
        w = world_latents.float()
        self.ref_stats = (w.mean(dim=(1, 2, 3), keepdim=True).unsqueeze(0),
                          w.std(dim=(1, 2, 3), keepdim=True).unsqueeze(0))
        return F

    def _renorm(self, z, strength):
        """Pull a finished chunk's per-channel moments back to the world's.

        Autoregressive rollout with a model that was never trained on its own
        outputs accumulates error, and here that error is measurably a SCALE
        divergence: with the original bidirectional weights, generated latent std
        grows ~2x over 20 chunks and reaches 3x the world's, at which point the
        VAE's input range is exceeded and frames decode to flat grey
        (contrast 0.018 vs the world's 0.194 -- out_sweep/round1).

        Matching the first two moments per channel is the cheapest correction that
        targets exactly that failure, and it is applied BEFORE the chunk's K/V is
        recomputed and committed, so the cached history stays in-distribution and
        the correction cannot accumulate. strength blends: 0 = off, 1 = full
        moment match. It cannot fix a wrong direction, only a wrong scale, so if
        content quality is the problem this will not rescue it.
        """
        if strength <= 0 or self.ref_stats is None:
            return z
        mu, sd = self.ref_stats
        zm = z.mean(dim=(2, 3, 4), keepdim=True)
        zs = z.std(dim=(2, 3, 4), keepdim=True)
        z_n = (z - zm) / (zs + 1e-5) * sd.to(z.dtype) + mu.to(z.dtype)
        return (1.0 - strength) * z + strength * z_n


    @torch.no_grad()
    def generate_chunk(self, chunk_frames=4, num_steps=3, generator=None,
                       sampler='dpm', shift=5.0, guidance=None, ctx_neg=None,
                       rope_start=None, t_max=1.0, anchor=None, latent_norm=0.0):
        """Emit one event chunk autoregressively. Returns (latents, seconds).

        t_max < 1.0 warm-starts the chunk instead of denoising from pure noise:
            z_init = (1 - t_max) * anchor + t_max * noise
        where `anchor` is the last clean latent frame, broadcast over the chunk.
        This is SDEdit-style continuation. It is worth doing here because
        diag/context_len_probe.py measured this model's velocity error at 0.33 for
        t=0.8 versus 0.14 for t=0.5 -- denoising from t=1 integrates through the
        region where it is least accurate. Trade-off: lower t_max means less
        motion diversity, since the chunk starts closer to the previous frame.
        """
        t_start = self.n_frames if rope_start is None else rope_start
        if t_start + chunk_frames > self.max_frames:
            raise RuntimeError(f'temporal index {t_start + chunk_frames} exceeds '
                               f'max_frames {self.max_frames}')
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        noise = torch.randn(1, 16, chunk_frames, self.h_lat, self.w_lat,
                            device=self.device, dtype=torch.float32,
                            generator=generator)
        if t_max < 1.0:
            if anchor is None:
                raise ValueError('t_max < 1.0 requires an anchor latent frame')
            a = anchor.to(self.device, torch.float32).unsqueeze(0)
            if a.shape[2] != chunk_frames:
                a = a[:, :, -1:].expand(-1, -1, chunk_frames, -1, -1)
            z = (1 - t_max) * a + t_max * noise
        else:
            z = noise

        def predict(z_in, t_frac):
            v = self.chunk_forward(z_in, t_frac, t_start).float().unsqueeze(0)
            if guidance and ctx_neg is not None:
                vu = self.chunk_forward(z_in, t_frac, t_start,
                                        ctx=ctx_neg).float().unsqueeze(0)
                v = vu + guidance * (v - vu)
            return v

        if sampler == 'euler':
            ts = torch.linspace(t_max, 0.0, num_steps + 1)
            for i in range(num_steps):
                tc, tn = float(ts[i]), float(ts[i + 1])
                z = z + (tn - tc) * predict(z, tc)
        elif sampler == 'dpm':
            sch = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            sig = get_sampling_sigmas(num_steps, shift)
            if t_max < 1.0:                     # rescale sigmas into [0, t_max]
                sig = sig * t_max               # keep it a numpy array
            tsteps, _ = retrieve_timesteps(sch, device=self.device, sigmas=sig)
            for tv in tsteps:
                v = predict(z, timestep_to_train_scale(tv))
                z = sch.step(v, tv, z, return_dict=False)[0]
        else:
            raise ValueError(sampler)

        z = self._renorm(z, latent_norm)

        # recompute the finished chunk's K/V at t=0 so history is clean, then commit
        self.chunk_forward(z, 0.0, t_start)
        self._commit(chunk_frames)

        torch.cuda.synchronize()
        return z[0], time.perf_counter() - t0

    @torch.no_grad()
    def stream(self, num_chunks, chunk_frames=4, num_steps=3, seed=0,
               log_every=5, anchor=None, world_anchor=None, world_anchor_weight=0.0,
               **kw):
        """world_anchor_weight w > 0 blends a fixed WORLD reference frame into the
        warm start anchor:  anchor = (1-w)·last_generated + w·world_ref.

        Rationale: pinning the worlds K/V stabilises attention but the warm-start
        anchor still forms a chain (each chunk starts from the previous chunks
        output), so errors compound and long streams drift. v0.3 casts W as the
        persistent carrier of scene and character identity, so letting W enter the
        initialisation too is the faithful reading, not just a hack.
        """
        g = torch.Generator(device=self.device).manual_seed(seed)
        lats, times = [], []
        for u in range(num_chunks):
            lat, dt = self.generate_chunk(chunk_frames=chunk_frames,
                                          num_steps=num_steps, generator=g,
                                          anchor=anchor, **kw)
            anchor = lat[:, -1:]        # newest clean frame anchors the next chunk
            if world_anchor_weight > 0 and world_anchor is not None:
                w = world_anchor_weight
                anchor = (1 - w) * anchor + w * world_anchor.to(anchor.device,
                                                                anchor.dtype)
            lats.append(lat)
            times.append(dt)
            if log_every and (u + 1) % log_every == 0:
                print(f'  chunk {u+1}/{num_chunks}: {dt*1000:7.1f} ms '
                      f'({dt/chunk_frames*1000:6.1f} ms/latent-frame, '
                      f'cache {self.cache.num_tokens} tok)')
        return torch.cat(lats, dim=1), times
