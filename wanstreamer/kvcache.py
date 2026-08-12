"""Preallocated K/V cache for block-causal streaming.

Replaces streaming_blocks.KVFrameCache, which stored a deque of per-frame tensors
and rebuilt the context with `torch.cat` on every `get()`. The profile in
PROGRESS.md §6 counted 302 `aten::cat` calls per single frame-forward.

Here the cache is one preallocated buffer per layer and the context is a *view*,
so steady-state streaming does no allocation and no copying beyond writing the
new frame's K/V.

Commit semantics matter for multi-step denoising. The current chunk's K/V changes
at every denoising step, so it must not be committed until the chunk is final:

    for step in steps:                     # chunk still noisy
        cache.write(layer, k, v)           # scratch region at [len, len+S)
        ctx_k, ctx_v = cache.context(layer, S)   # past + current, one view
    cache.commit(S)                        # chunk finalised, becomes past

Because the cache only ever holds past + current frames, full attention over that
view IS block-causal -- bidirectional within the current chunk, unrestricted over
the past, and no future keys exist to leak. No mask is needed; see
tests/test_attention_fallback.py case 5.
"""
import torch


class StreamingKVCache:
    """Per-layer ring-free preallocated K/V store with an uncommitted scratch tail."""

    def __init__(self, num_layers, max_tokens, num_heads, head_dim,
                 device, dtype=torch.bfloat16, scratch_tokens=None):
        self.num_layers = num_layers
        self.max_tokens = max_tokens
        self.scratch = scratch_tokens if scratch_tokens is not None else max_tokens
        cap = max_tokens + self.scratch
        self.k = torch.zeros(num_layers, cap, num_heads, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros(num_layers, cap, num_heads, head_dim,
                             device=device, dtype=dtype)
        self.length = 0          # committed tokens
        self._pending = 0        # tokens written to scratch, not yet committed
        # Monotone count of eviction events. `BufferPrefixKV` reconstructs a
        # recorded step's prefix as `buffer[:upto]`, which is only that step's
        # actual prefix if nothing has been evicted since it was recorded --
        # eviction shifts surviving keys down and discards the oldest outright.
        # The self-forcing trainer compares this counter against the value it
        # stamped on each recorded block and refuses to take the gradient on a
        # block whose prefix has since moved. See stream.trim_to_window.
        self.evictions = 0

    def reset(self):
        self.length = 0
        self._pending = 0
        self.evictions = 0

    def write(self, layer, k, v):
        """Write the current (uncommitted) chunk's K/V at [length, length+S)."""
        s = k.shape[1]
        if self.length + s > self.k.shape[1]:
            raise RuntimeError(
                f'K/V cache overflow: {self.length} committed + {s} pending > '
                f'capacity {self.k.shape[1]}. Raise max_tokens or evict.')
        self.k[layer, self.length:self.length + s].copy_(k[0])
        self.v[layer, self.length:self.length + s].copy_(v[0])
        self._pending = s

    def context(self, layer, s):
        """View of past + current: ([1, length+s, n, d], same for v). No copy."""
        end = self.length + s
        return (self.k[layer, :end].unsqueeze(0),
                self.v[layer, :end].unsqueeze(0))

    def commit(self, s):
        """Promote the pending chunk to committed history."""
        self.length += s
        self._pending = 0

    def evict_front(self, s, protect=0):
        """Drop `s` tokens from the front of the evictable region.

        `protect` pins the first `protect` tokens permanently -- used to keep the
        WORLD block resident while sliding a window over the event stream, which
        is the behaviour the papers describe (W is persistent, events stream).
        Eviction therefore removes the OLDEST EVENTS, not the world.
        """
        if s <= 0:
            return
        evictable = self.length - protect
        s = min(s, max(0, evictable))
        if s == 0:
            return
        keep = self.length - protect - s          # events surviving the eviction
        if keep > 0:
            src = protect + s
            self.k[:, protect:protect + keep].copy_(self.k[:, src:src + keep])
            self.v[:, protect:protect + keep].copy_(self.v[:, src:src + keep])
        self.length = protect + keep
        self.evictions += 1

    @property
    def num_tokens(self):
        return self.length

    def memory_bytes(self):
        return self.k.numel() * self.k.element_size() * 2
