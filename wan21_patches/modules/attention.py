# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# MODIFIED for Blackwell (sm_100) / no-flash-attn operation.
#
# Why: every attention call site in this repo (model.py:149/179/220/222,
# clip.py:85/197, streaming_blocks.py) calls `flash_attention`, NOT `attention`.
# Upstream `flash_attention` ends in `assert FLASH_ATTN_2_AVAILABLE`, so with no
# flash_attn wheel (none exists for sm_100) every forward pass raised a bare
# AssertionError. The SDPA fallback that upstream put in `attention()` had zero
# callers and was dead code.
#
# Fix: `flash_attention` now dispatches to a correct SDPA implementation when no
# flash-attn is installed. Unlike upstream's `attention()`, this fallback does NOT
# discard q_lens/k_lens -- it materialises an explicit boolean mask whenever the
# lengths imply real padding. Silently attending over padding was the specific
# "runs fine, quietly wrong" failure mode we had to rule out.
import os
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
    'sdpa_attention',
    'sdpa_backend_ctx',
]

# --- SDPA backend selection -------------------------------------------------
# cuDNN attention is not an alternative to SDPA; it is one of SDPA's backends.
# Measured on this machine (B200 / sm_100, torch 2.13.0+cu130, cuDNN 9.2.0),
# bf16, 12 heads x 128 dim, time per call:
#
#   shape (Lq x Lk)         FLASH    MEM_EFF   CUDNN    MATH
#   7800 x 7800  (480x832)  0.967    2.258     0.281    OOM-ish (1.4 GiB)
#   4600 x 4600  (640x368)  0.353    0.820     0.117    4.955
#   7800 x  512  (cross)    0.074    0.161     0.030    0.922
#   1560 x 6240  (stream)   0.316    0.579     0.079    2.420
#   1560 x 6240 + bool mask   n/a    0.747     0.212    2.850
#
# So cuDNN is 3-4x faster than the flash backend here, and it is the only fused
# backend that accepts an explicit attn_mask (flash rejects non-null masks, which
# would otherwise drop the padded path to MATH at ~13x the cost).
#
# torch's default dispatch already prefers cuDNN at these shapes, but that is an
# implicit heuristic that varies by version and shape. Pin the order explicitly
# so the fast path is guaranteed rather than incidental. All backends stay
# enabled, so an unsupported shape degrades instead of raising.
#
# NOTE: because cuDNN was already the default, this pinning is a robustness
# measure, not a speedup -- do not book it as one.
_BACKEND_ORDER = ['CUDNN_ATTENTION', 'FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH']


def _build_backend_ctx():
    """Return a callable giving a context manager that pins SDPA backend order."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return lambda: nullcontext()

    # WAN_SDPA_BACKEND forces a single backend, for benchmarking/debugging only.
    # It disables all fallbacks, so e.g. WAN_SDPA_BACKEND=FLASH_ATTENTION will
    # raise "No available kernel" on any masked call. That is intended.
    override = os.environ.get('WAN_SDPA_BACKEND', '').strip().upper()
    order = [override] if override else _BACKEND_ORDER
    backends = [getattr(SDPBackend, n) for n in order if hasattr(SDPBackend, n)]
    if not backends:
        return lambda: nullcontext()

    # set_priority=True keeps every listed backend enabled but fixes the order.
    try:
        with sdpa_kernel(backends, set_priority=True):
            pass
    except TypeError:
        return lambda: sdpa_kernel(backends)
    return lambda: sdpa_kernel(backends, set_priority=True)


sdpa_backend_ctx = _build_backend_ctx()


def _needs_mask(lens, length):
    """True if `lens` describes anything shorter than the padded `length`."""
    if lens is None:
        return False
    return bool(torch.as_tensor(lens).min().item() < length)


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Correct SDPA replacement for flash_attn_varlen_func.

    Shapes match the flash-attn convention:
        q: [B, Lq, Nq, C1]   k: [B, Lk, Nk, C1]   v: [B, Lk, Nk, C2]
    Returns [B, Lq, Nq, C2] in q's original dtype.

    Semantics deliberately mirror flash-attn's varlen kernel:

    * `q_lens` / `k_lens` are HONOURED. Keys at index >= k_lens[b] are masked out;
      query rows at index >= q_lens[b] are zeroed on output. Upstream's
      `attention()` set attn_mask=None here and only warned, which computes
      attention over padding and is silently wrong.
    * `causal=True` uses BOTTOM-RIGHT alignment when Lq != Lk, matching
      flash-attn (query i sees keys 0 .. i + Lk - Lq). Note torch's
      `is_causal=True` is TOP-LEFT aligned instead, so the two disagree for
      Lq != Lk -- that discrepancy is exactly the streaming-KV-cache bug this
      project had to fix. For block-causal streaming where the K/V cache holds
      only past+current frames, the correct call is `causal=False`: causality is
      already enforced by cache contents, and a mask would wrongly serialise
      tokens *within* a frame.
    """
    assert tuple(window_size) == (-1, -1), (
        f'sliding-window attention (window_size={window_size}) has no SDPA '
        'fallback; it would be silently ignored.')

    b, lq, nq, _ = q.shape
    lk, nk = k.shape[1], k.shape[2]
    out_dtype = q.dtype

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    if q_scale is not None:
        q = q * q_scale

    # [B, L, N, C] -> [B, N, L, C]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # grouped-query attention: replicate k/v heads to match q heads
    if nq != nk:
        assert nq % nk == 0, f'Nq ({nq}) must be divisible by Nk ({nk})'
        rep = nq // nk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    mask_k = _needs_mask(k_lens, lk)
    mask_q = _needs_mask(q_lens, lq)

    if not mask_k and (not causal or lq == lk):
        # Fast path: no padding, and either no mask or a square causal mask
        # (square => top-left and bottom-right alignment coincide).
        with sdpa_backend_ctx():
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=causal,
                dropout_p=dropout_p, scale=softmax_scale)
    elif (mask_k and not causal
          and int(torch.as_tensor(k_lens).min()) > 0):
        # Key-padding only. The mask does not depend on the query index, so a
        # [B, 1, 1, Lk] mask is exactly equivalent to the [B, 1, Lq, Lk] one
        # built below and SDPA broadcasts it. Worth a special case: the block-
        # causal trainer batches blocks with differing prefix lengths, where the
        # dense form would be a 244 MiB bool tensor per attention call.
        kl = torch.as_tensor(k_lens, device=q.device).reshape(b, 1, 1, 1)
        keep = torch.arange(lk, device=q.device).reshape(1, 1, 1, lk) < kl
        with sdpa_backend_ctx():
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, is_causal=False,
                dropout_p=dropout_p, scale=softmax_scale)
    else:
        # Build an explicit boolean keep-mask [B, 1, Lq, Lk].
        keep = torch.ones(b, 1, lq, lk, dtype=torch.bool, device=q.device)

        if mask_k:
            kl = torch.as_tensor(k_lens, device=q.device).reshape(b, 1, 1, 1)
            key_idx = torch.arange(lk, device=q.device).reshape(1, 1, 1, lk)
            keep &= key_idx < kl

        if causal:
            # bottom-right aligned: key j visible to query i iff j <= i + (lk - lq)
            qi = torch.arange(lq, device=q.device).reshape(1, 1, lq, 1)
            kj = torch.arange(lk, device=q.device).reshape(1, 1, 1, lk)
            keep &= kj <= qi + (lk - lq)

        # A row with nothing visible would give NaN from softmax. Let such rows
        # attend to key 0, then zero the result below.
        dead = ~keep.any(dim=-1, keepdim=True)
        if dead.any():
            keep = keep | (dead & (torch.arange(lk, device=q.device) == 0).reshape(1, 1, 1, lk))

        with sdpa_backend_ctx():
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, is_causal=False,
                dropout_p=dropout_p, scale=softmax_scale)
        if dead.any():
            out = out.masked_fill(dead, 0.0)

    out = out.transpose(1, 2).contiguous()  # -> [B, Lq, Nq, C2]

    if mask_q:
        ql = torch.as_tensor(q_lens, device=out.device).reshape(b, 1, 1, 1)
        qi = torch.arange(lq, device=out.device).reshape(1, lq, 1, 1)
        out = out * (qi < ql)

    return out.to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        # No flash-attn (e.g. Blackwell sm_100): use the correct SDPA path.
        return sdpa_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
