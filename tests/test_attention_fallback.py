"""Correctness tests for the SDPA attention fallback (Phase 2, item 5).

Verifies `sdpa_attention` against a naive float32 reference, and — critically —
includes CONTROLS that must fail: each control re-runs the same comparison
against the *buggy* behaviour we replaced, and asserts the test would have caught
it. Without the controls these checks could pass vacuously.

Run:  PYTHONPATH=wan21_repo python tests/test_attention_fallback.py
"""
import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wan21_repo'))
from wan.modules.attention import sdpa_attention, FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE

DEV = 'cuda'
FAILURES = []


def reference(q, k, v, k_lens=None, causal=False, align='bottom_right'):
    """Naive float32 attention. q:[B,Lq,N,C] k,v:[B,Lk,N,C] -> [B,Lq,N,C]."""
    b, lq, n, c = q.shape
    lk = k.shape[1]
    qf = q.float().transpose(1, 2)          # [B,N,Lq,C]
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    logits = (qf @ kf.transpose(-1, -2)) / (c ** 0.5)   # [B,N,Lq,Lk]

    keep = torch.ones(b, 1, lq, lk, dtype=torch.bool, device=q.device)
    if k_lens is not None:
        kl = torch.as_tensor(k_lens, device=q.device).reshape(b, 1, 1, 1)
        keep &= torch.arange(lk, device=q.device).reshape(1, 1, 1, lk) < kl
    if causal:
        qi = torch.arange(lq, device=q.device).reshape(1, 1, lq, 1)
        kj = torch.arange(lk, device=q.device).reshape(1, 1, 1, lk)
        offset = (lk - lq) if align == 'bottom_right' else 0
        keep &= kj <= qi + offset

    logits = logits.masked_fill(~keep, float('-inf'))
    probs = logits.softmax(dim=-1).nan_to_num(0.0)
    return (probs @ vf).transpose(1, 2)     # [B,Lq,N,C]


# Tolerance is on max_abs_error / mean_abs_value, a deliberately harsh metric.
# Measured on this machine for a correct implementation, varying only dtype:
#     bf16 2.1e-2 | fp16 2.9e-3 | fp32 1.2e-5
# i.e. the residual scales with mantissa bits (8 / 11 / 24) and is pure rounding,
# not a logic error. Real logic errors in the controls below land at 2.8 .. 197,
# so 5e-2 keeps ~50x separation between "bf16 noise" and "actually broken".
BF16_TOL = 5e-2


def check(name, got, want, tol=BF16_TOL, expect_close=True):
    got, want = got.float(), want.float()
    denom = want.abs().mean().clamp_min(1e-6)
    err = (got - want).abs().max().item() / denom.item()
    close = err <= tol
    ok = close == expect_close
    verdict = 'PASS' if ok else 'FAIL'
    kind = 'must match' if expect_close else 'must DIFFER (control)'
    print(f'  [{verdict}] {name}: rel_err={err:.4f} ({kind})')
    if not ok:
        FAILURES.append(name)
    return err


def mk(b, lq, lk, n=4, c=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    f = lambda l: torch.randn(b, l, n, c, generator=g, device=DEV, dtype=torch.bfloat16)
    return f(lq), f(lk), f(lk)


def dispatched_kernel(mask=None):
    """Name the CUDA kernel SDPA actually dispatches to, via the profiler."""
    from torch.profiler import profile, ProfilerActivity
    q = torch.randn(1, 12, 512, 128, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, 12, 1024, 128, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, 12, 1024, 128, device=DEV, dtype=torch.bfloat16)
    from wan.modules.attention import sdpa_backend_ctx
    try:
        with sdpa_backend_ctx():
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # warmup
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                torch.cuda.synchronize()
    except RuntimeError as e:
        # Only reachable via a hard WAN_SDPA_BACKEND override: e.g. forcing
        # FLASH_ATTENTION cannot serve a masked call. The pinned default order
        # always includes fallbacks, so this cannot happen in normal operation.
        return f'<unavailable for this forced backend: {str(e)[:40]}>'
    names = [e.key for e in prof.key_averages() if e.self_device_time_total > 0]
    hit = [n for n in names if any(s in n.lower() for s in ('cudnn', 'fmha', 'flash', 'attention', 'gemm'))]
    return (hit or names or ['<none>'])[0][:70]


def main():
    print(f'flash_attn available: FA2={FLASH_ATTN_2_AVAILABLE} FA3={FLASH_ATTN_3_AVAILABLE}')
    print(f'(fallback under test is active only when both are False)')
    forced = os.environ.get('WAN_SDPA_BACKEND', '(pinned order: cuDNN>flash>mem_eff>math)')
    print(f'SDPA backend: {forced}')
    print(f'  unmasked path dispatches to: {dispatched_kernel()}')
    m = torch.ones(1, 1, 512, 1024, dtype=torch.bool, device=DEV)
    print(f'  masked   path dispatches to: {dispatched_kernel(m)}\n')

    # ---- 1. plain full attention, no padding, no causal ----------------------
    print('1. Full attention, no padding, no mask')
    q, k, v = mk(2, 128, 128, seed=1)
    check('full attention vs reference', sdpa_attention(q, k, v), reference(q, k, v))

    # ---- 2. k_lens padding: THE bug PROMPT.md §4 warned about ---------------
    # Upstream `attention()` set attn_mask=None and only warned, so it attended
    # over padding. Here padding is filled with large garbage so ignoring the
    # mask cannot go unnoticed.
    print('\n2. k_lens padding honoured (PROMPT.md §4 caveat)')
    q, k, v = mk(2, 64, 96, seed=2)
    k_lens = torch.tensor([96, 40], device=DEV)
    k[1, 40:] = 30.0          # garbage in the padded region
    v[1, 40:] = -30.0
    check('padded attention vs masked reference',
          sdpa_attention(q, k, v, k_lens=k_lens),
          reference(q, k, v, k_lens=k_lens))
    # CONTROL: the old ignore-the-mask behaviour must NOT match the reference.
    check('CONTROL padding ignored (old behaviour)',
          sdpa_attention(q, k, v, k_lens=None),
          reference(q, k, v, k_lens=k_lens), expect_close=False)

    # ---- 3. square causal ---------------------------------------------------
    print('\n3. Square causal (Lq == Lk)')
    q, k, v = mk(2, 64, 64, seed=3)
    check('square causal vs reference',
          sdpa_attention(q, k, v, causal=True),
          reference(q, k, v, causal=True))
    # CONTROL: without the causal mask it must differ.
    check('CONTROL causal mask disabled',
          sdpa_attention(q, k, v, causal=False),
          reference(q, k, v, causal=True), expect_close=False)

    # ---- 4. non-square causal: alignment matters ---------------------------
    # flash-attn varlen uses BOTTOM-RIGHT alignment; torch is_causal=True uses
    # TOP-LEFT. For Lq != Lk these differ drastically, which is the streaming
    # KV-cache bug (q = current frame, k = past frames + current frame).
    print('\n4. Non-square causal alignment (Lq=32, Lk=128)')
    q, k, v = mk(1, 32, 128, seed=4)
    check('non-square causal is bottom-right aligned',
          sdpa_attention(q, k, v, causal=True),
          reference(q, k, v, causal=True, align='bottom_right'))
    # CONTROL: top-left alignment (what torch is_causal=True would give) must differ.
    check('CONTROL top-left alignment',
          sdpa_attention(q, k, v, causal=True),
          reference(q, k, v, causal=True, align='top_left'), expect_close=False)
    # Show what torch's own is_causal does, to document the discrepancy.
    tl = F.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2),
        is_causal=True).transpose(1, 2)
    check('torch is_causal=True IS top-left (documents the trap)',
          tl, reference(q, k, v, causal=True, align='top_left'))

    # ---- 5. block-causal streaming: correct call is causal=False -----------
    # q = current frame's S tokens; k/v = [past frames, current frame].
    # The cache holds no future keys, so full attention over it *is* block-causal:
    # bidirectional within the current frame, unrestricted over the past.
    print('\n5. Block-causal streaming equivalence (S=32 tokens/frame, 3 past frames)')
    S, P = 32, 3
    g = torch.Generator(device=DEV).manual_seed(5)
    f = lambda l: torch.randn(1, l, 4, 64, generator=g, device=DEV, dtype=torch.bfloat16)
    q_cur = f(S)
    k_all, v_all = f(S * (P + 1)), f(S * (P + 1))
    want = reference(q_cur, k_all, v_all)               # unmasked == block-causal
    check('streaming: causal=False == block-causal',
          sdpa_attention(q_cur, k_all, v_all, causal=False), want)
    # CONTROL: causal=True (what streaming_blocks.py did) must differ — it wrongly
    # serialises tokens inside the current frame.
    check('CONTROL causal=True serialises intra-frame tokens',
          sdpa_attention(q_cur, k_all, v_all, causal=True), want, expect_close=False)

    # ---- 6. grouped-query attention ---------------------------------------
    print('\n6. Grouped-query attention (Nq=8, Nk=2)')
    g = torch.Generator(device=DEV).manual_seed(6)
    q = torch.randn(1, 48, 8, 64, generator=g, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, 48, 2, 64, generator=g, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, 48, 2, 64, generator=g, device=DEV, dtype=torch.bfloat16)
    check('GQA vs reference with expanded heads',
          sdpa_attention(q, k, v),
          reference(q, k.repeat_interleave(4, dim=2), v.repeat_interleave(4, dim=2)))

    print()
    if FAILURES:
        print(f'FAILED ({len(FAILURES)}): ' + ', '.join(FAILURES))
        return 1
    print('All attention-fallback checks passed (controls failed as required).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
