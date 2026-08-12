"""Equivalence test for wanstreamer.core against a naive reference.

The optimised core changes three numerically-relevant things at once:
  * RoPE precomputed in float32 complex instead of rebuilt in float64 per call
  * K/V context is a view into a preallocated buffer instead of torch.cat
  * modulation chunks computed once per step instead of per (block x frame)

This checks it against a deliberately naive implementation that mirrors
streaming_blocks.py's structure (upstream float64 rope_apply + torch.cat cache),
using a small randomly-initialised WanModel so the test is fast.

Includes CONTROLS that must FAIL -- each reintroduces a real bug. Without them
the comparison could pass vacuously.

Run: PYTHONPATH=wan21_repo python tests/test_streaming_core.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wan21_repo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wan.modules.model import WanModel
from wan.modules.attention import flash_attention
from wanstreamer import (make_rope_table, make_cache, ModulationCache,
                         sequence_forward, latent_geometry)

DEV = 'cuda'
# Justified in tests/test_attention_fallback.py: bf16 max-abs/mean-abs residual
# for a correct implementation sits at ~2-3e-2; real bugs land 50x+ higher.
TOL = 6e-2
FAILURES = []


def ref_rope(x, f, h, w, freqs, t_offset):
    """Upstream rope_apply, float64, with a temporal offset."""
    n, c = x.size(2), x.size(3) // 2
    fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    out = []
    for i in range(x.shape[0]):
        seq = f * h * w
        xi = torch.view_as_complex(x[i, :seq].to(torch.float64).reshape(seq, n, -1, 2))
        fi = torch.cat([
            fs[0][t_offset:t_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq, 1, -1)
        out.append(torch.view_as_real(xi * fi).flatten(2))
    return torch.stack(out).float()


@torch.no_grad()
def reference_forward(model, latents, e, e0, freqs, ctx, ctx_lens, hp, wp,
                      rope_mode='abs', commit=True):
    """Naive per-frame causal forward with a torch.cat K/V cache."""
    B, C, F = latents.shape[0], latents.shape[1], latents.shape[2]
    cache = [None] * len(model.blocks)
    outs = []
    for fi in range(F):
        x = model.patch_embedding(latents[:, :, fi:fi + 1])
        grid = torch.stack([torch.tensor(x.shape[2:], dtype=torch.long, device=DEV)
                            for _ in range(B)])
        x = x.flatten(2).transpose(1, 2)
        t_off = fi if rope_mode == 'abs' else 0
        for li, blk in enumerate(model.blocks):
            with torch.amp.autocast('cuda', dtype=torch.float32):
                ec = (blk.modulation + e0).chunk(6, dim=1)
            sa_in = blk.norm1(x).float() * (1 + ec[1]) + ec[0]
            b, s = sa_in.shape[0], sa_in.shape[1]
            n = blk.num_heads
            d = blk.dim // n
            sa = blk.self_attn
            q = sa.norm_q(sa.q(sa_in)).view(b, s, n, d)
            k = sa.norm_k(sa.k(sa_in)).view(b, s, n, d)
            v = sa.v(sa_in).view(b, s, n, d)
            q = ref_rope(q, 1, hp, wp, freqs, t_off)
            k = ref_rope(k, 1, hp, wp, freqs, t_off)
            kk, vv = k.to(torch.bfloat16), v.to(torch.bfloat16)
            if cache[li] is not None:
                kk = torch.cat([cache[li]['k'], kk], dim=1)
                vv = torch.cat([cache[li]['v'], vv], dim=1)
            y = flash_attention(q=q.to(torch.bfloat16), k=kk, v=vv,
                                window_size=(-1, -1), causal=False)
            y = sa.o(y.flatten(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * ec[2]
            x = x + blk.cross_attn(blk.norm3(x), ctx, ctx_lens)
            yf = blk.ffn(blk.norm2(x).float() * (1 + ec[4]) + ec[3])
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + yf * ec[5]
            if commit:
                cache[li] = {'k': kk, 'v': vv}
        outs.append(model.unpatchify(model.head(x, e), grid)[0])
    return torch.cat(outs, dim=1)


def check(name, got, want, expect_close=True):
    got, want = got.float(), want.float()
    err = (got - want).abs().max().item() / want.abs().mean().clamp_min(1e-6).item()
    ok = (err <= TOL) == expect_close
    kind = 'must match' if expect_close else 'must DIFFER (control)'
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: rel_err={err:.4f} ({kind})')
    if not ok:
        FAILURES.append(name)


def main():
    torch.manual_seed(0)
    # small model, real code paths
    dim, heads, layers = 192, 4, 3
    model = WanModel(model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16,
                     dim=dim, ffn_dim=384, freq_dim=256, text_dim=4096, out_dim=16,
                     num_heads=heads, num_layers=layers, qk_norm=True,
                     cross_attn_norm=True, eps=1e-6).to(DEV).eval()
    for p in model.parameters():           # init_weights zeroes the head
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)

    # 640x368 target geometry
    h_lat, w_lat, hp, wp = latent_geometry(640, 368)
    print(f'geometry: 640x368 -> latent {h_lat}x{w_lat} -> patches {hp}x{wp} '
          f'= {hp*wp} tokens/frame')

    F = 4
    lat = torch.randn(1, 16, F, h_lat, w_lat, device=DEV, dtype=torch.bfloat16)
    ctx = torch.randn(1, 512, dim, device=DEV, dtype=torch.float32)
    ctx_lens = torch.tensor([512], device=DEV, dtype=torch.long)
    freqs = model.freqs.to(DEV)
    with torch.amp.autocast('cuda', enabled=False):
        e = torch.randn(1, dim, device=DEV)
        e0 = torch.randn(1, 6, dim, device=DEV)

    def new_impl(rope_max=None, commit=True, shuffle_mod=False):
        tbl = make_rope_table(model, hp, wp, rope_max or F, DEV)
        cache = make_cache(model, hp * wp, F, DEV)
        mod = ModulationCache(model.blocks, e0)
        if shuffle_mod:
            mod.chunks = list(reversed(mod.chunks))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            return sequence_forward(model, lat, e, mod, tbl, cache, ctx, ctx_lens,
                                    start_index=0, commit=commit)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        ref_abs = reference_forward(model, lat, e, e0, freqs, ctx, ctx_lens, hp, wp,
                                    rope_mode='abs')
        ref_zero = reference_forward(model, lat, e, e0, freqs, ctx, ctx_lens, hp, wp,
                                     rope_mode='zero')
        ref_nocommit = reference_forward(model, lat, e, e0, freqs, ctx, ctx_lens,
                                         hp, wp, rope_mode='abs', commit=False)

    print('\n1. Optimised core == naive reference (float64 RoPE + torch.cat cache)')
    check('sequence_forward vs reference', new_impl(), ref_abs)

    print('\n2. Controls -- each must differ from the correct reference')
    # temporal RoPE collapsed to index 0 == the streaming_blocks.py bug
    check('CONTROL rope temporal index 0', ref_zero, ref_abs, expect_close=False)
    # never committing means frames cannot see the past
    check('CONTROL cache never committed', new_impl(commit=False), ref_abs,
          expect_close=False)
    check('CONTROL reference, cache never committed', ref_nocommit, ref_abs,
          expect_close=False)
    # modulation chunks applied to the wrong blocks
    check('CONTROL modulation blocks reversed', new_impl(shuffle_mod=True), ref_abs,
          expect_close=False)

    print('\n3. Cache mechanics')
    tbl = make_rope_table(model, hp, wp, F, DEV)
    cache = make_cache(model, hp * wp, F, DEV)
    S = hp * wp
    assert cache.num_tokens == 0
    k = torch.randn(1, S, heads, dim // heads, device=DEV, dtype=torch.bfloat16)
    cache.write(0, k, k)
    ck, _ = cache.context(0, S)
    ok_view = ck.shape[1] == S and cache.num_tokens == 0
    print(f'  [{"PASS" if ok_view else "FAIL"}] uncommitted write visible in context '
          f'but not in length ({ck.shape[1]} tokens, length={cache.num_tokens})')
    if not ok_view:
        FAILURES.append('cache uncommitted semantics')
    cache.commit(S)
    ck2, _ = cache.context(0, S)
    ok_c = cache.num_tokens == S and ck2.shape[1] == 2 * S
    print(f'  [{"PASS" if ok_c else "FAIL"}] after commit, length={cache.num_tokens}, '
          f'next context={ck2.shape[1]} tokens')
    if not ok_c:
        FAILURES.append('cache commit semantics')
    # context must be a view, not a copy
    is_view = ck2.data_ptr() == cache.k[0].data_ptr()
    print(f'  [{"PASS" if is_view else "FAIL"}] context is a zero-copy view')
    if not is_view:
        FAILURES.append('cache zero-copy')
    # overflow must raise, not silently corrupt
    try:
        for _ in range(F + 4):
            cache.write(0, k, k)
            cache.commit(S)
        raised = False
    except RuntimeError:
        raised = True
    print(f'  [{"PASS" if raised else "FAIL"}] overflow raises instead of corrupting')
    if not raised:
        FAILURES.append('cache overflow')

    print()
    if FAILURES:
        print(f'FAILED ({len(FAILURES)}): ' + ', '.join(FAILURES))
        return 1
    print('All streaming-core checks passed (controls failed as required).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
