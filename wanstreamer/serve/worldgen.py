"""Generate new opening worlds from text with the Wan2.1-1.3B base model.

A "world" is just the 21 latent frames a stream is primed from. The four shipped
ones were produced this way once and cached (`base_seconds` in each file is how long
it took), so nothing stops any prompt from having one.

This is the base model, not the student: full bidirectional attention over the
whole sequence, CFG, and a real 30-50 step schedule. It is slow by design (~52 s at
30 steps on an A100-40GB) and is the only part of the project that is not real time.
The student then extends the result indefinitely at ~1x.
"""

import time
from pathlib import Path

import torch

LATENT_FRAMES = 21  # -> 81 pixel frames, the shipped world length


def _sampling_sigmas(steps, shift):
    """Wan's own shifted inference schedule -- correct for the *undistilled* base
    model, unlike the uniform few-step spacing the student is rolled out with."""
    import numpy as np

    s = np.linspace(1, 0, steps + 1)[:steps]
    return list(shift * s / (1 + (shift - 1) * s)) + [0.0]


class WorldGenerator:
    """Lazy holder for the base model. 5.7 GB on disk, ~2.6 GB resident in bf16."""

    def __init__(self, weights, cfg, device="cuda"):
        self.weights = Path(weights)
        self.cfg = cfg
        self.device = device
        self._model = None

    @property
    def loaded(self):
        return self._model is not None

    @property
    def available(self):
        return self.weights.exists()

    def load(self):
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                f"{self.weights} not found — world generation needs the Wan2.1 base "
                "transformer. Re-run setup.sh without SKIP_BASE=1.")
        from safetensors.torch import load_file
        from wan.modules.model import WanModel

        c = self.cfg
        m = WanModel(dim=c.dim, ffn_dim=c.ffn_dim, freq_dim=c.freq_dim,
                     num_heads=c.num_heads, num_layers=c.num_layers,
                     window_size=c.window_size, qk_norm=True,
                     cross_attn_norm=True, eps=1e-6)
        m.load_state_dict(load_file(str(self.weights)), strict=True)
        self._model = m.to(self.device, c.param_dtype).eval().requires_grad_(False)

    def unload(self):
        self._model = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(self, pos, neg, size=(640, 368), steps=30, guide=5.0, shift=5.0,
                 seed=0, progress=None):
        """pos/neg: [1, 512, 4096] umt5 embeddings -> latents [16, 21, h, w].

        `progress(i, steps)` fires after every step so the UI can show a real
        determinate bar rather than a spinner for a minute.
        """
        self.load()
        m = self._model
        dtype = self.cfg.param_dtype
        h, w = size[1] // 8, size[0] // 8
        seq_len = LATENT_FRAMES * (h // 2) * (w // 2)

        t0 = time.time()
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn((16, LATENT_FRAMES, h, w), generator=g,
                        device=self.device, dtype=torch.float32)
        ctx_p = [pos.to(self.device, dtype).squeeze(0)]
        ctx_n = [neg.to(self.device, dtype).squeeze(0)]

        sig = _sampling_sigmas(steps, shift)
        for i in range(steps):
            t = torch.full((1,), sig[i] * 1000.0, device=self.device)
            with torch.amp.autocast("cuda", dtype=dtype):
                vc = m([x.to(dtype)], t=t, context=ctx_p, seq_len=seq_len)[0].float()
                vu = m([x.to(dtype)], t=t, context=ctx_n, seq_len=seq_len)[0].float()
            # classifier free guidance: the student never needs this, the base does
            x = x + (sig[i + 1] - sig[i]) * (vu + guide * (vc - vu))
            if progress:
                progress(i + 1, steps)

        return x, time.time() - t0
