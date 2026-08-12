"""Prompt conditioning for the demo: the 96 prompt bank, plus free text encoding.

Two sources of conditioning, deliberately kept distinct:

* **The bank** (`data/prompts.pt`) -- 96 umt5-xxl embeddings that every clip in the
  project was generated and trained under. Zero cost, and exactly the conditioning
  the published numbers refer to.
* **Free text** -- encoded here, on this machine, with umt5-xxl (11.4 GB, loaded
  lazily on first use). Note that umt5 embeddings are mildly hardware-dependent, so
  text encoded here is not numerically identical to what the same string would give
  on the training box. It looks fine; it just isn't the *same* conditioning, so
  free-text results are not strictly comparable to the bank's published metrics.

Both paths end in the same shape: [1, 512, 4096], zero padded, which is what
`WanModel` expects.
"""

from pathlib import Path

import torch

TEXT_LEN = 512

THEMES = [
    ("People", 0, 40),
    ("Animals", 40, 56),
    ("Nature", 56, 72),
    ("City", 72, 86),
    ("Objects", 86, 96),
]


def theme_of(idx):
    for name, lo, hi in THEMES:
        if lo <= idx < hi:
            return name
    return "Other"


class PromptBank:
    def __init__(self, path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.texts = d["prompts"]
        self.pos = d["pos"]  # [96, 512, 4096] fp16
        self.neg = d["neg"]
        self.neg_prompt = d["neg_prompt"]

    def __len__(self):
        return len(self.texts)

    def embedding(self, idx):
        return self.pos[idx : idx + 1].float()

    def catalogue(self):
        return [
            {"idx": i, "text": t, "theme": theme_of(i)} for i, t in enumerate(self.texts)
        ]


class TextEncoder:
    """Lazy umt5-xxl. 11.4 GB -- only loaded if someone actually types a prompt."""

    def __init__(self, checkpoint, tokenizer_path, wan_repo, device="cuda"):
        self.checkpoint = str(checkpoint)
        self.tokenizer_path = str(tokenizer_path)
        self.wan_repo = str(wan_repo)
        self.device = device
        self._model = None

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        if self._model is not None:
            return
        from wan.modules.t5 import T5EncoderModel

        self._model = T5EncoderModel(
            text_len=TEXT_LEN, dtype=torch.bfloat16, device=self.device,
            checkpoint_path=self.checkpoint, tokenizer_path=self.tokenizer_path)

    @torch.no_grad()
    def encode(self, text):
        """-> [1, 512, 4096] float32, zero-padded exactly as the bank is."""
        self.load()
        ctx = self._model([text], self.device)[0]  # [L, 4096], L <= 512
        out = torch.zeros(TEXT_LEN, ctx.shape[1], dtype=torch.float32, device=ctx.device)
        out[: ctx.shape[0]] = ctx.float()
        return out.unsqueeze(0).cpu()

    def unload(self):
        self._model = None
        torch.cuda.empty_cache()
