"""Prompt encoding with a content-addressed cache.

The T5 encoder (umt5-xxl) runs on CPU here to keep the GPU free, which costs
~40 s per prompt -- worth caching. The original cache key in run_streaming.py was
`abs(hash(prompt)) % 10**8`; Python salts `hash()` for str per process
(PYTHONHASHSEED), so the key changed on every invocation and the cache never hit.
That is why diag/ accumulated 19 ctx_*.pt files for ~3 distinct prompts. sha1 of
the text is stable across processes.
"""
import gc
import hashlib
import os

import torch


def prompt_key(prompt):
    return hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:16]


def encode_prompt(prompt, neg_prompt, ckpt_dir, cfg, cache_dir, text_len=512):
    """Return (pos, neg) embeddings [1, text_len, dim], cached on disk."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'ctx_{prompt_key(prompt)}.pt')
    if os.path.exists(path):
        return torch.load(path, map_location='cpu')

    from wan.modules.t5 import T5EncoderModel
    print(f'Loading T5 (CPU) to encode prompt -> {os.path.basename(path)} ...')
    t5 = T5EncoderModel(text_len=text_len, dtype=cfg.t5_dtype,
                        device=torch.device('cpu'),
                        checkpoint_path=f'{ckpt_dir}/{cfg.t5_checkpoint}',
                        tokenizer_path=f'{ckpt_dir}/google/umt5-xxl')

    def enc(p):
        x = t5([p], torch.device('cpu'))[0].float()
        if x.shape[0] < text_len:
            x = torch.cat([x, torch.zeros(text_len - x.shape[0], x.shape[1])], 0)
        return x[:text_len].unsqueeze(0)

    blob = {'pos': enc(prompt), 'neg': enc(neg_prompt), 'prompt': prompt}
    del t5
    gc.collect()
    torch.save(blob, path)
    return blob
