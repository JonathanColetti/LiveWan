"""Streaming wrapper around the Wan2.1 VAE decoder.

The offline path (`scripts/demo.py`) decodes the whole latent sequence in one call at
the end of a run. A live stream cannot: it has to emit pixels every block. Stock
`WanVAE_.decode` clears its causal-conv feature cache on entry and exit, so calling it
once per block would restart the temporal convolutions and seam every 3 latent frames.
The decoder is already causal and already walks the sequence one latent frame at a
time ( the only thing between it and a continuous stream is that `clear_cache()`).
This keeps the cache alive across calls instead.

Frame arithmetic: the first latent frame of a stream decodes to 1 pixel frame, every
later one to 4 (temporal stride 4). So a 21-frame world -> 81 pixel frames, and each
subsequent 3-frame block -> 12 pixel frames = 750 ms at 16 fps.
"""

import torch


class StreamingVAEDecoder:
    def __init__(self, vae_path, wan_repo=None, device="cuda", dtype=torch.float16):
        from wan.modules.vae import WanVAE

        self.wrapper = WanVAE(vae_pth=str(vae_path), dtype=dtype, device=device)
        self.model = self.wrapper.model
        self.device, self.dtype = device, dtype
        self.scale = self.wrapper.scale
        self.reset()

    def reset(self):
        self.model.clear_cache()

    @torch.no_grad()
    def decode(self, z):
        """z: [C, F, H, W] latents -> uint8 pixels [T, H*8, W*8, 3] (RGB).

        Continues the previous call's temporal context; call `reset()` to cut.
        """
        z = z.to(self.device, torch.float32).unsqueeze(0).clamp(-4, 4)
        mean, inv_std = self.scale
        z = z / inv_std.view(1, -1, 1, 1, 1).float() + mean.view(1, -1, 1, 1, 1).float()

        with torch.amp.autocast("cuda", dtype=self.dtype):
            x = self.model.conv2(z)
            outs = []
            for i in range(x.shape[2]):
                self.model._conv_idx = [0]
                outs.append(self.model.decoder(
                    x[:, :, i:i + 1], feat_cache=self.model._feat_map,
                    feat_idx=self.model._conv_idx))
            out = torch.cat(outs, dim=2)

        out = out.float().clamp_(-1, 1).squeeze(0)  # [3, T, H, W]
        out = ((out.permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return out.cpu()

    def memory_bytes(self):
        return sum(t.numel() * t.element_size()
                   for t in (self.model._feat_map or []) if torch.is_tensor(t))
