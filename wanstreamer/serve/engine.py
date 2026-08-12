"""Session engine for the browser demo.

This is a thin serving layer over the project's own inference core: the student is
rolled out by `wanstreamer.stream.FewStepStreamer`, which is the code the checkpoint
was distilled and measured under. Nothing here reimplements the streaming maths --
the block loop, the block-causal K/V cache, the renoise sampler and `latent_norm`
all live in the core, and this module only drives them and turns latents into JPEG.

One GPU, one stream. A background worker generates blocks and pushes frames into a
bounded queue. The queue is deliberately short (~1.5 s) because it *is* the steering
latency (anything a viewer has already been sent cannot be steered any more).

Two controls, doing genuinely different things:

  steer  -- swap the cross-attention conditioning, keep the K/V cache. The scene
            continues. Cheap, instant, no reload.
  scene  -- tear the stream down and reopen from another world. A cut.
"""

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..stream import FewStepStreamer
from .paths import WORLDS_DIR
from .streamdecode import StreamingVAEDecoder
from .conditioning import PromptBank, TextEncoder
from .worldgen import WorldGenerator

WORLDS = [0, 44, 60, 82]  # the four shipped caches; generated ones get ids from 1000
GENERATED_BASE_ID = 1000
FPS = 16

# Defaults matching the published command in the model card. `shift=1.0` is the
# uniform few-step spacing the student was distilled under (see
# wanstreamer.stream.few_step_schedule); `window` is in LATENT FRAMES, not blocks.
DEFAULTS = dict(block=3, steps=2, window=6, latent_norm=1.0, shift=1.0,
                sampler="renoise")


def _encode_jpeg(arr, quality=88):
    import cv2

    ok, buf = cv2.imencode(".jpg", arr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


class StreamExhausted(RuntimeError):
    pass


@dataclass
class Status:
    state: str = "idle"  # idle | loading | generating | streaming | error
    detail: str = ""
    world: int | None = None
    prompt: str = ""
    prompt_source: str = ""  # "bank" | "text"
    stats: dict = field(default_factory=dict)
    error: str = ""
    progress: float | None = None  # 0..1 while generating a world


class Engine:
    def __init__(self, assets, wan_repo, base_dir, weights=None, device="cuda",
                 jpeg_quality=88, buffer_seconds=1.5, worlds_dir=None,
                 allow_worldgen=True, size=(640, 368), compile_vae=True):
        self.assets = Path(assets)
        self.wan_repo = Path(wan_repo)
        self.base_dir = Path(base_dir)
        self.weights = Path(weights or self.assets / "checkpoints/t14b_b64/latest.pt")
        self.worlds_dir = Path(worlds_dir or WORLDS_DIR)
        self.allow_worldgen = allow_worldgen
        self.size = size
        self.device = device
        self.jpeg_quality = jpeg_quality
        self.compile_vae = compile_vae
        self.maxframes = int(buffer_seconds * FPS)

        self.worlds = {}
        self.worldgen = None
        self.status = Status()
        self.lock = threading.Lock()
        self._genlock = threading.Lock()
        self.frames = queue.Queue(maxsize=self.maxframes)
        self._worker = None
        self._stop = threading.Event()
        self._pending_prompt = None

        self.model = self.decoder = self.streamer = None
        self._cur_emb = None
        self.bank = self.encoder = None
        self.step = None
        self.cfg = None
        self._loaded = False
        self._timings = {}
        self._frames_emitted = 0
        self._blocks = 0

    # ------------------------------------------------------------------ load

    def load(self, progress=None):
        def say(msg):
            self.status.state, self.status.detail = "loading", msg
            if progress:
                progress(msg)

        say("reading the prompt bank")
        self.bank = PromptBank(self.assets / "data/prompts.pt")

        say(f"loading the student ({self.weights.name})")
        self.model, self.cfg, self.step = self._load_student()

        say("loading the Wan2.1 VAE")
        torch.backends.cudnn.benchmark = True
        self.decoder = StreamingVAEDecoder(
            self.base_dir / "Wan2.1_VAE.pth", self.wan_repo, self.device
        )

        if self.compile_vae:
            say("compiling the decoder (one-off, ~40 s)")
            try:
                import torch._dynamo as dynamo

                dynamo.config.recompile_limit = 64
                self.decoder.model.decoder = torch.compile(
                    self.decoder.model.decoder, dynamic=False)
                w = torch.load(self.assets / "out/world_p60.pt", map_location="cpu",
                               weights_only=False)
                self.decoder.decode(w["latents"][:, :6].to(self.device))
                self.decoder.reset()
            except Exception as e:  # an optimisation, never a requirement
                self.status.detail = f"decoder compile skipped: {e}"

        tok = self.base_dir / "umt5-tokenizer"
        self.encoder = TextEncoder(
            self.base_dir / "models_t5_umt5-xxl-enc-bf16.pth",
            tok if tok.exists() else "google/umt5-xxl", self.wan_repo, self.device,
        )
        if self.allow_worldgen:
            self.worldgen = WorldGenerator(
                self.base_dir / "diffusion_pytorch_model.safetensors",
                self.cfg, self.device,
            )
        self._index_worlds()
        self._loaded = True
        self.status.state, self.status.detail = "idle", "ready"

    def _load_student(self):
        """Build the stock WanModel and load the distilled weights into it.

        Mirrors scripts/demo.py: a checkpoint that silently half-loaded would report
        base-model quality as if it were trained, so every parameter is checked.
        """
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel
        from safetensors.torch import load_file

        cfg = WAN_CONFIGS["t2v-1.3B"]
        m = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                     num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                     window_size=cfg.window_size, qk_norm=True,
                     cross_attn_norm=True, eps=1e-6)
        m.load_state_dict(
            load_file(str(self.base_dir / "diffusion_pytorch_model.safetensors")),
            strict=True)
        sd = torch.load(self.weights, map_location="cpu", weights_only=False)
        src = sd.get("model", sd)
        res = m.load_state_dict(src, strict=False)
        got = {n for n, _ in m.named_parameters()} - set(res.missing_keys)
        if len(got) != len(list(m.named_parameters())) or res.unexpected_keys:
            raise RuntimeError(
                f"bad checkpoint load: {len(res.missing_keys)} missing, "
                f"{len(res.unexpected_keys)} unexpected")
        return m.to(self.device).eval().requires_grad_(False), cfg, sd.get("step")

    # ---------------------------------------------------------------- worlds

    def _index_worlds(self):
        self.worlds = {
            w: {"path": self.assets / f"out/world_p{w}.pt", "prompt": self.bank.texts[w],
                "generated": False, "seconds": None}
            for w in WORLDS
        }
        self.worlds_dir.mkdir(parents=True, exist_ok=True)
        for meta_path in sorted(self.worlds_dir.glob("world_*.json")):
            try:
                meta = json.loads(meta_path.read_text())
                pt = meta_path.with_suffix(".pt")
                if pt.exists():
                    self.worlds[int(meta["id"])] = {
                        "path": pt, "prompt": meta.get("prompt", ""),
                        "generated": True, "seconds": meta.get("seconds")}
            except Exception:
                continue  # a half-written world should not stop the server booting

    def _next_world_id(self):
        used = [i for i in self.worlds if i >= GENERATED_BASE_ID]
        return max(used) + 1 if used else GENERATED_BASE_ID

    @property
    def worldgen_available(self):
        return bool(self.worldgen and self.worldgen.available)

    def generate_world(self, text=None, idx=None, steps=30, seed=0, guide=5.0):
        """Make a new opening world from text (or a bank prompt) with the base model.

        Slow (~52 s at 30 steps) and the only non-real-time step in the project.
        """
        if not self.worldgen_available:
            raise RuntimeError(
                "world generation is unavailable — the Wan2.1 base transformer "
                "(diffusion_pytorch_model.safetensors) is not present")
        if not self._genlock.acquire(blocking=False):
            raise RuntimeError("already generating a world")
        try:
            self._stop_worker()
            pos, label, source = self.resolve_prompt(idx, text)
            self.status = Status(state="generating", detail="encoding the prompt",
                                 prompt=label, prompt_source=source, progress=0.0)

            def on_step(i, n):
                self.status.progress = i / n
                self.status.detail = f"denoising the world — step {i} of {n}"

            lat, secs = self.worldgen.generate(
                pos, self.bank.neg.float(), size=self.size, steps=steps, seed=seed,
                guide=guide, progress=on_step)

            self.status.detail, self.status.progress = "decoding", 1.0
            self.decoder.reset()
            pixels = self.decoder.decode(lat)
            self.decoder.reset()

            wid = self._next_world_id()
            path = self.worlds_dir / f"world_{wid}.pt"
            # keep the conditioning with the world: reopening needs no text encoder
            torch.save({"latents": lat.cpu(), "pixels": pixels,
                        "prompt_emb": pos.cpu(), "base_seconds": secs}, path)
            path.with_suffix(".json").write_text(json.dumps(
                {"id": wid, "prompt": label, "steps": steps, "seed": seed,
                 "guide": guide, "seconds": round(secs, 1)}, indent=1))
            self.worlds[wid] = {"path": path, "prompt": label, "generated": True,
                                "seconds": round(secs, 1)}
            self.status = Status(state="idle", detail="world ready", prompt=label,
                                 prompt_source=source)
            return wid
        except Exception as e:
            self.status = Status(state="error", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            self._genlock.release()

    def delete_world(self, wid):
        w = self.worlds.get(int(wid))
        if not w or not w["generated"]:
            raise ValueError("only generated worlds can be deleted")
        Path(w["path"]).unlink(missing_ok=True)
        Path(w["path"]).with_suffix(".json").unlink(missing_ok=True)
        self.worlds.pop(int(wid), None)



    def resolve_prompt(self, idx=None, text=None):
        """-> (embedding [1, 512, 4096], label, source)"""
        if text:
            text = text.strip()
            if not text:
                raise ValueError("empty prompt")
            return self.encoder.encode(text), text, "text"
        if idx is None:
            raise ValueError("need a prompt index or text")
        idx = int(idx)
        if not 0 <= idx < len(self.bank):
            extra = ""
            if idx >= GENERATED_BASE_ID:
                extra = (f" — {idx} looks like a generated world id, not a prompt. "
                         "Generated worlds carry their own conditioning: pass no "
                         "prompt_idx/prompt_text and it will be used.")
            raise ValueError(f"prompt index must be 0-{len(self.bank)-1}{extra}")
        return self.bank.embedding(idx), self.bank.texts[idx], "bank"

    def _set_text(self, emb):
        """Push raw umt5 conditioning into the streamer, embedded as the core wants."""
        emb = emb.to(self.device)
        with torch.amp.autocast("cuda", enabled=False):
            ctx = self.model.text_embedding(emb.float())
        self.streamer.set_text(ctx)
        self._cur_emb = emb   # kept so a crossfade can interpolate from it



    def start(self, world, idx=None, text=None, seed=0, steps=None, block=None,
              window=None, latent_norm=None, shift=None, sampler=None):
        if not self._loaded:
            raise RuntimeError("engine not loaded")
        world = int(world)
        entry = self.worlds.get(world)
        if entry is None:
            raise ValueError(f"unknown world {world}; have {sorted(self.worlds)}")

        block = block or DEFAULTS["block"]
        steps = steps or DEFAULTS["steps"]
        window = window or DEFAULTS["window"]
        shift = DEFAULTS["shift"] if shift is None else shift
        sampler = sampler or DEFAULTS["sampler"]
        latent_norm = DEFAULTS["latent_norm"] if latent_norm is None else latent_norm

        with self.lock:
            self._stop_worker()
            self.status = Status(state="loading", detail="opening the world", world=world)
            w = torch.load(entry["path"], map_location="cpu", weights_only=False)

            emb = label = source = None
            if idx is None and text is None:
                if not entry["generated"]:
                    idx = world
                elif w.get("prompt_emb") is not None:
                    emb, label, source = w["prompt_emb"].float(), entry["prompt"], "text"
                else:
                    text = entry["prompt"]
            if emb is None:
                emb, label, source = self.resolve_prompt(idx, text)
            self.status.prompt, self.status.prompt_source = label, source

            world_lat = w["latents"]
            nw = world_lat.shape[1]
            self.streamer = None
            torch.cuda.empty_cache()
            self.streamer = FewStepStreamer(
                self.model, width=self.size[0], height=self.size[1],
                max_frames=1024, device=self.device, dtype=self.cfg.param_dtype,
                window_frames=window,
                # the world's K/V is pinned; the window bounds only the events after it
                cache_frames=nw + window + block + 2,
                block_frames=block, num_steps=steps, shift=shift, sampler=sampler,
            )
            self.streamer.latent_norm = latent_norm
            self._set_text(emb)

            self.decoder.reset()
            self._frames_emitted = self._blocks = 0
            self._pending_prompt = None
            self._gen = torch.Generator(device=self.device).manual_seed(int(seed))

            t0 = time.time()
            self.streamer.set_world(world_lat.to(self.device, torch.float32))
            pixels = self.decoder.decode(world_lat.to(self.device, torch.float32))
            self._timings = {"start_s": round(time.time() - t0, 3)}
            self._frames_emitted = pixels.shape[0]

            self._stop.clear()
            self._worker = threading.Thread(target=self._run, args=(pixels,),
                                            daemon=True, name="livewan-worker")
            self._worker.start()
            self.status.state, self.status.detail = "streaming", ""
        return self.status

    def steer(self, idx=None, text=None, crossfade=0):
        if self.streamer is None or self.status.state != "streaming":
            raise RuntimeError("no stream is running")
        emb, label, source = self.resolve_prompt(idx, text)
        self._pending_prompt = (emb, int(crossfade))
        self.status.prompt, self.status.prompt_source = label, source
        return self.status

    def stop(self):
        with self.lock:
            self._stop_worker()
            self.status = Status(state="idle", detail="stopped")

    def _stop_worker(self):
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=15)
        self._worker = None
        self._drain()

    def _drain(self):
        try:
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass

    def _push(self, pixels):
        arr = pixels.numpy()
        for i in range(arr.shape[0]):
            data = _encode_jpeg(arr[i], self.jpeg_quality)
            while not self._stop.is_set():
                try:
                    self.frames.put(data, timeout=0.25)
                    break
                except queue.Full:
                    continue
            if self._stop.is_set():
                return

    def _run(self, world_pixels):
        blend = None
        try:
            self._push(world_pixels)
            while not self._stop.is_set():
                if self._pending_prompt is not None:
                    emb, xf = self._pending_prompt
                    self._pending_prompt = None
                    if xf and self._cur_emb is not None:
                        blend = [self._cur_emb.clone(), emb.to(self.device), 0, xf]
                    else:
                        blend = None
                        self._set_text(emb)
                st = self.streamer
                if st.n_world + st.n_frames + st.block_frames > st.max_frames:
                    raise StreamExhausted(
                        f"stream reached the {st.max_frames}-latent-frame RoPE "
                        "ceiling; start a new stream")

                t0 = time.time()
                z = st.generate_block(generator=self._gen)
                t_gen = time.time()
                pixels = self.decoder.decode(z[0].float())
                self._blocks += 1
                self._frames_emitted += pixels.shape[0]
                self._timings = {
                    "gen_s": round(t_gen - t0, 4),
                    "decode_s": round(time.time() - t_gen, 4),
                    "total_s": round(time.time() - t0, 4),
                }
                self.status.stats = self.stats()

                if blend:
                    src, dst, i, n = blend
                    i += 1
                    if i >= n:
                        blend = None
                        self._set_text(dst)
                    else:
                        blend[2] = i
                        self._set_text(src * (1 - i / n) + dst * (i / n))
                self._push(pixels)
        except StreamExhausted as e:
            self.status.state, self.status.detail = "idle", str(e)
        except Exception as e:  # surface, never die silently
            self.status.state, self.status.error = "error", f"{type(e).__name__}: {e}"



    def stats(self):
        st = self.streamer
        return {
            "frames": self._frames_emitted,
            "blocks": self._blocks,
            "latent_frames": (st.n_world + st.n_frames) if st else 0,
            "latent_frames_max": st.max_frames if st else 0,
            "seconds": self._frames_emitted / FPS,
            "kv_mb": (st.cache.memory_bytes() / 1e6) if st else 0.0,
            **self._timings,
        }

    def info(self):
        return {
            "step": self.step,
            "weights": self.weights.name,
            "worlds": [
                {"idx": i, "prompt": w["prompt"], "generated": w["generated"],
                 "seconds": 81 / FPS, "gen_seconds": w["seconds"]}
                for i, w in sorted(self.worlds.items())
            ],
            "prompts": self.bank.catalogue(),
            "fps": FPS,
            "encoder_loaded": self.encoder.loaded if self.encoder else False,
            "worldgen": self.worldgen_available,
            "buffer_frames": self.maxframes,
            "defaults": DEFAULTS,
        }
