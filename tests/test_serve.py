"""Checks for the serving layer (wanstreamer.serve).

The streaming maths is covered by tests/test_streaming_core.py; this file only
checks what the demo adds on top -- streaming decode, session lifecycle, steering
without a cache reset, and generated worlds.

Run with the demo server stopped (this wants the GPU to itself):

    python tests/test_serve.py
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from wanstreamer.serve import paths
from wanstreamer.serve.engine import GENERATED_BASE_ID, WORLDS, Engine

A = Path(paths.ASSETS)
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


def sharp(v):
    g = v.astype(np.float32).mean(-1)
    return np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()


def drain(engine, n):
    """Pull n JPEG frames off the queue and decode them to an array."""
    import cv2

    out = []
    while len(out) < n:
        try:
            out.append(engine.frames.get(timeout=20))
        except Exception:
            break
    return np.stack([cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
                     for f in out])


print("loading (this builds the student and the VAE)...")
tmp = Path(tempfile.mkdtemp())
eng = Engine(A, paths.WAN_REPO, paths.BASE_DIR, worlds_dir=tmp, compile_vae=False)
eng.load()
check("checkpoint is step 3000", eng.step == 3000, f"step={eng.step}")
check("defaults match the distilled config",
      eng.info()["defaults"]["sampler"] == "renoise"
      and eng.info()["defaults"]["shift"] == 1.0
      and eng.info()["defaults"]["window"] == 6,
      str(eng.info()["defaults"]))

try:
    print("\n[1] streaming VAE decode reproduces each cached world")
    # A fresh decode is not bit-exact against the stored pixels and the gap grows with
    # detail (world 82 is the chain-link fence). A *systematic* shift is what would
    # signal a bug, so the bias is checked too, not just the magnitude.
    for w in WORLDS:
        d = torch.load(A / f"out/world_p{w}.pt", map_location="cpu", weights_only=False)
        eng.decoder.reset()
        px = eng.decoder.decode(d["latents"].to(eng.device, torch.float32)).numpy()
        diff = px.astype(float) - d["pixels"].numpy().astype(float)
        check(f"world {w} decode", px.shape == tuple(d["pixels"].shape)
              and np.abs(diff).mean() < 6.0 and abs(diff.mean()) < 1.0,
              f"MAE={np.abs(diff).mean():.2f} bias={diff.mean():+.2f}")

    print("\n[2] every world streams without collapsing")
    for w in WORLDS:
        eng.start(w)
        v = drain(eng, 81 + 24)[81:]          # skip the world, keep 2 blocks
        ok = (np.isfinite(v).all() and v.std() > 15 and 20 < v.mean() < 235)
        check(f"world {w} streams", ok, f"mean={v.mean():.0f} std={v.std():.0f}")
        eng.stop()

    print("\n[3] the world's K/V is pinned, the event window is bounded")
    eng.start(60)
    drain(eng, 81 + 12)
    st = eng.streamer
    nw = st.n_world
    check("world frames pinned", nw == 21, f"n_world={nw}")
    drain(eng, 12 * 8)
    held = st.cache.num_tokens // st.S
    check("cache bounded by world + window", held <= nw + st.window_frames + st.block_frames,
          f"{held} frames held, world {nw} + window {st.window_frames}")
    check("world still pinned after eviction", st.n_world == nw)

    print("\n[4] steering keeps the cache (scene continues, no cut)")
    before_frames = eng._frames_emitted
    before_latent = st.n_frames
    a = drain(eng, 12)
    eng.steer(idx=57)
    time.sleep(0.5)
    b = drain(eng, 24)
    check("cache not reset by steer", st.n_frames > before_latent,
          f"{before_latent} -> {st.n_frames}")
    check("frame counter continuous", eng._frames_emitted > before_frames)
    seam = np.abs(b[0].astype(float) - a[-1].astype(float)).mean()
    check("no cut at the steer", seam < 60, f"frame-to-frame MAE across steer = {seam:.1f}")
    eng.stop()

    print("\n[5] generated worlds open on their own stored conditioning")
    # Regression: generated ids start at 1000 and are NOT bank indices.
    wid = GENERATED_BASE_ID
    d = torch.load(A / "out/world_p44.pt", map_location="cpu", weights_only=False)
    torch.save({"latents": d["latents"], "pixels": d["pixels"],
                "prompt_emb": eng.bank.embedding(44), "base_seconds": 1.0},
               tmp / f"world_{wid}.pt")
    (tmp / f"world_{wid}.json").write_text(json.dumps(
        {"id": wid, "prompt": "a generated test world", "seconds": 1.0}))
    eng._index_worlds()
    check("generated world is indexed",
          wid in eng.worlds and eng.worlds[wid]["generated"])

    eng.start(wid)                      # no prompt args -- the case the UI hit
    drain(eng, 81)
    check("opens with no prompt given", eng.status.state == "streaming",
          f"state={eng.status.state} prompt={eng.status.prompt!r}")
    check("used the stored embedding, not umt5", not eng.encoder.loaded)
    eng.stop()

    try:
        eng.start(wid, idx=wid)
        check("world id rejected as a prompt index", False, "no error raised")
    except ValueError as e:
        check("world id rejected as a prompt index", "generated world id" in str(e),
              str(e)[:60])
finally:
    eng.stop()
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL PASSED" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
