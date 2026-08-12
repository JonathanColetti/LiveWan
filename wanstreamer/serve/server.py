"""LiveWan browser demo server.

    livewan-serve --port 17070   # or: python -m wanstreamer.serve.server

Serves a single steerable stream: pick a world (or generate a new one from text),
then steer it with the 96-prompt bank or free text while it runs.
"""

import argparse
import asyncio
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from . import paths
from .auth import TokenAuth
from .conditioning import theme_of
from .engine import WORLDS, Engine
import cv2

INDEX = Path(__file__).resolve().parent / "web" / "index.html"

# Populated by main(); the endpoints read it rather than module-level globals built at
# import time, so importing this module has no side effects.
_S = {"engine": None, "boot": {"messages": [], "done": False, "error": None},
      "thumbs": {}, "clients": set()}


def E() -> Engine:
    return _S["engine"]


def build_parser():
    p = argparse.ArgumentParser(prog="livewan-serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", default=paths.ASSETS, help="the LiveWan weights/data dir")
    p.add_argument("--wan-repo", default=paths.WAN_REPO, help="Wan2.1 reference checkout")
    p.add_argument("--base-dir", default=paths.BASE_DIR, help="VAE / umt5 / base model")
    p.add_argument("--weights", default=None, help="defaults to checkpoints/t14b_b64/latest.pt")
    p.add_argument("--worlds-dir", default=paths.WORLDS_DIR, help="where generated worlds go")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=17070)
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--buffer-seconds", type=float, default=1.5)
    p.add_argument("--no-compile", action="store_true",
                   help="skip the one-off VAE decoder compile (~50 ms/block slower)")
    p.add_argument("--no-worldgen", action="store_true",
                   help="disable text->world generation (saves 2.6 GB VRAM)")
    p.add_argument("--token", default="1234",
                   help="shared token for ?token=/Bearer/cookie auth; empty string disables")
    return p



class StartReq(BaseModel):
    world: int
    prompt_idx: int | None = None
    prompt_text: str | None = None
    seed: int = 0
    steps: int = 2
    window: int = 6
    latent_norm: float = 1.0


class SteerReq(BaseModel):
    prompt_idx: int | None = None
    prompt_text: str | None = None
    crossfade: int = 0


class WorldReq(BaseModel):
    prompt_text: str | None = None
    prompt_idx: int | None = None
    steps: int = 30
    seed: int = 0
    guide: float = 5.0



def _make_thumb(wid):
    """One representative frame per world, cached in memory for the picker."""
    

    entry = E().worlds.get(int(wid))
    if not entry:
        return
    d = torch.load(entry["path"], map_location="cpu", weights_only=False)
    frame = d["pixels"][40].numpy()[:, :, ::-1]
    _S["thumbs"][int(wid)] = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()


def _boot_load():
    try:
        E().load(progress=lambda m: _S["boot"]["messages"].append(m))
        for w in list(E().worlds):
            _make_thumb(w)
        _S["boot"]["done"] = True
    except Exception as e:
        _S["boot"]["error"] = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def _lifespan(_app):
    task = asyncio.create_task(_pump())
    yield
    task.cancel()


app = FastAPI(title="LiveWan", lifespan=_lifespan)



@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX.read_text()


@app.get("/api/boot")
def boot():
    b = _S["boot"]
    return {"done": b["done"], "error": b["error"], "messages": b["messages"][-6:],
            "detail": E().status.detail if E() else ""}


@app.get("/api/info")
def info():
    if not _S["boot"]["done"]:
        raise HTTPException(503, "still loading")
    d = E().info()
    d["world_themes"] = {w: theme_of(w) for w in WORLDS}
    return d


@app.get("/api/status")
def status():
    e = E()
    s = e.status
    return {
        "state": s.state, "detail": s.detail, "world": s.world,
        "prompt": s.prompt, "prompt_source": s.prompt_source,
        "stats": s.stats, "error": s.error, "progress": s.progress,
        "buffered": e.frames.qsize(),
        "encoder_loaded": e.encoder.loaded if e.encoder else False,
    }


@app.get("/api/thumb/{world}")
def thumb(world: int):
    if world not in _S["thumbs"]:
        raise HTTPException(404, "no thumbnail")
    return Response(_S["thumbs"][world], media_type="image/jpeg")


@app.post("/api/start")
def start(r: StartReq):
    if not _S["boot"]["done"]:
        raise HTTPException(503, "still loading")
    try:
        E().start(r.world, r.prompt_idx, r.prompt_text, seed=r.seed, steps=r.steps,
                  window=r.window, latent_norm=r.latent_norm)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return status()


@app.post("/api/steer")
def steer(r: SteerReq):
    try:
        E().steer(r.prompt_idx, r.prompt_text, crossfade=r.crossfade)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return status()


@app.post("/api/world/new")
def world_new(r: WorldReq):
    """Generate a brand-new opening world from text. Slow: ~52 s at 30 steps."""
    if not _S["boot"]["done"]:
        raise HTTPException(503, "still loading")
    try:
        wid = E().generate_world(r.prompt_text, r.prompt_idx, steps=r.steps,
                                 seed=r.seed, guide=r.guide)
        _make_thumb(wid)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return {"world": wid, **E().info()}


@app.delete("/api/world/{wid}")
def world_delete(wid: int):
    try:
        E().delete_world(wid)
        _S["thumbs"].pop(wid, None)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return E().info()


@app.post("/api/stop")
def stop():
    E().stop()
    return status()



def _next_frame(timeout=0.1):
    try:
        return E().frames.get(True, timeout)
    except queue.Empty:
        return None


async def _pump():
    """Single reader of the frame queue, broadcasting to every viewer.

    One consumer only. If each socket drained the queue itself, two viewers would
    each get half the frames.
    """
    loop = asyncio.get_running_loop()
    clients = _S["clients"]
    last = 0.0
    while True:
        if not clients or E() is None:
            await asyncio.sleep(0.1)
            continue
        frame = await loop.run_in_executor(None, _next_frame, 0.1)
        dead = []
        if frame is not None:
            for c in list(clients):
                try:
                    await c.send_bytes(frame)
                except Exception:
                    dead.append(c)
        if time.time() - last > 0.5:
            last = time.time()
            payload = {"t": "status", **status()}
            for c in list(clients):
                try:
                    await c.send_json(payload)
                except Exception:
                    dead.append(c)
        for c in dead:
            clients.discard(c)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    _S["clients"].add(sock)
    try:
        await sock.send_json({"t": "status", **status()})
        while True:
            await sock.receive_text()  # client keepalive; ignored
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _S["clients"].discard(sock)



def main(argv=None):
    args = build_parser().parse_args(argv)
    _S["engine"] = Engine(
        args.assets, args.wan_repo, args.base_dir, args.weights,
        jpeg_quality=args.jpeg_quality, buffer_seconds=args.buffer_seconds,
        worlds_dir=args.worlds_dir, allow_worldgen=not args.no_worldgen,
        compile_vae=not args.no_compile,
    )
    threading.Thread(target=_boot_load, daemon=True, name="livewan-boot").start()

    served = TokenAuth(app, args.token) if args.token else app
    if args.token:
        print(f"token auth on; open  http://{args.host}:{args.port}/?token={args.token}")
    else:
        print(f"no auth; open  http://{args.host}:{args.port}/")
    uvicorn.run(served, host=args.host, port=args.port, log_level="warning",
                ws_max_size=16 * 2**20)


if __name__ == "__main__":
    main()
