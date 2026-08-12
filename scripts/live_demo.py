"""Live streaming demo: the student generates video continuously, in real time,
and you can change the event text mid-stream without ever restarting it.

    python scripts/live_demo.py --weights deliver/checkpoints/student_p4_t14b.pt \
        --world-cache out/world_p0.pt --port 17070

WHAT THIS DEMONSTRATES, precisely
  The model holds a persistent world W and extends it block by block, forever,
  at about real time on one A100. `POST /say` swaps the text conditioning on the
  next block boundary (480 ms at block 3) WITHOUT clearing the K/V cache, so the
  scene continues rather than cutting -- that is the whole point of the
  block-causal design and it is what a canned video cannot show.

WHAT IT IS NOT
  Not speech, not lip-sync, not a talking assistant. The model is text-to-video;
  it has no audio and was never trained to articulate words. Typing at it steers
  the SCENE. A face that appears to talk is generating plausible mouth motion,
  not saying your sentence. The UI says this too -- do not let a viewer infer
  otherwise.

HONEST LIMITS, measured (FINDINGS.md 5c and the long-stream runs)
  - Quality is good for roughly the first minute. By 160 s sharpness has fallen
    to ~73% of the world's and the shipped base additionally develops saturated
    lips and pink cheek blotches. The UI shows a drift meter and elapsed time so
    a viewer can watch this happen instead of being sold past it.
  - `--reseed-units` re-primes from the cached world when the stream gets old.
    That is a cut, not a fix, and it is off unless you ask for it.
  - Free-text needs the 11 GB umt5-xxl encoder (`--t5`). Without it the demo
    falls back to the pre-encoded 96-prompt bank, which still shows mid-stream
    switching, just not arbitrary text.
"""
import argparse
import gc
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'wan21_repo'))
sys.path.insert(0, HERE)

from safetensors.torch import load_file  # noqa: E402
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.modules.model import WanModel  # noqa: E402
from wan.modules.vae import WanVAE  # noqa: E402

from wanstreamer.core import latent_geometry  # noqa: E402
from wanstreamer.stream import FewStepStreamer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--size', default='640*368')
ap.add_argument('--weights', default=os.path.join(HERE, 'deliver/checkpoints/student_p4_t14b.pt'))
ap.add_argument('--ckpt', default=os.path.join(HERE, 'checkpoints/wan21_13b'))
ap.add_argument('--prompts', default=os.path.join(HERE, 'data/prompts.pt'))
ap.add_argument('--world-cache', default=os.path.join(HERE, 'out/world_p0.pt'))
ap.add_argument('--prompt-idx', type=int, default=0)
ap.add_argument('--block', type=int, default=3)
ap.add_argument('--steps', type=int, default=2)
ap.add_argument('--window', type=int, default=6)
ap.add_argument('--latent-norm', type=float, default=1.0)
ap.add_argument('--fps', type=int, default=25)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--port', type=int, default=17070)
ap.add_argument('--host', default='127.0.0.1')
ap.add_argument('--max-frames', type=int, default=4096,
                help='RoPE table size; caps stream length. 1024 is upstream, '
                     'which is only ~2.7 min at block 3.')
ap.add_argument('--reseed-units', type=int, default=0,
                help='re-prime from the world every N units (0 = never). A cut, '
                     'not a fix for drift -- see module docstring.')
ap.add_argument('--t5', action='store_true',
                help='load umt5-xxl (11 GB) so the chat box accepts free text')
ap.add_argument('--jpeg-quality', type=int, default=82)
ap.add_argument('--decode-device', default='',
                help="put the VAE on a second GPU (e.g. 'cuda:1') so decode "
                     'overlaps generation. They cost about the same per unit '
                     '(~438 ms generate, ~490 ms decode), so running them '
                     'serially caps the demo at ~0.5x real time; pipelined, '
                     'throughput is the slower of the two and reaches ~1x.')
args = ap.parse_args()

dev = torch.device('cuda')
dev_dec = torch.device(args.decode_device) if args.decode_device else dev
cfg = WAN_CONFIGS['t2v-1.3B']
DTYPE = cfg.param_dtype
size = SIZE_CONFIGS[args.size]
h_lat, w_lat, hp, wp = latent_geometry(size[0], size[1])
S = hp * wp
torch.manual_seed(args.seed)

print(f'=== live demo | {args.size} | unit = {args.block} latent frames = '
      f'{args.block * 4 / args.fps * 1000:.0f} ms of video ===')

blob = torch.load(args.prompts, map_location='cpu')
PROMPTS = blob['prompts']
ctx_neg = blob['neg'][0].float().to(dev).unsqueeze(0)


def build_student():
    m = WanModel(dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
                 num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                 window_size=cfg.window_size, qk_norm=True,
                 cross_attn_norm=True, eps=1e-6)
    m.load_state_dict(load_file(f'{args.ckpt}/diffusion_pytorch_model.safetensors'),
                      strict=True)
    sd = torch.load(args.weights, map_location='cpu')
    src = sd.get('model', sd)
    res = m.load_state_dict(src, strict=False)
    got = {n for n, _ in m.named_parameters()} - set(res.missing_keys)
    # A checkpoint that silently failed to load would demo the BASE model while
    # claiming to be the student, which is the most embarrassing possible bug.
    if len(got) != len(list(m.named_parameters())) or res.unexpected_keys:
        raise SystemExit(f'bad checkpoint load from {args.weights}')
    print(f'    student: {os.path.basename(args.weights)} '
          f'(step {sd.get("step", "?")}, all {len(got)} tensors matched)')
    return m.to(dev).eval().requires_grad_(False)


class StreamDecoder:
    """VAE decode that carries `feat_cache` across calls, so consecutive blocks
    join seamlessly. Resetting the cache per block would put a visible seam at
    every unit boundary."""

    def __init__(self, vae):
        self.m, self.scale, self.dtype = vae.model, vae.scale, vae.dtype
        self.m.clear_cache()

    def reset(self):
        self.m.clear_cache()

    @torch.no_grad()
    def decode(self, z):
        """z: [C, F, H, W] latent -> uint8 [F', H, W, 3]. First call after reset
        emits 1 pixel frame per latent frame boundary; later calls emit 4."""
        z = z.to(next(self.m.parameters()).device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=self.dtype):
            # generate_block returns [1, C, F, H, W]; set_world gives [C, F, H, W].
            zz = z if z.dim() == 5 else z.unsqueeze(0)
            zz = zz / self.scale[1].view(1, -1, 1, 1, 1) + self.scale[0].view(1, -1, 1, 1, 1)
            x = self.m.conv2(zz)
            outs = []
            for i in range(x.shape[2]):
                self.m._conv_idx = [0]
                outs.append(self.m.decoder(x[:, :, i:i + 1], feat_cache=self.m._feat_map,
                                           feat_idx=self.m._conv_idx))
            out = torch.cat(outs, 2).float().clamp_(-1, 1).squeeze(0)
        return ((out.permute(1, 2, 3, 0) + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


class Live:
    """Owns the model and the generation thread. All mutation of the streamer
    happens on that thread; the HTTP handlers only post intents into `pending`."""

    def __init__(self):
        self.vae = WanVAE(vae_pth=f'{args.ckpt}/{cfg.vae_checkpoint}', device=dev_dec)
        wb = torch.load(args.world_cache, map_location='cpu')
        self.world_lat = wb['latents'].to(dev)[:, :9]
        self.student = build_student()
        self.t5 = None
        if args.t5:
            from wan.modules.t5 import T5EncoderModel
            print('    loading umt5-xxl (11 GB) for free-text chat ...')
            self.t5 = T5EncoderModel(
                text_len=512, dtype=cfg.t5_dtype, device=dev,
                checkpoint_path=f'{args.ckpt}/{cfg.t5_checkpoint}',
                tokenizer_path=f'{args.ckpt}/google/umt5-xxl')
            print('    t5 ready -- chat box accepts free text')

        self.frames = deque(maxlen=int(args.fps * 8))   # ~8 s of slack
        self.latq = queue.Queue(maxsize=3)
        self.lock = threading.Lock()
        self.dec_ms = 0.0
        self.pending = None          # (kind, payload) from HTTP
        self.stop = False
        self.units = 0
        self.gen_seconds = 0.0
        self.video_seconds = 0.0
        # Wall clock, not summed generate time. Once decode is pipelined onto a
        # second GPU, video/generate_time would report ~1.1x while the viewer
        # actually receives frames at the slower of the two stages -- so the
        # headline number has to be measured against the clock.
        self.t_start = time.time()
        self.last_ms = 0.0
        self.caption = PROMPTS[args.prompt_idx]
        self.status_note = ''
        self.dec = StreamDecoder(self.vae)
        self._build_streamer()

    def _ctx_from_text(self, text):
        if self.t5 is None:
            return None
        with torch.no_grad():
            emb = self.t5([text], dev)[0]
        return emb.float().unsqueeze(0).to(dev)

    def _build_streamer(self):
        nw = self.world_lat.shape[1]
        self.st = FewStepStreamer(
            self.student, width=size[0], height=size[1],
            max_frames=args.max_frames, device=dev, dtype=DTYPE,
            window_frames=args.window, time_scale=1000.0,
            cache_frames=nw + args.window + args.block + 2,
            block_frames=args.block, num_steps=args.steps,
            shift=1.0, sampler='renoise')
        ctx = blob['pos'][args.prompt_idx].float().to(dev).unsqueeze(0)
        with torch.amp.autocast('cuda', enabled=False):
            self.st.set_text(self.student.text_embedding(ctx))
        # re-priming restores the ORIGINAL conditioning, so the caption must go
        # back too -- otherwise the UI claims a prompt the model is not using
        self.caption = PROMPTS[args.prompt_idx]
        self.st.latent_norm = args.latent_norm
        self.st.set_world(self.world_lat)
        # Hand the reset to the decode thread THROUGH the queue rather than
        # touching the decoder here: it keeps the two threads ordered without a
        # lock, so a reset can never land between a block and its own decode.
        self.latq.put(('reset', self.world_lat))
        self.units = 0
        self.video_seconds = 0.0
        self.t_start = time.time()

    def _push(self, arr):
        with self.lock:
            for f in arr:
                self.frames.append(f)

    def say(self, text=None, idx=None):
        self.pending = ('text', (text, idx))

    def reset(self):
        self.pending = ('reset', None)

    def _apply_pending(self):
        p, self.pending = self.pending, None
        if p is None:
            return
        kind, payload = p
        if kind == 'reset':
            self._build_streamer()
            self.status_note = 'stream re-primed from the world'
            return
        text, idx = payload
        if idx is not None and 0 <= idx < len(PROMPTS):
            ctx = blob['pos'][idx].float().to(dev).unsqueeze(0)
            self.caption = PROMPTS[idx]
        elif text:
            emb = self._ctx_from_text(text)
            if emb is None:
                self.status_note = 'free text needs --t5; pick a preset instead'
                return
            ctx = emb
            self.caption = text
        else:
            return
        with torch.amp.autocast('cuda', enabled=False):
            self.st.set_text(self.student.text_embedding(ctx))
        self.status_note = 'event text switched without clearing K/V'

    def run(self):
        """Generation thread. Produces latents only; decoding happens on the
        decode thread so the two overlap."""
        gen = torch.Generator(device='cuda').manual_seed(args.seed)
        while not self.stop:
            self._apply_pending()
            if args.reseed_units and self.units and self.units % args.reseed_units == 0:
                self._build_streamer()
                self.status_note = f'auto re-primed after {args.reseed_units} units'
            t0 = time.time()
            try:
                z = self.st.generate_block(generator=gen)
            except (IndexError, AssertionError) as e:
                self.status_note = f'stream ended at the RoPE limit: {e}'
                self._build_streamer()
                continue
            self.last_ms = (time.time() - t0) * 1000
            self.gen_seconds += self.last_ms / 1000
            self.units += 1
            # blocks when the decoder falls behind, which is the back-pressure
            self.latq.put(('block', z))

    def run_decode(self):
        """Decode thread. Also owns frame-buffer resets, so they stay ordered
        with respect to the blocks they follow."""
        while not self.stop:
            try:
                kind, z = self.latq.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == 'reset':
                self.dec.reset()
                with self.lock:
                    self.frames.clear()
            t0 = time.time()
            arr = self.dec.decode(z)
            self.dec_ms = (time.time() - t0) * 1000
            self.video_seconds += arr.shape[0] / args.fps
            self._push(arr)

    def pop(self):
        with self.lock:
            return self.frames.popleft() if self.frames else None


PAGE = """<!doctype html><meta charset=utf-8><title>Wan-Streamer live</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0;padding:24px}
 .wrap{max-width:900px;margin:0 auto}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#888;margin:0 0 16px}
 img{width:100%;max-width:840px;background:#000;border-radius:6px;display:block}
 .row{display:flex;gap:8px;margin:12px 0}
 input[type=text]{flex:1;padding:10px;border-radius:6px;border:1px solid #333;background:#1a1a1a;color:#eee}
 button{padding:10px 14px;border-radius:6px;border:1px solid #333;background:#222;color:#eee;cursor:pointer}
 button:hover{background:#2c2c2c}
 .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}
 .stat{background:#1a1a1a;border-radius:6px;padding:8px 10px}
 .stat b{display:block;font-size:17px;color:#fff}
 .warn{background:#2a2118;border:1px solid #5a4632;border-radius:6px;padding:10px 12px;margin:12px 0;color:#e3c9a3}
 .cap{color:#9ad;margin:8px 0;min-height:20px}
 .note{color:#7a7;min-height:18px}
 .presets button{font-size:12px;padding:6px 9px}
</style>
<div class=wrap>
<h1>Wan-Streamer &mdash; live block-causal stream</h1>
<p class=sub>One A100. The model is extending a persistent world, block by block,
right now. Nothing here is pre-rendered.</p>
<img id=v src="/video.mjpg">
<p class=cap id=cap></p>
<div class=row>
  <input type=text id=t placeholder="Describe what should happen next...">
  <button onclick=say()>Send</button>
  <button onclick=reset()>Re-prime</button>
</div>
<div class=presets id=presets></div>
<div class=stats>
  <div class=stat><b id=rt>-</b>speed vs real time</div>
  <div class=stat><b id=ms>-</b>ms per 480 ms unit</div>
  <div class=stat><b id=vs>-</b>stream length</div>
  <div class=stat><b id=un>-</b>units generated</div>
</div>
<p class=note id=note></p>
<div class=warn>
<b>What this is not.</b> This model is text-to-video with no audio and no
lip-sync. It was never trained to articulate speech, so the face is producing
plausible mouth motion, not saying your words &mdash; your text steers the
<i>scene</i>, not a voice.<br><br>
<b>Watch it age.</b> Quality is good for roughly the first minute. By ~160 s
sharpness falls to about 73% of the source world's and colour artifacts creep in.
That is a real, unfixed limitation, not a rendering glitch &mdash; hit
<i>Re-prime</i> to restart from the world.
</div>
</div>
<script>
async function say(){const t=document.getElementById('t');if(!t.value)return;
 await fetch('/say',{method:'POST',body:JSON.stringify({text:t.value})});t.value='';}
async function pick(i){await fetch('/say',{method:'POST',body:JSON.stringify({idx:i})});}
async function reset(){await fetch('/reset',{method:'POST'});}
document.getElementById('t').addEventListener('keydown',e=>{if(e.key==='Enter')say()});
async function tick(){try{const r=await(await fetch('/status')).json();
 document.getElementById('rt').textContent=r.realtime.toFixed(2)+'x';
 document.getElementById('ms').textContent=r.last_ms.toFixed(0);
 document.getElementById('vs').textContent=r.video_seconds.toFixed(1)+'s';
 document.getElementById('un').textContent=r.units;
 document.getElementById('cap').textContent='> '+r.caption;
 document.getElementById('note').textContent=r.note;
 if(!window._p){window._p=1;document.getElementById('presets').innerHTML=
   r.presets.map((p,i)=>`<button onclick="pick(${p.i})">${p.t}</button>`).join(' ');}
 }catch(e){}}
setInterval(tick,700);tick();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith('/status'):
            L = self.server.live
            wall = time.time() - L.t_start
            rt = (L.video_seconds / wall) if wall > 0 else 0.0
            presets = [{'i': i, 't': PROMPTS[i][:40]} for i in (0, 44, 60, 82)]
            return self._json({'realtime': rt, 'last_ms': L.last_ms,
                               'dec_ms': L.dec_ms,
                               'video_seconds': L.video_seconds, 'units': L.units,
                               'caption': L.caption, 'note': L.status_note,
                               'presets': presets})
        if self.path.startswith('/video.mjpg'):
            return self._mjpeg()
        body = PAGE.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        L, period = self.server.live, 1.0 / args.fps
        nxt = time.time()
        try:
            while not L.stop:
                f = L.pop()
                if f is None:
                    time.sleep(0.01)
                    nxt = time.time()
                    continue
                buf = io.BytesIO()
                Image.fromarray(f).save(buf, 'JPEG', quality=args.jpeg_quality)
                d = buf.getvalue()
                self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n'
                                 b'Content-Length: ' + str(len(d)).encode() + b'\r\n\r\n')
                self.wfile.write(d)
                self.wfile.write(b'\r\n')
                nxt += period
                time.sleep(max(0, nxt - time.time()))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            data = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            data = {}
        if self.path.startswith('/reset'):
            self.server.live.reset()
        else:
            self.server.live.say(text=data.get('text'), idx=data.get('idx'))
        self._json({'ok': True})


def main():
    live = Live()
    threading.Thread(target=live.run, daemon=True).start()
    threading.Thread(target=live.run_decode, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.live = live
    srv.daemon_threads = True
    print(f'\n    serving on http://{args.host}:{args.port}  (Ctrl-C to stop)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        live.stop = True


if __name__ == '__main__':
    main()
