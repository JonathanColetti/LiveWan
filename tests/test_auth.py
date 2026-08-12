"""Token gate checks. No GPU needed.

    python tests/test_auth.py
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

import requests
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket

from wanstreamer.serve.auth import COOKIE, TokenAuth

PORT = 17099
BASE = f"http://127.0.0.1:{PORT}"
fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
    if not ok:
        fails.append(name)


app = FastAPI()


@app.get("/")
def root():
    return {"ok": True}


@app.websocket("/ws")
async def ws(s: WebSocket):
    await s.accept()
    await s.send_text("hello")
    await s.close()


server = uvicorn.Server(uvicorn.Config(TokenAuth(app, "1234"), host="127.0.0.1",
                                       port=PORT, log_level="error"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(50):
    try:
        requests.get(BASE + "/", timeout=0.5)
        break
    except Exception:
        time.sleep(0.2)

print("HTTP")
check("no token rejected", requests.get(BASE + "/").status_code, 401)
check("wrong token rejected", requests.get(BASE + "/?token=nope").status_code, 401)
r = requests.get(BASE + "/?token=1234")
check("query token accepted", r.status_code, 200)
check("cookie issued on query auth", COOKIE in r.cookies, True)
check("bearer accepted",
      requests.get(BASE + "/", headers={"Authorization": "Bearer 1234"}).status_code, 200)
check("cookie accepted", requests.get(BASE + "/", cookies={COOKIE: "1234"}).status_code, 200)


async def ws_open(suffix="", headers=None):
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws{suffix}",
                                  additional_headers=headers or {}) as w:
        return await w.recv()


async def ws_denied():
    try:
        await ws_open()
        return "accepted"
    except Exception:
        return "rejected"


print("\nWebSocket (the frame stream -- a plain HTTP middleware would miss this)")
check("no token rejected", asyncio.run(ws_denied()), "rejected")
check("query token accepted", asyncio.run(ws_open("?token=1234")), "hello")
check("cookie accepted", asyncio.run(ws_open("", {"Cookie": f"{COOKIE}=1234"})), "hello")

print("\nDisabled gate")
open_app = uvicorn.Server(uvicorn.Config(TokenAuth(app, ""), host="127.0.0.1",
                                         port=PORT + 1, log_level="error"))
threading.Thread(target=open_app.run, daemon=True).start()
time.sleep(1.5)
check("empty token disables auth",
      requests.get(f"http://127.0.0.1:{PORT+1}/").status_code, 200)

print("\n" + ("ALL PASSED" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
