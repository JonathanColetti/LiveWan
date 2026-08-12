"""Shared token gate for the demo server.

Starlette's `BaseHTTPMiddleware` never sees websocket scopes, and the frame stream is
a websocket, so this is a plain ASGI wrapper rather than a FastAPI middleware.

A token supplied in the query string is echoed back as a cookie, so visiting
`/?token=...` once is enough: the websocket handshake then carries it automatically
and the URL can be shared without the query string surviving in every request.
"""

from urllib.parse import parse_qs

COOKIE = "livewan_token"


class TokenAuth:
    def __init__(self, app, token):
        self.app, self.token = app, token

    def _supplied(self, scope):
        headers = {k.lower(): v for k, v in (scope.get("headers") or [])}
        token = parse_qs(scope.get("query_string", b"").decode()).get("token", [None])[0]
        if token:
            return token
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer "):
            return auth[7:]
        for part in headers.get(b"cookie", b"").decode().split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE:
                return v
        return None

    async def __call__(self, scope, receive, send):
        if not self.token or scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        if self._supplied(scope) != self.token:
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
            return await send({"type": "http.response.body",
                               "body": b"unauthorized -- append ?token=<token> to the URL"})

        if scope["type"] == "http":
            async def _send(msg):
                if msg["type"] == "http.response.start":
                    msg["headers"] = list(msg.get("headers") or []) + [
                        (b"set-cookie",
                         f"{COOKIE}={self.token}; Path=/; SameSite=Lax; Max-Age=604800".encode())
                    ]
                await send(msg)

            return await self.app(scope, receive, _send)
        await self.app(scope, receive, send)
