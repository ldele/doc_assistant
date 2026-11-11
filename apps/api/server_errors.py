"""An engine error reaches the installed window as a 500, not as "Failed to fetch".

The installed window calls ``http://127.0.0.1:8001`` from its own origin, so it reads a response
only when that response names the origin in ``Access-Control-Allow-Origin``. Starlette answers an
unhandled exception from its outermost layer, outside CORS, so that 500 carried no CORS headers:
the browser discarded it and the page saw ``TypeError: Failed to fetch`` — the same text a dead
engine gives (0.7.3, *"Couldn't load connections"* on a document page). In dev, Vite proxies
``/api`` on the same origin, so the same failure read as a plain 500 there.

This layer is mounted inside CORS. When a route raises before anything was sent, it answers the
500 itself (Starlette's own body, so the window's error text is unchanged), then re-raises: the
server still logs the traceback and a test client still sees the exception. Once a response has
started — a chat's event stream — it only re-raises, since a second response cannot be sent.
"""

from __future__ import annotations

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ServerErrorCorsMiddleware:
    """Answer an unhandled exception with a 500 from inside CORS, then re-raise it.

    Pure ASGI rather than ``BaseHTTPMiddleware``, so the chat's event stream passes through
    untouched (same reason as ``LaunchTokenMiddleware``)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            if not started:
                await PlainTextResponse("Internal Server Error", status_code=500)(
                    scope, receive, send
                )
            raise
