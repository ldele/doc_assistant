"""Security S-6 (docs/security.md S1b): a request that changes something carries the launch token.

The desktop shell mints a random token at every launch, hands it to the API sidecar in
``DOC_API_TOKEN`` and to its own window through the ``api_token`` command, and the window sends
it in ``X-Doc-Api-Token``. Any method but GET, HEAD and OPTIONS must carry it, so a web page that
reaches the loopback port cannot add, index, change or delete. Reads stay open: CORS keeps a page
from reading the answers, and the host guard (S-5) keeps it from becoming same-origin.

Unset or blank, nothing is checked — ``just api`` with the Vite dev loop, the test suite and the
headless image run as before. The shell always sets it.
"""

from __future__ import annotations

import os
import secrets

import structlog
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

log = structlog.get_logger(__name__)

# The names of a variable and a header, not secrets; the token itself only ever comes from the env.
TOKEN_ENV = "DOC_API_TOKEN"  # nosec B105
TOKEN_HEADER = "X-Doc-Api-Token"  # nosec B105
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_REFUSAL = "This request needs the token the app was started with."


def configured_token() -> str | None:
    """The token from ``DOC_API_TOKEN``; ``None`` when unset or blank (then nothing is checked)."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    return token or None


class LaunchTokenMiddleware:
    """Answer 401 to an unsafe-method request whose ``X-Doc-Api-Token`` is not the token.

    Pure ASGI rather than ``BaseHTTPMiddleware``, so the chat's event stream passes through
    untouched. Mounted inside CORS, so a refusal still carries the CORS headers and the window
    reads it as a 401, not as a network error."""

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        self.app = app
        self._expected = token.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in _SAFE_METHODS:
            await self.app(scope, receive, send)
            return
        sent = b""
        for name, value in scope["headers"]:
            if name == TOKEN_HEADER.lower().encode("latin-1"):
                sent = value
                break
        if secrets.compare_digest(sent, self._expected):
            await self.app(scope, receive, send)
            return
        log.warning(
            "security_token_refused",
            method=scope["method"],
            path=scope["path"],
            header_present=bool(sent),
        )
        await JSONResponse({"detail": _REFUSAL}, status_code=401)(scope, receive, send)
