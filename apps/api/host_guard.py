"""The host guard (S-5), and the line it writes when it refuses a request (S-11).

Starlette's ``TrustedHostMiddleware`` answers a foreign ``Host`` with 400 and says nothing, so
the periodic check (`docs/security.md` §6) had no record of a refused DNS-rebinding attempt. This
subclass writes ``security_host_refused`` before the refusal: the control's name and the host
asked for, never the request's content.
"""

from __future__ import annotations

import structlog
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import Receive, Scope, Send

log = structlog.get_logger(__name__)

#: A host header is a name, but a hostile one can be long; the log keeps its head.
_HOST_CHARS = 100


class LoggedTrustedHostMiddleware(TrustedHostMiddleware):
    """`TrustedHostMiddleware` that logs each host it refuses."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and not self.allow_any:
            host = Headers(scope=scope).get("host", "").split(":")[0]
            allowed = any(
                host == pattern or (pattern.startswith("*") and host.endswith(pattern[1:]))
                for pattern in self.allowed_hosts
            )
            if not allowed:
                log.warning("security_host_refused", control="host_guard", host=host[:_HOST_CHARS])
        await super().__call__(scope, receive, send)
