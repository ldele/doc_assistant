"""The MCP stdio bridge: for a client that can only start a command (ADR-066 choice 2, A4).

A client starts ``doc-assistant-api --mcp-bridge`` (or ``python -m apps.api --mcp-bridge`` in a
checkout) and speaks MCP over the command's input and output, one JSON message per line. The
bridge forwards each message to the running engine's ``POST /mcp`` with the access token from
``DOC_MCP_TOKEN``, and writes back what the engine answers. It holds no library and loads no
model: Provenote must be running, and a message the engine cannot take gets a JSON-RPC error
saying why, so the client's model can tell the person.

Transport only, stdlib only, and started before the app stack is imported (`__main__`), so it
costs a fraction of a second, not the engine's start.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, TextIO

# The name of a variable, not a secret.
TOKEN_ENV = "DOC_MCP_TOKEN"  # nosec B105
_TIMEOUT_S = 300


def _url() -> str:
    port = int(os.environ.get("DOC_API_PORT", "8001"))
    return f"http://127.0.0.1:{port}/mcp"


def _error(msg_id: Any, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": message}}


def forward(line: str, token: str, url: str) -> str | None:
    """Send one message to the engine; the line to write back, or ``None`` for nothing."""
    try:
        message = json.loads(line)
    except ValueError:
        return json.dumps(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Not JSON"}}
        )
    msg_id = message.get("id") if isinstance(message, dict) else None
    request = urllib.request.Request(  # a fixed loopback URL, never one a user typed
        url,
        data=line.encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # nosec B310
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
        except ValueError:
            detail = str(e)
        return json.dumps(_error(msg_id, detail)) if msg_id is not None else None
    except (urllib.error.URLError, OSError):
        reason = "Provenote is not running, so its library cannot be reached. Start the app."
        return json.dumps(_error(msg_id, reason)) if msg_id is not None else None
    return body.strip() or None


def run_bridge(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Forward lines until the client closes the input. Returns the process exit code."""
    source = stdin or sys.stdin
    sink = stdout or sys.stdout
    if hasattr(sink, "reconfigure"):
        sink.reconfigure(encoding="utf-8")
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        sys.stderr.write(f"{TOKEN_ENV} is not set: copy the setup from Provenote's Settings.\n")
        return 2
    url = _url()
    for line in source:
        if not line.strip():
            continue
        out = forward(line, token, url)
        if out is not None:
            sink.write(out + "\n")
            sink.flush()
    return 0
