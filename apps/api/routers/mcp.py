"""The MCP endpoint, and the window's routes for turning it on (ADR-066 choices 2 and 9, A4).

``POST /mcp`` is what an outside AI program calls: JSON-RPC in, JSON-RPC out, answered by
`doc_assistant.mcp_server`. It has its own access token (``Authorization: Bearer …``), so the
window's launch token does not apply to it (`launch_token._OWN_TOKEN_PATHS`), and it answers
nothing while the person's switch is off. A request carrying an ``Origin`` from a web page is
refused: the host guard (S-5) already stops DNS rebinding, and a browser has no business here.
There is no ``GET /mcp``: the server answers each request in its response and never pushes.

``/api/mcp*`` are the window's: the status (no secret in it), the two switches, the setup (the
token, only by POST, which needs the launch token), and replacing the token.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from apps.api.models.mcp import (
    McpCallPayload,
    McpCommandSetup,
    McpSettingsUpdate,
    McpSetupPayload,
    McpStatusPayload,
)
from doc_assistant import app_settings, credentials, mcp_server
from doc_assistant.read_surface import SURFACE_VERSION

router = APIRouter()

#: The engine's fixed port (the desktop shell's ``API_PORT``), so a client's setup stays valid.
_PORT = int(os.environ.get("DOC_API_PORT", "8001"))
URL = f"http://127.0.0.1:{_PORT}/mcp"
_OFF = "The MCP endpoint is off. Turn it on in Provenote's Settings."


def _log(request: Request) -> mcp_server.CallLog:
    log = getattr(request.app.state, "mcp_calls", None)
    if log is None:
        log = mcp_server.CallLog()
        request.app.state.mcp_calls = log
    return log


def _status(request: Request) -> McpStatusPayload:
    conversations = app_settings.get_mcp_conversations()
    return McpStatusPayload(
        enabled=app_settings.get_mcp_enabled(),
        conversations=conversations,
        token_hint=credentials.mcp_token_hint(),
        url=URL,
        tools=mcp_server.offered_tools(conversations),
        surface_version=SURFACE_VERSION,
        recent_calls=[McpCallPayload(**vars(c)) for c in _log(request).recent()],
    )


def _bridge_command(token: str) -> McpCommandSetup:
    """The command that runs the bridge: the installed engine program, or Python in a checkout."""
    env = {"DOC_MCP_TOKEN": token}
    if getattr(sys, "frozen", False):
        return McpCommandSetup(command=sys.executable, args=["--mcp-bridge"], env=env)
    root = str(Path(__file__).resolve().parents[3])
    return McpCommandSetup(
        command=sys.executable,
        args=["-m", "apps.api", "--mcp-bridge"],
        env={**env, "PYTHONPATH": root},
    )


def _setup(token: str) -> McpSetupPayload:
    command = _bridge_command(token)
    entry = {
        "mcpServers": {
            mcp_server.SERVER_NAME: {
                "command": command.command,
                "args": command.args,
                "env": command.env,
            }
        }
    }
    return McpSetupPayload(
        token=token,
        url=URL,
        header=f"Authorization: Bearer {token}",
        command=command,
        config_json=json.dumps(entry, indent=2),
    )


@router.get("/api/mcp")
def mcp_status(request: Request) -> McpStatusPayload:
    """Whether the endpoint is on, which tools it offers, and the latest calls. No secret."""
    return _status(request)


@router.post("/api/mcp/settings")
def mcp_settings(request: Request, body: McpSettingsUpdate) -> McpStatusPayload:
    """Turn the endpoint or its conversation tools on or off. The first time the endpoint is
    turned on, a token is minted."""
    if body.enabled is not None:
        if body.enabled and credentials.get_mcp_token() is None:
            credentials.replace_mcp_token()
        app_settings.set_mcp_enabled(body.enabled)
    if body.conversations is not None:
        app_settings.set_mcp_conversations(body.conversations)
    return _status(request)


@router.post("/api/mcp/setup")
def mcp_setup() -> McpSetupPayload:
    """The token and what to paste into a client. A POST, so only the window can ask for it."""
    token = credentials.get_mcp_token() or credentials.replace_mcp_token()
    return _setup(token)


@router.post("/api/mcp/token/replace")
def mcp_replace_token() -> McpSetupPayload:
    """A new token: every client holding the old one is cut off until it is given this one."""
    return _setup(credentials.replace_mcp_token())


def _refuse(status: int, message: str, **headers: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=status, headers=headers or None)


@router.post("/mcp")
async def mcp_endpoint(request: Request) -> Response:
    """One JSON-RPC message from an outside AI program."""
    if not app_settings.get_mcp_enabled():
        return _refuse(404, _OFF)
    if request.headers.get("origin"):
        return _refuse(403, "The MCP endpoint does not answer web pages.")
    expected = credentials.get_mcp_token() or ""
    sent = request.headers.get("authorization", "")
    scheme, _, token = sent.partition(" ")
    if (
        not expected
        or scheme.lower() != "bearer"
        or not secrets.compare_digest(token.strip().encode(), expected.encode())
    ):
        return _refuse(
            401, "This needs Provenote's MCP access token.", **{"WWW-Authenticate": "Bearer"}
        )
    try:
        message: Any = json.loads(await request.body())
    except ValueError:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": mcp_server.PARSE_ERROR, "message": "Not JSON"},
            }
        )
    controller = request.app.state.controller
    ctx = mcp_server.McpContext(
        retrieve=controller.rag.retrieve_with_scores,
        store=controller.rag.db,
        conversations=app_settings.get_mcp_conversations(),
        on_call=_log(request).record,
    )
    if isinstance(message, list):  # a batch, which older protocol versions allow
        answers = [a for a in (mcp_server.handle(m, ctx) for m in message) if a is not None]
        return JSONResponse(answers) if answers else Response(status_code=202)
    answer = mcp_server.handle(message, ctx)
    return JSONResponse(answer) if answer is not None else Response(status_code=202)


@router.get("/mcp")
def mcp_no_stream() -> Response:
    """No stream to open: every answer comes back in the response to its request."""
    return Response(status_code=405, headers={"Allow": "POST"})
