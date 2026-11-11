"""`ServerErrorCorsMiddleware`: a 500 only before a response starts; the error always re-raises.

The window-facing half (the 500 names the window's origin) is in `test_api_window_origins.py`.
"""

from __future__ import annotations

import asyncio

import pytest
from apps.api.server_errors import ServerErrorCorsMiddleware
from starlette.types import Message, Receive, Scope, Send


async def _raises_before_sending(scope: Scope, receive: Receive, send: Send) -> None:
    raise RuntimeError("before")


async def _raises_mid_stream(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"data: token\n\n", "more_body": True})
    raise RuntimeError("mid-stream")


async def _receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


async def _run(app: object) -> list[Message]:
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {"type": "http", "method": "GET", "path": "/x", "headers": []}
    with pytest.raises(RuntimeError):
        await ServerErrorCorsMiddleware(app)(scope, _receive, send)  # type: ignore[arg-type]
    return sent


def test_an_error_before_sending_answers_500_then_reraises() -> None:
    sent = asyncio.run(_run(_raises_before_sending))
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert [m["status"] for m in starts] == [500]


def test_an_error_mid_stream_sends_no_second_response() -> None:
    sent = asyncio.run(_run(_raises_mid_stream))
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert [m["status"] for m in starts] == [200]
