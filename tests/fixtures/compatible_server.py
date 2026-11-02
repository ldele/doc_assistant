"""A stdlib OpenAI-compatible server on 127.0.0.1, for proving ADR-060's adapter at $0.

Two families of routes on one port:

* ``/open/v1/...`` takes no key, like a server on this computer (LM Studio, a lab server).
* ``/cloud/v1/...`` refuses any request without ``Authorization: Bearer <key>``, standing in for a
  cloud preset. ``/cloud/v1/key`` answers like OpenRouter's key check.

Both serve ``GET /models`` and ``POST /chat/completions``. A chat answer streams the way real
servers do, with the parts the reader must ignore: a role-only chunk, an SSE comment, a reasoning
field, a usage chunk with empty ``choices``, then ``[DONE]``. Every request is recorded (method,
path, headers, JSON body) so a test can pin exactly what was sent.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_ANSWER = (
    "Retrieval gives the model passages to ground its answer in [1], ",
    "which lowers unsupported claims [2].",
)


class FakeCompatibleServer:
    """Start with ``with FakeCompatibleServer() as server:``; ``server.url("open")`` is a base."""

    def __init__(
        self,
        *,
        key: str = "test-key",
        models: tuple[str, ...] = ("fake-model", "fake-model-large"),
        answer: tuple[str, ...] = DEFAULT_ANSWER,
        usage: tuple[int, int] | None = (1800, 40),
    ) -> None:
        self.key = key
        self.models = models
        self.answer = answer
        self.usage = usage
        self.requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def url(self, family: str = "open") -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/{family}/v1"

    def __enter__(self) -> FakeCompatibleServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def chat_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"].endswith("/chat/completions")]

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # keep the test output quiet
                return

            def _record(self, body: Any) -> None:
                server.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )

            def _authorised(self) -> bool:
                if not self.path.startswith("/cloud/"):
                    return True
                return self.headers.get("Authorization") == f"Bearer {server.key}"

            def _json(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._record(None)
                if not self._authorised():
                    self._json(401, {"error": {"message": "Invalid API key"}})
                elif self.path.endswith("/models"):
                    self._json(200, {"object": "list", "data": [{"id": m} for m in server.models]})
                elif self.path.endswith("/key"):
                    self._json(200, {"data": {"label": "test"}})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"null")
                self._record(body)
                if not self._authorised():
                    self._json(401, {"error": {"message": "Invalid API key"}})
                    return
                if not self.path.endswith("/chat/completions"):
                    self._json(404, {"error": {"message": "not found"}})
                    return
                if not body.get("stream"):
                    text = "".join(server.answer)
                    self._json(200, {"choices": [{"message": {"content": text}}]})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events: list[str] = [
                    ": keep-alive",
                    _data({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
                    _data({"choices": [{"index": 0, "delta": {"reasoning": "Let me think."}}]}),
                ]
                events += [
                    _data({"choices": [{"index": 0, "delta": {"content": t}}]})
                    for t in server.answer
                ]
                if server.usage is not None:
                    prompt, completion = server.usage
                    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
                    events.append(_data({"choices": [], "usage": usage}))
                events.append("data: [DONE]")
                for event in events:
                    self.wfile.write((event + "\n\n").encode("utf-8"))
                    self.wfile.flush()

        return Handler


def _data(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload)
