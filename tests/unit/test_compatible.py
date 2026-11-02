"""Tests for the OpenAI-compatible client (ADR-060, choices 1, 6 and 8; the ADR's tests 1, 2, 6).

No network beyond 127.0.0.1: requests go through ``httpx.MockTransport`` or the stdlib fake server
in ``tests/fixtures/compatible_server.py``. No paid call is possible from here.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator

import httpx
import pytest
import structlog

from doc_assistant import compatible
from doc_assistant.compatible import (
    CompatibleChatModel,
    CompatibleClient,
    CompatibleError,
    Usage,
    list_models,
    read_stream,
    request_body,
    request_headers,
    stream_chat,
)
from doc_assistant.providers import Endpoint
from doc_assistant.tracking import TokenCounter
from tests.fixtures.compatible_server import FakeCompatibleServer


def _sse(*payloads: object) -> list[str]:
    return [p if isinstance(p, str) else "data: " + json.dumps(p) for p in payloads]


# ============================================================
# Test 1: what is sent
# ============================================================


def test_the_body_is_model_messages_and_stream_only() -> None:
    body = request_body("m", [{"role": "user", "content": "q"}], stream=True)
    assert set(body) == {"model", "messages", "stream"}


def test_the_headers_carry_a_key_only_when_there_is_one() -> None:
    assert request_headers(None) == {"Content-Type": "application/json"}
    assert request_headers("k")["Authorization"] == "Bearer k"
    assert set(request_headers("k")) == {"Content-Type", "Authorization"}


def test_a_request_sends_nothing_else_and_only_to_its_address() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="data: [DONE]\n\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    point = Endpoint("groq", "Groq", "https://api.groq.com/openai/v1", "gsk-secret")
    list(stream_chat(point, "llama", [{"role": "user", "content": "q"}], client=client))
    (request,) = seen
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": "llama",
        "messages": [{"role": "user", "content": "q"}],
        "stream": True,
    }
    assert request.headers["authorization"] == "Bearer gsk-secret"
    # Nothing that names this app, and none of the parameters the norm lets vendors refuse.
    sent = {k.lower() for k in request.headers}
    assert not any(h.startswith(("x-title", "http-referer", "x-app")) for h in sent)


def test_the_one_shot_client_never_sends_temperature_or_a_token_cap() -> None:
    with FakeCompatibleServer() as server:
        client = CompatibleClient(Endpoint("s", "S", server.url("open")), "fake-model")
        text = client.complete([{"role": "user", "content": "q"}], temperature=0.0, max_tokens=400)
    assert text.startswith("Retrieval gives")
    (request,) = server.chat_requests()
    assert set(request["body"]) == {"model", "messages", "stream"}


# ============================================================
# Test 2: the stream reader on recorded streams
# ============================================================


def test_the_reader_takes_content_and_ignores_the_rest() -> None:
    lines = _sse(
        ": keep-alive",
        "event: message",
        "id: 7",
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"reasoning": "hidden"}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "hidden too"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
        {"choices": [{"index": 0, "delta": {"content": None}}]},
        {"choices": [{"index": 0, "delta": {"content": " world"}}]},
        {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 2}},
        "data: [DONE]",
        {"choices": [{"index": 0, "delta": {"content": "after done"}}]},
    )
    assert list(read_stream(lines)) == ["Hello", " world", Usage(120, 2)]


def test_an_unreadable_chunk_is_skipped_not_fatal() -> None:
    lines = ["data: {not json", *_sse({"choices": [{"delta": {"content": "ok"}}]})]
    assert list(read_stream(lines)) == ["ok"]


def test_an_error_object_mid_stream_raises() -> None:
    lines = _sse(
        {"choices": [{"delta": {"content": "half"}}]},
        {"error": {"message": "context length exceeded"}},
    )
    events = read_stream(lines)
    assert next(events) == "half"
    with pytest.raises(CompatibleError, match="context length exceeded"):
        next(events)


def _client(response: httpx.Response) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda _r: response))


_POINT = Endpoint("s", "S", "http://localhost:1234/v1")


def test_a_4xx_with_json_names_the_servers_reason() -> None:
    client = _client(httpx.Response(400, json={"error": {"message": "unknown field: tools"}}))
    with pytest.raises(CompatibleError, match="answered 400: unknown field: tools") as info:
        list(stream_chat(_POINT, "m", [], client=client))
    assert info.value.status == 400


def test_a_5xx_with_html_never_echoes_the_page() -> None:
    client = _client(httpx.Response(502, text="<html><body>Bad gateway</body></html>"))
    with pytest.raises(CompatibleError) as info:
        list(stream_chat(_POINT, "m", [], client=client))
    assert info.value.reason == "the server answered 502"


class _Dropped(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b'data: {"choices": [{"delta": {"content": "half"}}]}\n\n'
        raise httpx.ReadError("connection reset")


def test_a_dropped_connection_mid_answer_is_an_error_not_a_short_answer() -> None:
    client = _client(httpx.Response(200, stream=_Dropped()))
    events = stream_chat(_POINT, "m", [], client=client)
    assert next(events) == "half"
    with pytest.raises(CompatibleError, match="dropped"):
        next(events)


def test_a_server_that_cannot_be_reached_raises_connect_error_as_it_is() -> None:
    # The turn's failure sentence matches it by name (`helpers._is_unreachable`).
    def refuse(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    with pytest.raises(httpx.ConnectError):
        list(stream_chat(_POINT, "m", [], client=client))


# ============================================================
# Test 6: model lists
# ============================================================


def _lister(status: int, **kwargs: object) -> httpx.Client:
    return _client(httpx.Response(status, **kwargs))  # type: ignore[arg-type]


def test_a_list_in_the_norms_shape_with_windows() -> None:
    client = _lister(
        200,
        json={
            "data": [
                {"id": "b", "context_window": 131072},
                {"id": "a", "context_length": 8192},
                {"id": "a"},
                {"no": "id"},
            ]
        },
    )
    listing = list_models("https://api.groq.com/openai/v1", "k", client=client)
    assert (listing.status, listing.models) == ("ok", ("a", "b"))
    assert listing.windows == {"a": 8192, "b": 131072}
    assert listing.fetched_at


def test_a_bare_array_as_mistrals_docs_show_it() -> None:
    client = _lister(200, json=[{"id": "mistral-small", "max_context_length": 32768}])
    listing = list_models("https://api.mistral.ai/v1", "k", client=client)
    assert (listing.models, listing.windows) == (("mistral-small",), {"mistral-small": 32768})


@pytest.mark.parametrize(
    ("status", "key", "expected"),
    [
        (401, "k", "refused"),
        (403, "k", "refused"),
        (400, "k", "refused"),  # a GET with no parameters is refused for its key (Gemini)
        (401, None, "needs_key"),
        (404, "k", "error"),
        (500, "k", "error"),
    ],
)
def test_list_statuses(status: int, key: str | None, expected: str) -> None:
    assert list_models("https://x.example/v1", key, client=_lister(status)).status == expected


def test_a_404_says_to_check_the_path() -> None:
    assert "/v1" in list_models("https://x.example", "k", client=_lister(404)).detail


def test_a_timeout_is_unreachable_and_never_raises() -> None:
    def slow(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    client = httpx.Client(transport=httpx.MockTransport(slow))
    listing = list_models("https://api.groq.com/openai/v1", "k", client=client)
    assert (listing.status, listing.detail) == ("unreachable", "Could not reach api.groq.com.")


def test_bad_json_is_an_error_not_an_empty_list() -> None:
    listing = list_models("https://x.example/v1", "k", client=_lister(200, text="<html>"))
    assert listing.status == "error"


def test_a_key_check_path_refuses_a_bad_key_even_when_the_list_is_open() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/key"):
            return httpx.Response(401)
        return httpx.Response(200, json={"data": [{"id": "m"}]})  # OpenRouter's open list

    client = httpx.Client(transport=httpx.MockTransport(handler))
    listing = list_models("https://openrouter.ai/api/v1", "bad", key_check="/key", client=client)
    assert listing.status == "refused"


def test_no_log_line_carries_a_key() -> None:
    with structlog.testing.capture_logs() as logs:
        list_models("https://x.example/v1", "sk-secret-123", client=_lister(500))
        list_models("https://x.example/v1", "sk-secret-123", client=_lister(200, text="x"))
    assert "sk-secret-123" not in repr(logs)


# ============================================================
# The two call shapes over a real socket
# ============================================================


def test_the_chat_model_streams_and_reports_usage_to_the_counter() -> None:
    from langchain_core.messages import HumanMessage

    with FakeCompatibleServer(usage=(1500, 30)) as server:
        model = CompatibleChatModel(
            provider_id="s", name_shown="S", address=server.url("open"), model="fake-model"
        )
        counter = TokenCounter()
        text = "".join(
            c.content  # type: ignore[misc]
            for c in model.stream([HumanMessage("q")], config={"callbacks": [counter]})
        )
    assert text.startswith("Retrieval gives the model passages")
    assert (counter.input_tokens, counter.output_tokens) == (1500, 30)
    (request,) = server.chat_requests()
    assert request["body"]["messages"] == [{"role": "user", "content": "q"}]
    assert "authorization" not in request["headers"]


def test_the_chat_model_invoke_joins_the_stream() -> None:
    from langchain_core.messages import HumanMessage

    with FakeCompatibleServer(usage=None) as server:
        model = CompatibleChatModel(
            provider_id="s", name_shown="S", address=server.url("open"), model="fake-model"
        )
        result = model.invoke([HumanMessage("q")])
    assert result.content == "".join(
        (
            "Retrieval gives the model passages to ground its answer in [1], ",
            "which lowers unsupported claims [2].",
        )
    )


# ============================================================
# TLS in the frozen build (KI-10's path)
# ============================================================


def test_the_frozen_build_verifies_tls_against_the_os_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import truststore

    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(compatible.httpx, "Client", lambda **kw: captured.update(kw) or kw)
    compatible.http_client(5.0)
    assert isinstance(captured["verify"], truststore.SSLContext)


def test_a_dev_run_keeps_httpx_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(compatible.httpx, "Client", lambda **kw: captured.update(kw) or kw)
    compatible.http_client(5.0)
    assert "verify" not in captured
