"""A Claude model from Opus 4.7 on, driven offline through the real client classes.

Those models answer ``temperature`` with a 400, and from Claude Opus 5 they think by default: the
reply opens with a thinking block, and the thinking spends the same ``max_tokens`` as the text. The
default, Claude Haiku 4.5, does neither, and the reviewer and judge must keep their temperature 0.

No network: ``ChatAnthropic`` runs for real over a fake stream of the SDK's own event types.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types import (
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta
from langchain_core.documents import Document

from doc_assistant import llm
from doc_assistant.pipeline import RAGPipeline, build_chat_model

OLDER = [
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-latest",
]
NEWER = [
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-1",
    "claude-mythos-5-1",
]


def _thinking_stream(model: str, text: str) -> Iterator[Any]:
    """The events of a reply that thinks first, display omitted, as these models send it."""
    usage = Usage(input_tokens=10, output_tokens=1)
    message = Message(
        id="msg_1",
        content=[],
        model=model,
        role="assistant",
        type="message",
        usage=usage,
    )
    yield RawMessageStartEvent(message=message, type="message_start")
    yield RawContentBlockStartEvent(
        content_block=ThinkingBlock(signature="", thinking="", type="thinking"),
        index=0,
        type="content_block_start",
    )
    yield RawContentBlockDeltaEvent(
        delta=ThinkingDelta(thinking="", type="thinking_delta"),
        index=0,
        type="content_block_delta",
    )
    yield RawContentBlockDeltaEvent(
        delta=SignatureDelta(signature="sig", type="signature_delta"),
        index=0,
        type="content_block_delta",
    )
    yield RawContentBlockStopEvent(index=0, type="content_block_stop")
    yield RawContentBlockStartEvent(
        content_block=TextBlock(text="", type="text"),
        index=1,
        type="content_block_start",
    )
    for piece in text.split(" "):
        yield RawContentBlockDeltaEvent(
            delta=TextDelta(text=piece + " ", type="text_delta"),
            index=1,
            type="content_block_delta",
        )
    yield RawContentBlockStopEvent(index=1, type="content_block_stop")
    yield RawMessageDeltaEvent(
        delta=Delta(stop_reason="end_turn"),
        type="message_delta",
        usage=MessageDeltaUsage(output_tokens=20),
    )
    yield RawMessageStopEvent(type="message_stop")


@pytest.fixture
def thinking_chat(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real ``ChatAnthropic`` on a newer model whose requests go to the fake stream."""
    from langchain_anthropic import ChatAnthropic

    requests: list[dict[str, Any]] = []

    def create(**payload: Any) -> Iterator[Any]:
        requests.append(payload)
        return _thinking_stream(payload["model"], "the answer [1]")

    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(ChatAnthropic, "_client", fake)
    return SimpleNamespace(
        model=build_chat_model("anthropic", "claude-opus-5-5"), requests=requests
    )


def _pipeline(chat_model: Any) -> RAGPipeline:
    rag = RAGPipeline.__new__(RAGPipeline)
    rag.llm = chat_model
    return rag


def test_the_streamed_answer_is_text_only(thinking_chat: Any) -> None:
    """The controller does ``full_answer += tok``, so one list-shaped chunk fails the turn."""
    tokens = list(
        _pipeline(thinking_chat.model).stream_answer("q", [Document(page_content="text")])
    )
    assert all(isinstance(t, str) for t in tokens)
    assert "".join(tokens).strip() == "the answer [1]"


def test_a_rewritten_question_is_text_only(thinking_chat: Any) -> None:
    assert (
        _pipeline(thinking_chat.model).rewrite("q", [{"role": "user", "content": "hi"}]).strip()
        == "the answer [1]"
    )


def test_expand_query_reads_the_text_of_a_thinking_reply(
    monkeypatch: pytest.MonkeyPatch, thinking_chat: Any
) -> None:
    from langchain_anthropic import ChatAnthropic

    def create(**payload: Any) -> Iterator[Any]:
        return _thinking_stream(payload["model"], '["one", "two"]')

    monkeypatch.setattr(
        ChatAnthropic, "_client", SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    assert _pipeline(thinking_chat.model).expand_query("q") == ["q", "one", "two"]


def test_the_chat_request_carries_no_sampling_params(thinking_chat: Any) -> None:
    list(_pipeline(thinking_chat.model).stream_answer("q", [Document(page_content="text")]))
    assert not {"temperature", "top_p", "top_k"} & thinking_chat.requests[0].keys()


@pytest.mark.parametrize("model", OLDER)
def test_older_models_are_recognised(model: str) -> None:
    assert llm.is_older_claude(model)


@pytest.mark.parametrize("model", [*NEWER, "claude-sonnet-6", "m", ""])
def test_anything_else_counts_as_newer(model: str) -> None:
    """Sending no ``temperature`` never fails, so a name this list has not met takes that side."""
    assert not llm.is_older_claude(model)


def test_chat_room_for_thinking_only_where_a_model_thinks() -> None:
    assert build_chat_model("anthropic", "claude-haiku-4-5").max_tokens == 1024
    assert build_chat_model("anthropic", "claude-opus-5-5").max_tokens > 1024


class _FakeMessages:
    def __init__(self, content: list[Any]) -> None:
        self.content = content
        self.request: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.request = kwargs
        return SimpleNamespace(content=self.content)


def _client(monkeypatch: pytest.MonkeyPatch, model: str, content: list[Any]) -> Any:
    messages = _FakeMessages(content)

    class _FakeAnthropic:
        def __init__(self, **_: Any) -> None:
            self.messages = messages

    monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)
    return llm.AnthropicClient(model, api_key="test-key"), messages  # pragma: allowlist secret


@pytest.mark.parametrize("model", OLDER)
def test_a_pinned_instrument_keeps_its_temperature(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    client, messages = _client(monkeypatch, model, [TextBlock(text="ok", type="text")])
    client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=400)
    assert messages.request["temperature"] == 0.0
    assert messages.request["max_tokens"] == 400
    assert "output_config" not in messages.request


@pytest.mark.parametrize("model", NEWER)
def test_a_newer_model_is_sent_no_temperature_and_has_room_to_think(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    client, messages = _client(monkeypatch, model, [TextBlock(text="ok", type="text")])
    client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=400)
    assert not {"temperature", "top_p", "top_k"} & messages.request.keys()
    assert messages.request["max_tokens"] > 400
    assert messages.request["output_config"] == {"effort": "low"}


def test_one_shot_text_is_found_behind_a_thinking_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading only the first block returned the response's repr, which then failed to parse."""
    blocks = [
        ThinkingBlock(signature="sig", thinking="", type="thinking"),
        TextBlock(text='  {"ok": true}  ', type="text"),
    ]
    client, _ = _client(monkeypatch, "claude-opus-5-5", blocks)
    reply = client.complete([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=400)
    assert reply == '{"ok": true}'


def test_a_reply_with_no_text_is_empty_not_a_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """All thinking and no text, as when ``max_tokens`` runs out: the caller reports it as such."""
    blocks = [ThinkingBlock(signature="sig", thinking="", type="thinking")]
    client, _ = _client(monkeypatch, "claude-opus-5-5", blocks)
    assert client.complete([{"role": "user", "content": "hi"}], temperature=0, max_tokens=9) == ""
