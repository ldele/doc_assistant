"""KI-65: a wiki topic's summary prompt fits the model's context window, at any topic size.

Before the fix the prompt carried every member's material (~175 tokens a paper, no
budget), so an 87-paper topic sent 15,245 tokens into Ollama's 4,096-token default and the
server silently kept only the end. These tests hold the bound with a fake client and a
fake chunk store — no network, no DB — at 0, 1, 20 and 10,000 documents (the robustness
contract), and check that a sampled note says so.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from doc_assistant import llm
from doc_assistant.config import WIKI_CONTEXT_TOKENS
from doc_assistant.knowledge import wiki
from doc_assistant.knowledge.wiki import (
    CHAT_FRAME_TOKENS,
    DocRef,
    TopicNote,
    compute_gap_signals,
    estimate_tokens,
    fit_window,
    material_budget,
    render_note_markdown,
    sample_order,
    summarize_cluster,
)

MAX_TOKENS = 400  # summarize_cluster's default answer budget

# A production-shaped excerpt: sample_chunks cuts each chunk to 300 characters. Non-ASCII
# on purpose (Greek, superscripts, micro sign), so the byte-based estimate is exercised.
_EXCERPT = (
    "Sharp-wave ripples (150-250 Hz) in CA1 coincide with replay of place-cell sequences; "
    "blocking them with optogenetic silencing of PV⁺ interneurons impaired β-band coupling "
    "and spatial memory consolidation in rats (n = 12, 40 µm tetrodes), while Ca²⁺ imaging "
    "showed the same cells reactivated during quiet wakefulness and slow-wave sleep."
)[:300]


def _paper(i: int) -> DocRef:
    return DocRef(
        doc_id=f"d{i}",
        doc_hash=f"hash{i:05d}",
        filename=f"paper_{i:05d}.pdf",
        title=f"Hippocampal replay and the consolidation of spatial memory, study {i}",
        year=1990 + i % 35,
        keywords=[f"keyword-{k}" for k in range(8)],
    )


class _FakeStore:
    """A ``ChunkFetcher`` that records which documents it was asked for."""

    def __init__(self) -> None:
        self.requested: list[str] = []

    def __call__(self, doc_ids: list[str]) -> dict[str, list[str]]:
        self.requested.extend(doc_ids)
        return {d: [_EXCERPT, _EXCERPT, _EXCERPT] for d in doc_ids}


class _FakeClient:
    """An ``LLMClient``-shaped fake that records each prompt."""

    def __init__(self, reply: str = '{"title": "Replay", "summary": "S.", "tags": ["a"]}'):
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
        assert max_tokens == MAX_TOKENS
        self.prompts.append(messages[0]["content"])
        return self.reply


def _fits(prompt: str, context_tokens: int) -> bool:
    return estimate_tokens(prompt) + CHAT_FRAME_TOKENS + MAX_TOKENS <= context_tokens


# ============================================================
# The budget at 0, 1, 20 and 10,000 documents
# ============================================================


def test_zero_documents_make_no_call():
    client = _FakeClient()
    out = summarize_cluster([], _FakeStore(), client)
    assert client.prompts == []
    assert (out.summary, out.summarised) == ("", [])
    assert out.title == "Untitled topic"


def test_one_document_is_sent_whole():
    client, store = _FakeClient(), _FakeStore()
    out = summarize_cluster([_paper(0)], store, client)
    (prompt,) = client.prompts
    assert "PAPERS IN THIS TOPIC:" in prompt  # not the sampled heading
    assert _paper(0).title in prompt and _EXCERPT in prompt
    assert out.summarised == ["d0"]
    assert _fits(prompt, WIKI_CONTEXT_TOKENS)


def test_twenty_documents_all_fit_the_default_window():
    """The size at which KI-65 began to cut (~20 papers) is now sent in full."""
    docs = [_paper(i) for i in range(20)]
    client = _FakeClient()
    out = summarize_cluster(docs, _FakeStore(), client)
    (prompt,) = client.prompts
    assert "a sample of" not in prompt
    assert sorted(out.summarised) == sorted(d.doc_id for d in docs)
    assert all(d.title in prompt for d in docs if d.title)
    assert _fits(prompt, WIKI_CONTEXT_TOKENS)


def test_ten_thousand_documents_are_sampled_inside_the_window():
    docs = [_paper(i) for i in range(10_000)]
    client, store = _FakeClient(), _FakeStore()
    out = summarize_cluster(docs, store, client)
    (prompt,) = client.prompts
    shown = len(out.summarised)
    assert 20 <= shown < 10_000
    assert f"PAPERS IN THIS TOPIC (a sample of {shown} of 10000):" in prompt
    assert _fits(prompt, WIKI_CONTEXT_TOKENS)
    # The store is read for the sample (plus at most one batch), not for 10,000 papers.
    assert len(store.requested) <= shown + wiki._FETCH_BATCH


@pytest.mark.parametrize("n_docs", [0, 1, 2, 20, 87, 10_000])
@pytest.mark.parametrize("context_tokens", [1024, 2048, 4096, 8192, 32768])
def test_prompt_plus_answer_never_exceed_the_window(n_docs: int, context_tokens: int):
    client = _FakeClient()
    out = summarize_cluster(
        [_paper(i) for i in range(n_docs)], _FakeStore(), client, context_tokens=context_tokens
    )
    for prompt in client.prompts:
        assert _fits(prompt, context_tokens)
    if n_docs:
        assert len(client.prompts) == 1
        assert 1 <= len(out.summarised) <= n_docs


# ============================================================
# Degraded inputs
# ============================================================


def test_a_window_too_small_for_the_fixed_text_makes_no_call():
    client = _FakeClient()
    out = summarize_cluster([_paper(0)], _FakeStore(), client, context_tokens=MAX_TOKENS)
    assert client.prompts == []
    assert out.summarised == []
    assert material_budget(1, context_tokens=MAX_TOKENS, max_tokens=MAX_TOKENS) == 0


def test_a_block_larger_than_the_budget_is_clipped_not_dropped():
    giant = DocRef(doc_id="g", doc_hash="hg", filename="g.pdf", title="é" * 50_000)
    client = _FakeClient()
    out = summarize_cluster([giant], _FakeStore(), client, context_tokens=1024)
    (prompt,) = client.prompts
    assert _fits(prompt, 1024)
    assert out.summarised == ["g"]


def test_a_failed_call_claims_no_sample():
    class _Down:
        def complete(self, *a: Any, **k: Any) -> str:
            raise ConnectionError("ollama is not running")

    out = summarize_cluster([_paper(i) for i in range(100)], _FakeStore(), _Down())
    assert (out.summary, out.summarised) == ("", [])


def test_an_empty_summary_claims_no_sample():
    client = _FakeClient(reply='{"title": "T", "summary": "", "tags": []}')
    out = summarize_cluster([_paper(i) for i in range(100)], _FakeStore(), client)
    assert out.summarised == []


# ============================================================
# Determinism of the sample
# ============================================================


def test_sample_order_ignores_input_order():
    docs = [_paper(i) for i in range(200)]
    shuffled = docs[:]
    random.Random(7).shuffle(shuffled)
    assert [d.doc_id for d in sample_order(docs)] == [d.doc_id for d in sample_order(shuffled)]


def test_the_sample_is_the_same_on_every_run_and_spread_over_the_topic():
    docs = [_paper(i) for i in range(1000)]
    first = summarize_cluster(docs, _FakeStore(), _FakeClient()).summarised
    shuffled = docs[:]
    random.Random(3).shuffle(shuffled)
    assert summarize_cluster(shuffled, _FakeStore(), _FakeClient()).summarised == first
    # Not a block at either end of the list: KI-65's truncation kept the last papers only.
    positions = sorted(int(d[1:]) for d in first)
    assert positions[0] < 250 and positions[-1] > 750


# ============================================================
# The note says when it was written from a sample
# ============================================================


def _note(n_docs: int, summarised: list[str], summary: str = "A summary.") -> TopicNote:
    docs = [_paper(i) for i in range(n_docs)]
    return TopicNote(
        topic_id="topic-x",
        title="Replay",
        docs=docs,
        summary=summary,
        tags=[],
        links=[],
        gap=compute_gap_signals(n_docs, 1),
        summarised=summarised,
    )


def test_a_sampled_note_says_how_many_papers_it_read():
    md = render_note_markdown(_note(87, [f"d{i}" for i in range(20)]))
    assert "summary_from: 20" in md
    assert "_Summary written from 20 of 87 papers" in md
    assert md.count("· *in summary*") == 20


def test_a_complete_or_absent_summary_adds_no_sample_line():
    full = render_note_markdown(_note(3, ["d0", "d1", "d2"]))
    none = render_note_markdown(_note(3, [], summary=""))
    for md in (full, none):
        assert "summary_from" not in md
        assert "Summary written from" not in md
        assert "in summary" not in md


# ============================================================
# The window an Ollama client asks for (fit_window)
# ============================================================


@pytest.mark.parametrize(
    ("trained", "own", "expected"),
    [
        (32768, None, 8192),  # the wiki's window, asked for explicitly
        (2048, None, 2048),  # a model trained shorter: the server would clamp, then cut
        (None, None, 8192),  # server silent: still explicit, with a warning
        (32768, 4096, 4096),  # a caller's smaller window is kept
    ],
)
def test_fit_window_gives_an_ollama_client_an_explicit_num_ctx(
    monkeypatch: pytest.MonkeyPatch, trained: int | None, own: int | None, expected: int
):
    monkeypatch.setattr(llm, "ollama_context_length", lambda model, host=None, **k: trained)
    client, window = fit_window(llm.OllamaClient("qwen2.5:7b", num_ctx=own), 8192)
    assert isinstance(client, llm.OllamaClient)
    assert window == expected
    assert client.num_ctx == expected


def test_fit_window_leaves_other_clients_alone(monkeypatch: pytest.MonkeyPatch):
    def _no_network(*a: Any, **k: Any) -> int:
        raise AssertionError("only an Ollama client is asked about its model")

    monkeypatch.setattr(llm, "ollama_context_length", _no_network)
    fake = _FakeClient()
    assert fit_window(fake, 8192) == (fake, 8192)
