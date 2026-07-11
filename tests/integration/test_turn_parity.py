"""Parity gate (PR-M0 ADR-1): every frontend renders the same ``TurnResult``.

The migration's core guarantee is behavioural identity — the CLI and the FastAPI/Tauri
renderers all consume one ``TurnEvent`` stream and must produce the same answer, the same
source citations, the same provenance id, and the same flagged-claim set. Here we drive one
controller turn, capture the event stream, feed it to a minimal harness mirroring the CLI's
consumption logic (no stdout), then assert it agrees with the canonical content the
structured ``TurnResult`` exposes (exactly what the API/Tauri serialize).

Deterministic: a fake pipeline supplies fixed retrieval + a fixed answer; a temp SQLite
backs provenance. No network, no corpus, no paid call (cpc §13).
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.documents import Document

from doc_assistant import chat_controller
from doc_assistant.chat_controller import (
    ChatController,
    Result,
    Session,
    Token,
    TurnResult,
)


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    test_engine = create_engine(f"sqlite:///{tmp_path / 'parity.db'}", future=True)
    Base.metadata.create_all(test_engine)
    factory = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", test_engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield tmp_path
    test_engine.dispose()


def _doc(text: str, **metadata: object) -> Document:
    return Document(page_content=text, metadata=metadata)


class FakeRAG:
    def __init__(self, scored: list[tuple[Document, float]], tokens: list[str]) -> None:
        self._scored = scored
        self._tokens = tokens
        self.llm = types.SimpleNamespace(model="fake-model")
        # ADR-011 (U1c): the effective provider/model chat_controller._is_local reads.
        self.provider = "anthropic"
        self.model = "claude-haiku-4-5-20251001"

    def chunk_count(self) -> int:
        return 7

    def rewrite(self, question: str, history: list[dict[str, str]], counter: object = None) -> str:
        return question

    def retrieve_with_scores(
        self, query: str, top_k: int = 10, *, use_multi_query: bool | None = None
    ) -> list[tuple[Document, float]]:
        return self._scored

    def stream_answer(
        self, question: str, docs: list[Document], counter: object = None
    ) -> Iterator[str]:
        yield from self._tokens


def _sources() -> list[tuple[Document, float]]:
    return [
        (
            _doc(
                "Neurons meet at synapses.",
                document_id="d1",
                chunk_index=0,
                filename="a.pdf",
                page=1,
            ),
            0.91,
        ),
        (
            _doc(
                "Discrete cells, not a net.",
                document_id="d2",
                chunk_index=1,
                filename="b.pdf",
                page=2,
            ),
            0.88,
        ),
    ]


def _events(tokens: list[str], temp_db, monkeypatch) -> tuple[list[object], TurnResult]:
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    controller = ChatController(rag=FakeRAG(_sources(), tokens))
    events = list(controller.handle_message(Session(), "How do neurons connect?"))
    result = next(e.result for e in events if isinstance(e, Result))
    return events, result


# --- minimal renderers (mirror the real apps' consumption of the stream) ---


def _canonical(result: TurnResult) -> str:
    """The content concatenation both renderers build from a TurnResult."""
    return (
        result.answer
        + result.sources_md
        + result.usage_md
        + result.provenance_card_md
        + result.claim_review_md
        + result.citation_note_md
    )


def _render_cli(events: list[object]) -> str:
    """Mirror apps/cli.py: stream tokens, then append the TurnResult's blocks."""
    streamed = ""
    blocks = ""
    for ev in events:
        if isinstance(ev, Token):
            streamed += ev.text
        elif isinstance(ev, Result):
            r = ev.result
            streamed = streamed or r.answer  # non-streamed turns print the answer directly
            blocks = (
                r.sources_md
                + r.usage_md
                + r.provenance_card_md
                + r.claim_review_md
                + r.citation_note_md
            )
    return streamed + blocks


# ============================================================
# Tests
# ============================================================


def test_renderers_render_identical_content(temp_db, monkeypatch):
    events, result = _events(
        ["Neurons meet at synapses [1]. ", "Discrete cells [2]."], temp_db, monkeypatch
    )

    cli = _render_cli(events)
    canonical = _canonical(result)

    # Byte-identical content across renderers (the parity guarantee): the CLI render equals
    # the canonical content the API/Tauri serialize from the TurnResult.
    assert cli == canonical

    # The salient facts appear in the rendering.
    assert result.answer in cli
    for s in result.sources:
        assert s.citation in cli
    assert result.record_id is not None
    assert result.record_id[:8] in cli


def test_flagged_claims_agree_across_renderers(temp_db, monkeypatch):
    # An uncited sentence is flagged; both renderers carry the same review block.
    events, result = _events(["Neurons are continuous."], temp_db, monkeypatch)

    assert len(result.flagged_claims) == 1
    cli = _render_cli(events)
    canonical = _canonical(result)
    assert cli == canonical
    # The flagged claim's text surfaces in the shared review block.
    assert result.flagged_claims[0].text in cli
    assert "claim(s) to review" in cli and "claim(s) to review" in canonical


def test_byte_identical_when_markers_absent(temp_db, monkeypatch):
    """PR-M1 ADR-2: with the epistemics sidecar absent/empty, the 7d marker join is a
    no-op — no chip, every `markers` empty, and `sources_md` is the citation-only form.
    This is the eval-comparability guarantee (markers must not perturb a clean turn)."""
    monkeypatch.setattr(chat_controller, "load_epistemics_index", lambda: {})
    monkeypatch.setattr(chat_controller, "load_marked_chunks", lambda ids: {})
    _, result = _events(["Neurons meet at synapses [1]."], temp_db, monkeypatch)

    assert all(s.markers == [] for s in result.sources)
    assert "⚠" not in result.sources_md
    # sources_md is exactly the citations joined — the M0 baseline form.
    expected = "\n\n---\n**Sources:**\n" + "\n".join(s.citation for s in result.sources)
    assert result.sources_md == expected
