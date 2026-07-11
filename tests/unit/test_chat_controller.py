"""Tests for the UI-agnostic turn controller (PR-M0).

No live LLM, no real corpus: a fake ``RAGPipeline`` supplies retrieval +
streamed tokens, and the DB-touching paths run against a temp SQLite (the ``temp_db``
fixture) or are monkeypatched. These exercise the dispatch order, the ADR-2 chunk-key
derivation, the TurnResult shape, claim adjudication, and the provenance-failure path.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.documents import Document

from doc_assistant import chat_controller, config
from doc_assistant.chat_controller import (
    ChatController,
    RagOverrides,
    Result,
    Session,
    Token,
    TurnResult,
    _build_retrieved_chunks,
)
from doc_assistant.epistemics import MARKER_CONTESTED, MARKER_SUPERSEDED, MarkedChunk
from doc_assistant.reviewer import ReviewResult

# ============================================================
# Fixtures / fakes
# ============================================================


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the SQLAlchemy engine at a temp SQLite file and create schema."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    test_engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(test_engine)
    factory = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", test_engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield tmp_path
    test_engine.dispose()


def _doc(text: str, **metadata: object) -> Document:
    return Document(page_content=text, metadata=metadata)


class FakeRAG:
    """Stand-in for RAGPipeline — no models, no DB, deterministic."""

    def __init__(self, scored: list[tuple[Document, float]], tokens: list[str]) -> None:
        self._scored = scored
        self._tokens = tokens
        self.llm = types.SimpleNamespace(model="fake-model")
        # ADR-010: every retrieve_with_scores call's (top_k, use_multi_query) args, in order —
        # lets a test assert what actually reached the pipeline without a real one.
        self.retrieve_calls: list[tuple[int, bool | None]] = []
        # ADR-011 (U1c): the effective provider/model + a set_chat_model spy.
        self.provider = "anthropic"
        self.model = "claude-haiku-4-5-20251001"
        self.set_chat_model_calls: list[tuple[str, str]] = []

    def chunk_count(self) -> int:
        return 42

    def set_chat_model(self, provider: str, model: str) -> None:
        self.set_chat_model_calls.append((provider, model))
        self.provider = provider
        self.model = model

    def rewrite(self, question: str, history: list[dict[str, str]], counter: object = None) -> str:
        return question

    def retrieve_with_scores(
        self, query: str, top_k: int = 10, *, use_multi_query: bool | None = None
    ) -> list[tuple[Document, float]]:
        self.retrieve_calls.append((top_k, use_multi_query))
        return self._scored

    def stream_answer(
        self, question: str, docs: list[Document], counter: object = None
    ) -> Iterator[str]:
        yield from self._tokens


def _three_clean_sources() -> list[tuple[Document, float]]:
    """Three distinct, high-score sources → no confidence signal fires → no reviewer."""
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
                "Signals cross the cleft.",
                document_id="d2",
                chunk_index=4,
                filename="b.pdf",
                page=2,
            ),
            0.88,
        ),
        (
            _doc(
                "Discrete cells, not a net.",
                document_id="d3",
                chunk_index=2,
                filename="c.pdf",
                page=3,
            ),
            0.85,
        ),
    ]


def _results(
    controller: ChatController,
    session: Session,
    text: str,
    overrides: RagOverrides | None = None,
) -> list[object]:
    return list(controller.handle_message(session, text, overrides=overrides))


def _final(events: list[object]) -> TurnResult:
    results = [e.result for e in events if isinstance(e, Result)]
    assert len(results) == 1, f"expected exactly one Result, got {len(results)}"
    return results[0]


# ============================================================
# ADR-2 — chunk_key derivation (pure; no DB)
# ============================================================


def test_chunk_key_flat_chunk_uses_epistemics_format():
    scored = [(_doc("x", document_id="doc1", chunk_index=3, filename="a.pdf"), 0.5)]
    chunks = _build_retrieved_chunks(scored)
    assert chunks[0].chunk_key == "doc1:3"  # matches epistemics.load_epistemics_index


def test_chunk_key_zero_index_is_not_dropped():
    # chunk_index 0 is falsy — guards the `is not None` check, not truthiness.
    scored = [(_doc("x", document_id="doc1", chunk_index=0), 0.5)]
    assert _build_retrieved_chunks(scored)[0].chunk_key == "doc1:0"


def test_chunk_key_parent_child_chunk_is_none():
    # PC parents carry parent_index, never chunk_index → None (M1 owns the mapping).
    scored = [(_doc("x", document_id="doc1", parent_index=2, filename="a.pdf"), 0.5)]
    assert _build_retrieved_chunks(scored)[0].chunk_key is None


def test_chunk_key_missing_document_id_is_none():
    scored = [(_doc("x", chunk_index=3), 0.5)]
    assert _build_retrieved_chunks(scored)[0].chunk_key is None


def test_chunk_key_is_not_persisted(temp_db, monkeypatch):
    # The join key is transient like full_text — never written to the answer record.
    from sqlalchemy import select

    from doc_assistant.db.models import AnswerRecord
    from doc_assistant.db.session import session_scope
    from doc_assistant.provenance import record_answer

    scored = [(_doc("x", document_id="doc1", chunk_index=3), 0.5)]
    rid = record_answer(query="q", answer="a", retrieved_chunks=_build_retrieved_chunks(scored))
    with session_scope() as s:
        raw = s.execute(
            select(AnswerRecord.retrieved_chunks_json).where(AnswerRecord.id == rid)
        ).scalar_one()
    assert "chunk_key" not in raw and "doc1:3" not in raw


# ============================================================
# Dispatch order
# ============================================================


def test_dispatch_slash_command_short_circuits(monkeypatch):
    monkeypatch.setattr(chat_controller, "execute_command", lambda cmd, arg: f"CMD::{cmd}::{arg}")
    controller = ChatController(rag=FakeRAG([], []))
    result = _final(_results(controller, Session(), "/library broken"))
    assert result.answer == "CMD::library::broken"
    assert result.sources == [] and result.record_id is None


def test_dispatch_pending_edit_routes_to_adjudication(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        chat_controller,
        "adjudicate_claim",
        lambda cid, decision, edited_text=None: calls.append((cid, decision, edited_text)),
    )
    controller = ChatController(rag=FakeRAG([], []))
    session = Session(awaiting_edit={"id": "claim-9", "n": 2})
    result = _final(_results(controller, session, "the corrected text"))
    assert calls == [("claim-9", "edited", "the corrected text")]
    assert result.answer == "✏️ Claim #2 updated."
    assert session.awaiting_edit is None  # reset


def test_dispatch_library_query(monkeypatch):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: True)
    monkeypatch.setattr(chat_controller, "answer_library_query", lambda t: "LIBRARY ANSWER")
    controller = ChatController(rag=FakeRAG([], []))
    result = _final(_results(controller, Session(), "how many pdfs?"))
    assert result.answer == "LIBRARY ANSWER"


def test_failing_command_is_surfaced_not_raised(monkeypatch):
    # A command that raises (e.g. an empty/missing DB) must yield a graceful error result,
    # never propagate out of the generator (which would break the SSE stream).
    def _boom(cmd, arg):
        raise RuntimeError("no such table: documents")

    monkeypatch.setattr(chat_controller, "execute_command", _boom)
    controller = ChatController(rag=FakeRAG([], []))
    result = _final(_results(controller, Session(), "/library"))
    assert "/library` failed" in result.answer and "no such table" in result.answer
    assert result.sources == [] and result.record_id is None


def test_failing_library_query_is_surfaced(monkeypatch):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: True)

    def _boom(text):
        raise RuntimeError("db gone")

    monkeypatch.setattr(chat_controller, "answer_library_query", _boom)
    controller = ChatController(rag=FakeRAG([], []))
    result = _final(_results(controller, Session(), "how many pdfs?"))
    assert "Library query failed" in result.answer and "db gone" in result.answer


def test_dispatch_rag_path_taken(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Cells ", "meet [1]."]))
    events = _results(controller, Session(), "How do neurons connect?")
    # Streamed tokens then exactly one Result (the RAG branch).
    assert [type(e).__name__ for e in events if isinstance(e, Token)] == ["Token", "Token"]
    result = _final(events)
    assert result.answer == "Cells meet [1]." and len(result.sources) == 3


# ============================================================
# TurnResult shape
# ============================================================


def test_ai_turn_result_shape(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    controller = ChatController(
        rag=FakeRAG(_three_clean_sources(), ["Neurons meet at synapses [1]."])
    )
    result = _final(_results(controller, Session(), "How do neurons connect?"))

    assert result.mode == "ai"
    assert result.answer == "Neurons meet at synapses [1]."
    assert [s.n for s in result.sources] == [1, 2, 3]
    assert result.sources[0].citation.startswith("[1] a.pdf")
    assert result.sources[0].chunk_key == "d1:0"  # flat chunk → epistemics key
    assert result.record_id is not None
    assert result.usage.is_local is False
    assert "**Sources:**" in result.sources_md
    assert "This turn:" in result.usage_md
    assert "Provenance" in result.provenance_card_md
    # Clean, cited claim → nothing flagged; the block says so.
    assert result.flagged_claims == []
    assert "grounded in cited evidence" in result.claim_review_md
    assert result.citation_note_md == ""  # citations are clean


def test_uncited_claim_is_flagged(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    # An uncited sentence → marker "unsupported" → surfaced for adjudication.
    controller = ChatController(
        rag=FakeRAG(_three_clean_sources(), ["Neurons are discrete cells."])
    )
    result = _final(_results(controller, Session(), "are neurons continuous?"))
    assert len(result.flagged_claims) == 1
    fc = result.flagged_claims[0]
    assert fc.n == 1 and fc.badge == "unsupported" and fc.claim_id
    assert "claim(s) to review" in result.claim_review_md


def test_human_mode_returns_evidence_only(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "SYNTHESIS_MODE", "human")
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["SHOULD NOT STREAM"]))
    events = _results(controller, Session(), "summarise the evidence")
    assert not any(isinstance(e, Token) for e in events)  # no interpretation stream
    result = _final(events)
    assert result.mode == "human"
    assert "Human synthesis mode" in result.answer
    assert result.flagged_claims == [] and result.record_id is None
    assert len(result.sources) == 3


# ============================================================
# adjudicate + provenance failure
# ============================================================


def test_adjudicate_passes_decision_and_edit(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        chat_controller,
        "adjudicate_claim",
        lambda cid, decision, edited_text=None: calls.append((cid, decision, edited_text)),
    )
    controller = ChatController(rag=FakeRAG([], []))
    controller.adjudicate("c1", "accepted")
    controller.adjudicate("c2", "edited", edited_text="fixed")
    assert calls == [("c1", "accepted", None), ("c2", "edited", "fixed")]


def test_provenance_failure_is_caught(monkeypatch):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "load_epistemics_index", lambda: {})  # no DB for the join

    def _boom(**kwargs):
        raise RuntimeError("db gone")

    monkeypatch.setattr(chat_controller, "record_answer", _boom)
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer text [1]."]))
    result = _final(_results(controller, Session(), "anything"))
    # Turn still completes with the answer; the failure rides in the card, record_id None.
    assert result.answer == "Answer text [1]."
    assert "Provenance capture failed" in result.provenance_card_md
    assert result.record_id is None and result.flagged_claims == []


# ============================================================
# PR-M1 — 7d marker surfacing (contested / superseded-trend chips)
# ============================================================


def _pc_sources() -> list[tuple[Document, float]]:
    """Three PC parents (parent_index, no chunk_index → chunk_key None)."""
    return [
        (
            _doc(
                "Long passage about discrete junctions in neurons.",
                document_id="d1",
                parent_index=0,
                filename="a.pdf",
                page=1,
            ),
            0.91,
        ),
        (
            _doc(
                "Signals cross the synaptic cleft.",
                document_id="d2",
                parent_index=1,
                filename="b.pdf",
                page=2,
            ),
            0.88,
        ),
        (
            _doc(
                "Cells are distinct units.",
                document_id="d3",
                parent_index=0,
                filename="c.pdf",
                page=3,
            ),
            0.85,
        ),
    ]


def test_markers_flat_join(monkeypatch, temp_db):
    # Flat chunk: direct join on chunk_key against the marker index.
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "EPISTEMICS_MARKERS_ENABLED", True)  # R7: off by default
    monkeypatch.setattr(
        chat_controller, "load_epistemics_index", lambda: {"d1:0": [MARKER_CONTESTED]}
    )
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer [1]."]))
    result = _final(_results(controller, Session(), "q"))
    assert result.sources[0].markers == [MARKER_CONTESTED]
    assert result.sources[1].markers == []  # d2:4 not in the index → quiet
    assert "⚠ contested in corpus" in result.sources_md  # chip in the shared block


def test_markers_pc_join_via_containment(monkeypatch, temp_db):
    # PC parent: no chunk_key → containment mapping (ADR-1).
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "EPISTEMICS_MARKERS_ENABLED", True)  # R7: off by default
    marked = {
        "d1": [MarkedChunk(chunk_index=0, text="discrete junctions", markers=[MARKER_SUPERSEDED])]
    }
    monkeypatch.setattr(chat_controller, "load_marked_chunks", lambda ids: marked)
    controller = ChatController(rag=FakeRAG(_pc_sources(), ["Answer [1]."]))
    result = _final(_results(controller, Session(), "q"))
    # d1's parent text contains "discrete junctions" → marker attached; others quiet.
    assert result.sources[0].markers == [MARKER_SUPERSEDED]
    assert result.sources[1].markers == [] and result.sources[2].markers == []
    assert "⚠ trend superseded" in result.sources_md


def test_markers_absent_is_byte_identical(monkeypatch, temp_db):
    # Sidecar empty → every markers empty → no chip → sources_md is the citation-only form.
    # (Flag on so this exercises the enabled-but-empty join path, not just the R7 gate.)
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "EPISTEMICS_MARKERS_ENABLED", True)
    monkeypatch.setattr(chat_controller, "load_epistemics_index", lambda: {})
    monkeypatch.setattr(chat_controller, "load_marked_chunks", lambda ids: {})
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer [1]."]))
    result = _final(_results(controller, Session(), "q"))
    assert all(s.markers == [] for s in result.sources)
    assert "⚠" not in result.sources_md  # quiet-on-clean → no chip


def test_marker_load_failure_does_not_break_turn(monkeypatch, temp_db):
    # Markers are advisory — a read failure (e.g. the chunk_epistemics table absent on an
    # older DB) must leave sources unmarked, never crash the turn / SSE stream.
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "EPISTEMICS_MARKERS_ENABLED", True)  # exercise the load

    def _boom():
        raise RuntimeError("no such table: chunk_epistemics")

    monkeypatch.setattr(chat_controller, "load_epistemics_index", _boom)
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer [1]."]))
    result = _final(_results(controller, Session(), "q"))
    assert result.answer == "Answer [1]."
    assert all(s.markers == [] for s in result.sources)


def test_markers_enabled_by_default(monkeypatch, temp_db):
    # KI-7 retirement / ADR-005 update (2026-07-07): EPISTEMICS_MARKERS_ENABLED now
    # defaults to True — markers rest on the concept-skeleton's Node A/B stance pass,
    # not the retired open-vocabulary graph. NOT setting the flag here — this exercises
    # the shipped default, not an explicit opt-in. A populated index (the skeleton
    # holding an opposing-stance edge) surfaces a chip; a clean index surfaces none.
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(
        chat_controller, "load_epistemics_index", lambda: {"d1:0": [MARKER_CONTESTED]}
    )
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer [1]."]))
    result = _final(_results(controller, Session(), "q"))
    assert result.sources[0].markers == [MARKER_CONTESTED]
    assert "⚠ contested in corpus" in result.sources_md

    monkeypatch.setattr(chat_controller, "load_epistemics_index", lambda: {})
    monkeypatch.setattr(chat_controller, "load_marked_chunks", lambda ids: {})
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer [1]."]))
    result = _final(_results(controller, Session(), "q"))
    assert all(s.markers == [] for s in result.sources)
    assert "⚠" not in result.sources_md


def test_markers_disabled_via_opt_out_flag(monkeypatch, temp_db):
    # The `false` opt-out still short-circuits the join entirely — even a populated
    # index leaves every source unmarked, no chip, and the load functions are never
    # called (the byte-identical M0/M1 no-marker path).
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "EPISTEMICS_MARKERS_ENABLED", False)
    loads: list[str] = []
    monkeypatch.setattr(
        chat_controller,
        "load_epistemics_index",
        lambda: loads.append("flat") or {"d1:0": [MARKER_CONTESTED]},
    )
    monkeypatch.setattr(
        chat_controller, "load_marked_chunks", lambda ids: loads.append("pc") or {}
    )
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer [1]."]))
    result = _final(_results(controller, Session(), "q"))
    assert all(s.markers == [] for s in result.sources)
    assert "⚠" not in result.sources_md
    assert loads == []  # gated out before any epistemics read


# ============================================================
# ADR-010 / SPRINT-010 (U1) — RAG-sandbox overrides
# ============================================================


def test_top_k_override_changes_retrieve_arg(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    rag = FakeRAG(_three_clean_sources(), ["Answer [1]."])
    controller = ChatController(rag=rag)
    _results(controller, Session(), "q", RagOverrides(top_k=3))
    assert rag.retrieve_calls == [(3, None)]  # eff_top_k=3 reached the pipeline call


def test_synthesis_mode_override_routes_to_human_even_when_default_is_ai(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "SYNTHESIS_MODE", "ai")  # locked default stays "ai"
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["SHOULD NOT STREAM"]))
    events = _results(controller, Session(), "q", RagOverrides(synthesis_mode="human"))
    assert not any(isinstance(e, Token) for e in events)  # no interpretation call made
    result = _final(events)
    assert result.mode == "human"


def test_overrides_none_reproduces_default_effective_values(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    rag = FakeRAG(_three_clean_sources(), ["Answer [1]."])
    controller = ChatController(rag=rag)
    # overrides=None (the default) — same as never passing overrides at all.
    result_none = _final(_results(controller, Session(), "q", None))
    result_omitted = _final(_results(controller, Session(), "q"))
    assert rag.retrieve_calls == [(10, None), (10, None)]  # both used the locked TOP_K
    assert result_none.answer == result_omitted.answer == "Answer [1]."
    # No "session override" note when every field is unset.
    assert "Session override" not in result_none.provenance_card_md
    assert "Session override" not in result_omitted.provenance_card_md


def test_all_none_fields_reproduce_default_effective_values(monkeypatch, temp_db):
    # An explicit RagOverrides() with every field None must be indistinguishable from
    # overrides=None — the "all fields None" case the spec calls out separately.
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    rag = FakeRAG(_three_clean_sources(), ["Answer [1]."])
    controller = ChatController(rag=rag)
    result = _final(_results(controller, Session(), "q", RagOverrides()))
    assert rag.retrieve_calls == [(10, None)]
    assert "Session override" not in result.provenance_card_md


def test_overrides_isolation_covers_all_five_fields(monkeypatch, temp_db):
    # The ⚠ correctness obligation (ADR-010 Decision 4 + the U1b amendment): a turn with all
    # five overrides set, then a turn with overrides=None, must use every locked default on
    # the second turn — proving no module-global was mutated. EPISTEMICS_MARKERS_ENABLED is
    # monkeypatched to False only to give the override something to differ from; nothing here
    # touches the module during the turns themselves (no monkeypatch in that path).
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "EPISTEMICS_MARKERS_ENABLED", False)
    monkeypatch.setattr(
        chat_controller, "load_epistemics_index", lambda: {"d1:0": [MARKER_CONTESTED]}
    )
    captured_full_text: list[str] = []
    real_record_answer = chat_controller.record_answer

    def spy(**kwargs):
        captured_full_text.append(kwargs["retrieved_chunks"][0].full_text)
        return real_record_answer(**kwargs)

    monkeypatch.setattr(chat_controller, "record_answer", spy)

    long_source = [(_doc("y" * 5000, document_id="d1", chunk_index=0, filename="a.pdf"), 0.9)]
    rag = FakeRAG(long_source, ["Answer [1]."])
    controller = ChatController(rag=rag)
    session = Session()

    overridden = _final(
        _results(
            controller,
            session,
            "q1",
            RagOverrides(
                top_k=2,
                use_multi_query=True,
                epistemics_markers_enabled=True,
                reviewer_evidence_chars=300,
            ),
        )
    )
    clean = _final(_results(controller, session, "q2", None))

    assert rag.retrieve_calls == [(2, True), (10, None)]  # second turn: locked defaults
    assert overridden.sources[0].markers == [MARKER_CONTESTED]  # override enabled it
    assert clean.sources[0].markers == []  # back to disabled (the locked default)
    assert len(captured_full_text[0]) == 300  # override
    assert len(captured_full_text[1]) == 1500  # back to REVIEWER_EVIDENCE_CHARS's default
    assert "Session override" in overridden.provenance_card_md
    assert "Session override" not in clean.provenance_card_md


def test_epistemics_markers_override_per_turn(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(chat_controller, "EPISTEMICS_MARKERS_ENABLED", False)  # locked: off
    monkeypatch.setattr(
        chat_controller, "load_epistemics_index", lambda: {"d1:0": [MARKER_CONTESTED]}
    )
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer [1]."]))
    result = _final(
        _results(controller, Session(), "q", RagOverrides(epistemics_markers_enabled=True))
    )
    assert result.sources[0].markers == [MARKER_CONTESTED]
    assert "epistemics_markers_enabled=True (default False)" in result.provenance_card_md


def test_reviewer_evidence_chars_override_per_turn(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    captured: dict[str, str] = {}
    real_record_answer = chat_controller.record_answer

    def spy(**kwargs):
        captured["full_text"] = kwargs["retrieved_chunks"][0].full_text
        return real_record_answer(**kwargs)

    monkeypatch.setattr(chat_controller, "record_answer", spy)
    long_source = [(_doc("z" * 5000, document_id="d1", chunk_index=0, filename="a.pdf"), 0.9)]
    controller = ChatController(rag=FakeRAG(long_source, ["Answer [1]."]))
    result = _final(
        _results(controller, Session(), "q", RagOverrides(reviewer_evidence_chars=500))
    )
    assert len(captured["full_text"]) == 500
    assert "reviewer_evidence_chars=500 (default 1500)" in result.provenance_card_md


def test_overrides_note_flags_only_the_differing_fields(monkeypatch, temp_db):
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    controller = ChatController(rag=FakeRAG(_three_clean_sources(), ["Answer [1]."]))
    result = _final(_results(controller, Session(), "q", RagOverrides(top_k=5)))
    assert "top_k=5 (default 10)" in result.provenance_card_md
    assert "synthesis_mode" not in result.provenance_card_md.split("Session override")[-1]


# ============================================================
# ADR-011 / SPRINT-012 (U1c) — desktop provider switch
# ============================================================


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the persisted user settings to a temp file — same pattern as
    tests/integration/test_api_settings_ingest.py, never the real data home."""
    monkeypatch.setattr(chat_controller.app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    return tmp_path


class _FakeRAGForConstruction:
    """Stands in for RAGPipeline ONLY to test ChatController.__init__'s apply-persisted-
    selection step, without loading any real model/embedder/reranker."""

    def __init__(self) -> None:
        self.provider = "anthropic"
        self.model = "claude-haiku-4-5-20251001"
        self.set_chat_model_calls: list[tuple[str, str]] = []

    def set_chat_model(self, provider: str, model: str) -> None:
        self.set_chat_model_calls.append((provider, model))
        self.provider = provider
        self.model = model


def test_persisted_selection_applied_at_construction(settings_file, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)  # ollama needs no key
    chat_controller.app_settings.set_llm_selection("ollama", "llama3.1:8b")
    monkeypatch.setattr(chat_controller, "RAGPipeline", _FakeRAGForConstruction)

    controller = ChatController()  # no injected rag → applies the persisted selection

    assert controller.rag.set_chat_model_calls == [("ollama", "llama3.1:8b")]
    assert (controller.rag.provider, controller.rag.model) == ("ollama", "llama3.1:8b")


def test_no_persisted_selection_skips_the_swap_at_construction(settings_file, monkeypatch):
    # Nothing persisted → the fresh RAGPipeline's own boot default is left alone (no needless
    # rebuild on the common no-switch boot).
    monkeypatch.setattr(chat_controller, "RAGPipeline", _FakeRAGForConstruction)
    controller = ChatController()
    assert controller.rag.set_chat_model_calls == []


def test_injected_rag_skips_the_apply_persisted_step(settings_file, monkeypatch):
    # A test-injected fake (cpc §13) must never be silently reconfigured by a leftover
    # persisted selection from a previous test/run.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    chat_controller.app_settings.set_llm_selection("ollama", "llama3.1:8b")
    rag = FakeRAG([], [])
    ChatController(rag=rag)
    assert rag.set_chat_model_calls == []


def test_reconfigure_persists_and_swaps_no_global_mutation(settings_file, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)  # ollama needs no key
    original_llm_provider = config.LLM_PROVIDER
    rag = FakeRAG([], [])
    controller = ChatController(rag=rag)

    controller.reconfigure("ollama", "llama3.1:8b")

    assert rag.set_chat_model_calls == [("ollama", "llama3.1:8b")]
    assert (rag.provider, rag.model) == ("ollama", "llama3.1:8b")
    assert chat_controller.app_settings.get_llm_selection() == ("ollama", "llama3.1:8b")
    assert original_llm_provider == config.LLM_PROVIDER  # no global mutation


def test_reconfigure_rejects_keyless_provider_and_does_not_swap(settings_file, monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    rag = FakeRAG([], [])
    controller = ChatController(rag=rag)
    with pytest.raises(ValueError, match="no credential"):
        controller.reconfigure("anthropic", "claude-haiku-4-5-20251001")
    assert rag.set_chat_model_calls == []  # rejected before the pipeline was ever touched
    assert chat_controller.app_settings.get_llm_selection() == (None, None)


def test_is_local_reflects_the_effective_provider_not_the_boot_constant(monkeypatch, temp_db):
    # Even if the boot-default LLM_PROVIDER is anthropic, a rag whose EFFECTIVE provider is
    # ollama (post-switch) must report is_local=True / no metered cost.
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    rag = FakeRAG(_three_clean_sources(), ["Answer [1]."])
    rag.provider = "ollama"
    controller = ChatController(rag=rag)
    result = _final(_results(controller, Session(), "q"))
    assert result.usage.is_local is True
    assert result.usage.cost_usd is None


def test_reviewer_follows_the_effective_provider_when_unpinned(monkeypatch, temp_db):
    # A flagged (low-confidence) answer triggers the reviewer call; it must resolve against
    # the rag's effective provider/model, not the config default, when REVIEWER_PROVIDER was
    # never explicitly pinned.
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    monkeypatch.setattr(config, "REVIEWER_PROVIDER_PINNED", False)
    captured: dict[str, object] = {}

    def fake_get_reviewer_client(provider=None, model=None):
        captured["provider"] = provider
        captured["model"] = model
        return object()

    monkeypatch.setattr(
        "doc_assistant.llm.get_reviewer_client", fake_get_reviewer_client, raising=False
    )
    monkeypatch.setattr(
        chat_controller, "review_answer", lambda prov, client: ReviewResult(error="stubbed")
    )
    # One weak, low-scoring source → fires a confidence signal → reviewer runs.
    weak_source = [(_doc("thin evidence", document_id="d1", chunk_index=0, filename="a.pdf"), 0.1)]
    rag = FakeRAG(weak_source, ["Answer [1]."])
    rag.provider = "ollama"
    rag.model = "llama3.1:8b"
    controller = ChatController(rag=rag)
    _final(_results(controller, Session(), "q"))
    assert captured == {"provider": "ollama", "model": "llama3.1:8b"}
