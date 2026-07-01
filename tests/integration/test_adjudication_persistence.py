"""Integration test for Chunk 2a adjudication persistence (answer_claims).

Exercises the full DB round-trip — record an answer, segment + eager-persist its
claims as ``pending``, then accept/reject/edit and read the log back — against a
fresh temp SQLite. No LLM/pipeline; the UI wiring that calls these lives in the
app shell (CLI / API / Tauri).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base
from doc_assistant.provenance import (
    RetrievedChunk,
    adjudicate_claim,
    get_claims,
    record_answer,
    record_claims,
)
from doc_assistant.synthesis import segment_claims


@pytest.fixture
def temp_db(tmp_path: Path) -> Iterator[Path]:
    """Bind the global session machinery to a fresh SQLite file."""
    db_path = tmp_path / "library.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine = session_mod._engine
    orig_factory = session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    try:
        yield db_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed_answer() -> tuple[str, list[str]]:
    """Record an answer + its segmented claims; return (answer_id, claim_ids)."""
    sources = [
        RetrievedChunk(filename="dpr.pdf", page=4, reranker_score=0.82),
        RetrievedChunk(filename="dpr.pdf", page=5, reranker_score=0.05),
    ]
    answer = "DPR beats BM25 [1]. The gain is small [2]. This part has no citation."
    answer_id = record_answer(
        query="how does DPR compare?", answer=answer, retrieved_chunks=sources
    )
    claim_ids = record_claims(answer_id, segment_claims(answer, sources))
    return answer_id, claim_ids


def test_claims_eager_persisted_as_pending(temp_db: Path) -> None:
    answer_id, claim_ids = _seed_answer()
    assert len(claim_ids) == 3
    claims = get_claims(answer_id)
    assert [c.claim_index for c in claims] == [0, 1, 2]
    assert all(c.decision == "pending" for c in claims)
    # Markers carried through: strong cite ok, weak cite weak, uncited unsupported.
    assert [c.marker for c in claims] == ["ok", "weak", "unsupported"]
    assert claims[0].citations[0]["filename"] == "dpr.pdf"


def test_accept_reject_edit_update_the_log(temp_db: Path) -> None:
    answer_id, claim_ids = _seed_answer()
    adjudicate_claim(claim_ids[0], "accepted")
    adjudicate_claim(claim_ids[1], "rejected")
    adjudicate_claim(claim_ids[2], "edited", edited_text="Corrected, grounded claim.")

    claims = {c.claim_index: c for c in get_claims(answer_id)}
    assert claims[0].decision == "accepted" and claims[0].edited_text is None
    assert claims[1].decision == "rejected"
    assert claims[2].decision == "edited" and claims[2].edited_text == "Corrected, grounded claim."


def test_edited_text_dropped_for_non_edit_decisions(temp_db: Path) -> None:
    answer_id, claim_ids = _seed_answer()
    adjudicate_claim(claim_ids[0], "accepted", edited_text="should be ignored")
    claim0 = next(c for c in get_claims(answer_id) if c.id == claim_ids[0])
    assert claim0.decision == "accepted"
    assert claim0.edited_text is None


def test_invalid_decision_raises(temp_db: Path) -> None:
    _, claim_ids = _seed_answer()
    with pytest.raises(ValueError):
        adjudicate_claim(claim_ids[0], "maybe")


def test_unknown_claim_id_raises(temp_db: Path) -> None:
    _seed_answer()
    with pytest.raises(KeyError):
        adjudicate_claim("does-not-exist", "accepted")
