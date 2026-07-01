"""Tests for the provenance card module (PR 5 / Integrity Chunk 1).

Most of these exercise the pure-function helpers without a real DB.
The persistence path is covered by a separate roundtrip test that uses
a temp SQLite file via the existing migrations machinery.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_assistant.provenance import (
    AnswerProvenance,
    RetrievedChunk,
    find_record_by_short_id,
    get_record,
    list_recent_records,
    prompt_version_hash,
    record_answer,
    template_hash,
)

# ============================================================
# Pure helpers — no DB
# ============================================================


def test_template_hash_is_stable():
    assert template_hash("hello") == template_hash("hello")
    assert template_hash("hello") != template_hash("hello ")


def test_template_hash_length():
    assert len(template_hash("anything")) == 12


def test_prompt_version_hash_stable_under_same_config():
    h1 = prompt_version_hash(
        template_hash="abc", top_k=10, use_parent_child=True, embedding_model="bge-base"
    )
    h2 = prompt_version_hash(
        template_hash="abc", top_k=10, use_parent_child=True, embedding_model="bge-base"
    )
    assert h1 == h2


def test_prompt_version_hash_changes_with_each_field():
    base = dict(template_hash="abc", top_k=10, use_parent_child=True, embedding_model="bge-base")
    base_hash = prompt_version_hash(**base)
    assert prompt_version_hash(**{**base, "template_hash": "def"}) != base_hash
    assert prompt_version_hash(**{**base, "top_k": 5}) != base_hash
    assert prompt_version_hash(**{**base, "use_parent_child": False}) != base_hash
    assert prompt_version_hash(**{**base, "embedding_model": "specter2"}) != base_hash


def test_prompt_version_hash_length():
    h = prompt_version_hash(
        template_hash="x", top_k=1, use_parent_child=False, embedding_model="y"
    )
    assert len(h) == 12


def test_answer_provenance_to_json_dict_serialises_datetime():
    from datetime import datetime, timezone

    prov = AnswerProvenance(
        id="abc",
        query="q",
        answer="a",
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    d = prov.to_json_dict()
    assert isinstance(d["created_at"], str)
    # Round-trip through json — no datetime values left
    assert json.dumps(d) is not None


def test_answer_provenance_to_json_dict_handles_no_datetime():
    prov = AnswerProvenance(id="abc", query="q", answer="a")
    d = prov.to_json_dict()
    assert d["created_at"] is None
    assert json.dumps(d) is not None


# ============================================================
# Persistence roundtrip (uses a temp SQLite via SQLITE_PATH override)
# ============================================================


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the SQLAlchemy engine at a temp SQLite file and create schema."""
    db_file = tmp_path / "test.db"
    db_url = f"sqlite:///{db_file}"

    # Build a fresh engine bound to the temp DB and patch session_scope to use it.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    test_engine = create_engine(db_url, future=True)
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, future=True
    )

    monkeypatch.setattr(session_mod, "_engine", test_engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", test_session_factory)

    yield db_file

    test_engine.dispose()


def _sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            filename="paper.pdf",
            doc_id="abc123",
            page=4,
            section="Methods",
            reranker_score=0.87,
            chunk_excerpt="The voltage-clamp technique...",
        ),
        RetrievedChunk(
            filename="paper.pdf",
            doc_id="abc123",
            page=5,
            section="Results",
            reranker_score=0.62,
            chunk_excerpt="Sodium currents were isolated...",
        ),
    ]


def test_record_answer_roundtrip(temp_db: Path):
    record_id = record_answer(
        query="What did HH measure?",
        answer="They measured ionic currents.",
        retrieved_chunks=_sample_chunks(),
        model_name="haiku",
        embedding_model="bge-base",
        prompt_version="v1",
        top_k=10,
        use_parent_child=True,
        token_input=100,
        token_output=50,
        latency_ms=1234.5,
    )
    assert record_id  # UUID was generated

    prov = get_record(record_id)
    assert prov is not None
    assert prov.query == "What did HH measure?"
    assert prov.answer == "They measured ionic currents."
    assert prov.model_name == "haiku"
    assert prov.embedding_model == "bge-base"
    assert prov.token_input == 100
    assert prov.token_output == 50
    assert prov.latency_ms == 1234.5
    assert len(prov.retrieved_chunks) == 2
    assert prov.retrieved_chunks[0].filename == "paper.pdf"
    assert prov.retrieved_chunks[0].reranker_score == 0.87


def test_record_answer_excludes_full_text_from_persistence(temp_db: Path):
    """The wide reviewer-only ``full_text`` is never written to the DB (display
    excerpt is); persisting it would bloat every record with parent-sized text."""
    from sqlalchemy import select

    from doc_assistant.db.models import AnswerRecord
    from doc_assistant.db.session import session_scope

    rid = record_answer(
        query="q",
        answer="a",
        retrieved_chunks=[
            RetrievedChunk(
                filename="p.pdf", chunk_excerpt="short display", full_text="WIDE GROUNDING TEXT"
            )
        ],
    )
    with session_scope() as session:
        raw = session.execute(
            select(AnswerRecord.retrieved_chunks_json).where(AnswerRecord.id == rid)
        ).scalar_one()
    assert "short display" in raw  # the display excerpt is persisted
    assert "WIDE GROUNDING TEXT" not in raw and "full_text" not in raw  # the wide text is not


def test_record_answer_excludes_chunk_key_from_persistence(temp_db: Path):
    """The transient 7d join key ``chunk_key`` (ADR-2) is never written to the DB —
    like ``full_text``, it is a join key, not stored provenance."""
    from sqlalchemy import select

    from doc_assistant.db.models import AnswerRecord
    from doc_assistant.db.session import session_scope

    rid = record_answer(
        query="q",
        answer="a",
        retrieved_chunks=[
            RetrievedChunk(filename="p.pdf", chunk_excerpt="shown", chunk_key="doc1:3")
        ],
    )
    with session_scope() as session:
        raw = session.execute(
            select(AnswerRecord.retrieved_chunks_json).where(AnswerRecord.id == rid)
        ).scalar_one()
    assert "shown" in raw  # display excerpt persists
    assert "chunk_key" not in raw and "doc1:3" not in raw  # the join key does not

    # And a loaded record round-trips with chunk_key defaulting to None (not in JSON).
    prov = get_record(rid)
    assert prov is not None and prov.retrieved_chunks[0].chunk_key is None


def test_get_record_returns_none_for_missing(temp_db: Path):
    assert get_record("does-not-exist") is None


def test_find_record_by_short_id_unique_match(temp_db: Path):
    rid = record_answer(query="q", answer="a", retrieved_chunks=[])
    short = rid[:8]
    prov = find_record_by_short_id(short)
    assert prov is not None
    assert prov.id == rid


def test_find_record_by_short_id_no_match(temp_db: Path):
    record_answer(query="q", answer="a", retrieved_chunks=[])
    assert find_record_by_short_id("nope0000") is None


def test_list_recent_records_orders_desc(temp_db: Path):
    import time

    ids = []
    for i in range(3):
        ids.append(record_answer(query=f"q{i}", answer=f"a{i}", retrieved_chunks=[]))
        time.sleep(0.01)  # ensure distinct timestamps

    records = list_recent_records()
    assert len(records) == 3
    # Most recent first
    assert records[0].id == ids[-1]
    assert records[-1].id == ids[0]


def test_list_recent_records_respects_limit(temp_db: Path):
    for i in range(5):
        record_answer(query=f"q{i}", answer=f"a{i}", retrieved_chunks=[])
    records = list_recent_records(limit=3)
    assert len(records) == 3
