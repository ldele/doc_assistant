"""Tests for the conversation-history read layer (feature-conversation-history.md).

Pure reads over ``AnswerRecord`` — grouping, ordering, title derivation, NULL exclusion, and
read-only rehydration with degraded citations. DB rows are seeded directly with explicit
``created_at`` so ordering assertions are deterministic (not dependent on insert-time clock
resolution).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from doc_assistant.conversations import (
    conversation_export_turns,
    get_conversation,
    list_conversations,
    set_conversation_meta,
)
from doc_assistant.db.models import AnswerRecord
from doc_assistant.db.session import session_scope


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the SQLAlchemy engine at a temp SQLite file and create schema."""
    db_file = tmp_path / "test.db"
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    test_engine = create_engine(f"sqlite:///{db_file}", future=True)
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, future=True
    )
    monkeypatch.setattr(session_mod, "_engine", test_engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", test_session_factory)
    yield db_file
    test_engine.dispose()


def _seed(
    session_id: str | None,
    query: str,
    *,
    answer: str = "an answer",
    when: datetime,
    original_query: str | None = None,
    chunks: list[dict] | None = None,
) -> None:
    with session_scope() as session:
        session.add(
            AnswerRecord(
                session_id=session_id,
                query=query,
                original_query=original_query,
                answer=answer,
                retrieved_chunks_json=json.dumps(chunks or []),
                created_at=when,
            )
        )


def test_list_conversations_groups_by_session(temp_db: Path):
    _seed("s1", "first q", when=datetime(2026, 7, 13, 10, 0, 0))
    _seed("s1", "second q", when=datetime(2026, 7, 13, 10, 5, 0))
    _seed("s2", "other q", when=datetime(2026, 7, 13, 11, 0, 0))

    convos = list_conversations()
    assert len(convos) == 2
    by_id = {c.session_id: c for c in convos}
    assert by_id["s1"].turn_count == 2
    assert by_id["s2"].turn_count == 1


def test_list_conversations_title_is_earliest_question(temp_db: Path):
    # Out-of-order inserts: the earliest *created_at*, not the insert order, sets the title.
    _seed("s1", "the second turn", when=datetime(2026, 7, 13, 10, 5, 0))
    _seed("s1", "the first turn", when=datetime(2026, 7, 13, 10, 0, 0))

    (convo,) = list_conversations()
    assert convo.title == "the first turn"


def test_list_conversations_title_prefers_original_query(temp_db: Path):
    _seed(
        "s1",
        "rewritten standalone query",
        original_query="what did the user actually type?",
        when=datetime(2026, 7, 13, 10, 0, 0),
    )
    (convo,) = list_conversations()
    assert convo.title == "what did the user actually type?"


def test_list_conversations_orders_newest_first(temp_db: Path):
    _seed("old", "q", when=datetime(2026, 7, 10, 9, 0, 0))
    _seed("new", "q", when=datetime(2026, 7, 13, 9, 0, 0))
    _seed("mid", "q", when=datetime(2026, 7, 12, 9, 0, 0))

    convos = list_conversations()
    assert [c.session_id for c in convos] == ["new", "mid", "old"]


def test_list_conversations_excludes_null_session_id(temp_db: Path):
    _seed(None, "pre-fix turn, no session", when=datetime(2026, 7, 1, 9, 0, 0))
    _seed("s1", "real turn", when=datetime(2026, 7, 13, 9, 0, 0))

    convos = list_conversations()
    assert [c.session_id for c in convos] == ["s1"]


def test_list_conversations_respects_limit(temp_db: Path):
    for i in range(5):
        _seed(f"s{i}", "q", when=datetime(2026, 7, 13, 10, i, 0))
    assert len(list_conversations(limit=3)) == 3


def test_list_conversations_empty(temp_db: Path):
    assert list_conversations() == []


def test_get_conversation_orders_turns_and_parses_sources(temp_db: Path):
    _seed(
        "s1",
        "q2",
        when=datetime(2026, 7, 13, 10, 5, 0),
        chunks=[
            {
                "filename": "paper.pdf",
                "page": 4,
                "section": "Methods",
                "chunk_excerpt": "excerpt A",
            },
            {"filename": "book.epub", "chunk_excerpt": "excerpt B"},
        ],
    )
    _seed("s1", "q1", when=datetime(2026, 7, 13, 10, 0, 0))

    detail = get_conversation("s1")
    assert detail is not None
    assert [t.question for t in detail.turns] == ["q1", "q2"]  # created_at order
    assert detail.title == "q1"

    sources = detail.turns[1].sources
    assert [s.n for s in sources] == [1, 2]
    assert sources[0].citation == '[1] paper.pdf \xb7 p.4 \xb7 "Methods"'
    assert sources[0].excerpt == "excerpt A"
    assert sources[1].citation == "[2] book.epub"  # no page/section


def test_get_conversation_unknown_returns_none(temp_db: Path):
    assert get_conversation("nope") is None


def test_get_conversation_ignores_null_session(temp_db: Path):
    _seed(None, "orphan", when=datetime(2026, 7, 1, 9, 0, 0))
    # A NULL session_id is not addressable as a conversation.
    assert get_conversation("") is None


# --- conversation_export_turns (durable export source) -----------------------


def test_conversation_export_turns_builds_from_records(temp_db: Path):
    _seed(
        "s1",
        "q1 rewritten",
        original_query="what is dpr?",
        when=datetime(2026, 7, 13, 10, 0, 0),
        answer="DPR uses dual encoders [1].",
        chunks=[
            {
                "filename": "dpr.pdf",
                "page": 3,
                "section": "Intro",
                "reranker_score": 0.71,
                "chunk_excerpt": "excerpt one",
            }
        ],
    )
    _seed("s1", "q2", when=datetime(2026, 7, 13, 10, 5, 0), answer="Second answer.")

    turns = conversation_export_turns("s1")
    assert [t.question for t in turns] == [
        "what is dpr?",
        "q2",
    ]  # created_at order; original_query wins
    assert turns[0].answer == "DPR uses dual encoders [1]."
    assert turns[0].standalone_query == "q1 rewritten"  # differed from original → recorded
    src = turns[0].sources[0]
    assert (src.n, src.filename, src.page, src.reranker_score) == (1, "dpr.pdf", 3, 0.71)
    assert turns[1].sources == []  # no chunks persisted


def test_conversation_export_turns_empty_for_unknown(temp_db: Path):
    assert conversation_export_turns("ghost") == []


# --- conversation management: pin / archive / soft-delete ---------------------


def test_pin_sorts_conversation_first(temp_db: Path):
    _seed("old", "q", when=datetime(2026, 7, 13, 10, 0, 0))
    _seed("new", "q", when=datetime(2026, 7, 13, 12, 0, 0))
    _seed("pinme", "q", when=datetime(2026, 7, 13, 9, 0, 0))  # the oldest
    set_conversation_meta("pinme", pinned=True)

    convos = list_conversations()
    assert convos[0].session_id == "pinme" and convos[0].pinned
    # non-pinned keep newest-first below the pin
    assert [c.session_id for c in convos[1:]] == ["new", "old"]


def test_soft_delete_excludes_and_restores(temp_db: Path):
    _seed("keep", "q", when=datetime(2026, 7, 13, 10, 0, 0))
    _seed("gone", "q", when=datetime(2026, 7, 13, 11, 0, 0))
    set_conversation_meta("gone", deleted=True)
    assert [c.session_id for c in list_conversations()] == ["keep"]  # hidden
    # the underlying records are retained — restore brings it back
    set_conversation_meta("gone", deleted=False)
    assert {c.session_id for c in list_conversations()} == {"keep", "gone"}


def test_archive_surfaces_as_flag_not_excluded(temp_db: Path):
    _seed("s1", "q", when=datetime(2026, 7, 13, 10, 0, 0))
    set_conversation_meta("s1", archived=True)
    (c,) = list_conversations()  # archived still listed; the frontend chooses to hide
    assert c.archived and not c.pinned


def test_set_meta_changes_only_given_fields(temp_db: Path):
    _seed("s1", "q", when=datetime(2026, 7, 13, 10, 0, 0))
    set_conversation_meta("s1", pinned=True)
    set_conversation_meta("s1", archived=True)  # pinned must survive
    (c,) = list_conversations()
    assert c.pinned and c.archived


def test_meta_absent_row_is_all_default(temp_db: Path):
    _seed("s1", "q", when=datetime(2026, 7, 13, 10, 0, 0))
    (c,) = list_conversations()
    assert not c.pinned and not c.archived


def test_rename_sets_and_reverts_title(temp_db: Path):
    _seed("s1", "the original question", when=datetime(2026, 7, 13, 10, 0, 0))
    set_conversation_meta("s1", title="My renamed chat")
    assert list_conversations()[0].title == "My renamed chat"
    d = get_conversation("s1")
    assert d is not None and d.title == "My renamed chat"
    # a blank title reverts to the derived first-question title
    set_conversation_meta("s1", title="   ")
    assert list_conversations()[0].title == "the original question"
