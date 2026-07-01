"""Integration test for concept_merge_suggestions wiring (fake embedder, temp DB)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.concept_semantics as cs
import doc_assistant.db.session as session_mod
from doc_assistant.concept_semantics import concept_merge_suggestions
from doc_assistant.concept_skeleton import add_concept
from doc_assistant.db.models import Base


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _fake_embed(texts: list[str]) -> list[list[float]]:
    # Deterministic toy vectors: the two "dense …" concepts collide; BM25 is orthogonal.
    out: list[list[float]] = []
    for t in texts:
        if t.lower().startswith("dense"):
            out.append([1.0, 0.0])
        elif t.lower().startswith("bm25"):
            out.append([0.0, 1.0])
        else:
            out.append([0.5, 0.5])
    return out


def test_merge_suggestions_flag_near_duplicates_not_distinct(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_concept("dense retrieval", aliases=["DPR"])
    add_concept("dense passage retrieval")
    add_concept("BM25", definition="A sparse lexical ranker.")
    monkeypatch.setattr(cs, "embed_texts", _fake_embed)

    pairs = concept_merge_suggestions(threshold=0.9)
    flagged = {frozenset((p.label_a, p.label_b)) for p in pairs}
    assert frozenset({"dense retrieval", "dense passage retrieval"}) in flagged  # near-dup caught
    assert all("BM25" not in (p.label_a, p.label_b) for p in pairs)  # distinct concept not flagged


def test_merge_suggestions_empty_for_single_concept(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_concept("BM25")
    monkeypatch.setattr(cs, "embed_texts", _fake_embed)
    assert concept_merge_suggestions(threshold=0.5) == []
