"""Integration test for concept_merge_suggestions wiring (fake embedder, temp DB)."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
import doc_assistant.knowledge.concept_semantics as cs
from doc_assistant import config
from doc_assistant.db.models import Base
from doc_assistant.knowledge.concept_semantics import concept_merge_suggestions
from doc_assistant.knowledge.concept_skeleton import add_concept


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


def _fake_embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
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


def test_the_merge_preview_is_the_merge(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ROADMAP 53 (2). The preview (``suggest_concepts --near``) and the merge
    (``curate_concepts --dedup``) differed in threshold (0.85 vs a hard-coded 0.9), embedder
    (specter2 vs bge-base) and text (label + definition vs label), so what the preview showed was
    not what the merge did. Both now go through ``dedup_pairs``: same texts reach the embedder,
    same model, same pairs — and an extraction artifact is a candidate in neither."""
    import scripts.curate_concepts as curate_cli
    import scripts.suggest_concepts as suggest_cli

    from doc_assistant.knowledge.concept_curation import dedup_pairs, is_artifact, load_concepts

    add_concept("dense retrieval", definition="Retrieval over learned dense vectors.")
    add_concept("dense passage retrieval")
    add_concept("dense 2015")  # an artifact label (pure-digit token)
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def recording_embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
        calls.append((tuple(texts), model))
        return _fake_embed(texts, model=model)

    monkeypatch.setattr(cs, "embed_texts", recording_embed)

    preview = concept_merge_suggestions(threshold=0.9, model="bge-base")
    survivors = [(cid, label) for cid, label in load_concepts() if not is_artifact(label)]
    label_by_id = dict(survivors)
    merge = [
        (label_by_id[a], label_by_id[b], c)
        for a, b, c in dedup_pairs(survivors, threshold=0.9, model="bge-base")
    ]

    assert [(p.label_a, p.label_b, p.cosine) for p in preview] == merge
    assert calls[0] == calls[1]  # identical texts and model reach the embedder
    assert "dense retrieval. Retrieval over learned dense vectors." in calls[0][0]
    assert not any("2015" in text for text in calls[0][0])

    # One pair of knobs: the library default and both CLIs name the merge's own model and
    # threshold, never the candidate-extraction embedder (CONCEPT_EMBED_MODEL, SPECTER2 — which
    # scores a median label pair 0.842 and would merge nearly everything; baseline 2026-09-17).
    signature = inspect.signature(dedup_pairs).parameters
    assert signature["threshold"].default == config.CONCEPT_MERGE_COSINE
    assert signature["model"].default == config.CONCEPT_MERGE_MODEL
    for cli in (curate_cli, suggest_cli):
        source = Path(cli.__file__).read_text(encoding="utf-8")
        assert "CONCEPT_MERGE_COSINE" in source and "CONCEPT_MERGE_MODEL" in source, cli.__name__


def test_anchor_ranked_downranks_off_topic_boilerplate(monkeypatch: pytest.MonkeyPatch) -> None:
    markdown = (
        "## **Abstract**\n\ndense retrieval beats bm25 on passage retrieval.\n\n"
        "## **1 Introduction**\n\nwe thank the funding committee and acknowledgements.\n"
    )
    monkeypatch.setattr(
        cs, "_load_paper_docs", lambda ids=None: [("d1", "paper.pdf", "Dense Retrieval", markdown)]
    )

    def fake_embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
        on = ("dense", "retrieval", "bm25", "passage")
        return [[1.0, 0.0] if any(w in t.lower() for w in on) else [0.0, 1.0] for t in texts]

    monkeypatch.setattr(cs, "embed_texts", fake_embed)
    results = cs.anchor_ranked_candidates(top_k=3, pool_k=30)
    assert len(results) == 1
    scored = results[0][2]
    assert scored  # non-empty
    # On-topic term ranks first; scores are sorted descending; boilerplate is pushed down.
    assert any(w in scored[0].term for w in ("dense", "retrieval", "bm25", "passage"))
    assert [s.anchor_cosine for s in scored] == sorted(
        (s.anchor_cosine for s in scored), reverse=True
    )
