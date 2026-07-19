"""Integration guard tests for the Node-A skeleton build.

Seeds a curated vocabulary + Citation/DocSimilarity into a temp file-backed SQLite, injects
a fake presence loader (no Chroma), and exercises the real DB read/write paths. Asserts:
the deterministic build makes zero LLM calls, writes the derived sidecar + skeleton.json,
is byte-identical on a re-run, and keeps the two lifecycles distinct (curated rows survive a
--force rebuild; derived rows are dropped + rebuilt). The chunk store is never touched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import (
    Base,
    Citation,
    Concept,
    ConceptAlias,
    ConceptEdge,
    ConceptPresenceRow,
    DocSimilarity,
    Document,
)
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.concept_skeleton import build_concept_skeleton


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


def _seed() -> dict[str, str]:
    """Two docs, two curated concepts (RAG + DPR), a citation + a similarity edge d1->d2."""
    ids: dict[str, str] = {}
    with session_scope() as session:
        for name in ("d1", "d2"):
            doc = Document(
                filename=f"{name}.pdf",
                source_original=f"{name}.pdf",
                doc_hash=f"h{name}",
                format="pdf",
            )
            session.add(doc)
            session.flush()
            ids[name] = str(doc.id)

        # graph_include is opt-in (ADR-018) and independent of `source` — a promoted
        # keyword is graph vocabulary here precisely because the flag, not the source,
        # decides membership.
        rag = Concept(label="RAG", source="keyword", graph_include=True)
        dpr = Concept(label="DPR", source="manual", graph_include=True)
        session.add_all([rag, dpr])
        session.flush()
        session.add(ConceptAlias(concept_id=rag.id, alias="retrieval-augmented generation"))

        session.add(Citation(source_document_id=ids["d1"], target_document_id=ids["d2"]))
        session.add(
            DocSimilarity(
                source_document_id=ids["d1"],
                target_document_id=ids["d2"],
                embedding_model="bge-base",
                score=0.71,
            )
        )
    return ids


def _fake_presence(ids: dict[str, str]):
    """RAG+DPR co-occur in d1:p0; RAG alone in d2:p0 → RAG spans d1,d2 and DPR sits in d1."""

    def loader(document_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
        return [
            (f"{ids['d1']}:p0", ids["d1"], "RAG and DPR are evaluated together here."),
            (f"{ids['d2']}:p0", ids["d2"], "RAG appears again in a second paper."),
        ]

    return loader


def _count(model: type) -> int:
    with session_scope() as session:
        return int(session.execute(select(func.count()).select_from(model)).scalar() or 0)


def test_node_a_build_writes_sidecar_and_provenance(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    result = build_concept_skeleton(
        apply=True,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )

    assert result.applied is True
    assert result.n_concepts == 2  # both curated concepts are nodes
    assert result.n_edges == 1  # the single RAG-DPR co-occurrence edge
    assert (skeleton_dir / "skeleton.json").exists()
    assert _count(ConceptEdge) == 1
    assert _count(ConceptPresenceRow) == 3  # RAG@d1, RAG@d2, DPR@d1

    # The edge carries citation + similarity provenance (d1->d2 spans RAG[d1,d2] / DPR[d1]).
    with session_scope() as session:
        edge = session.execute(select(ConceptEdge)).scalar_one()
        provenance = set(json.loads(edge.provenance_json))
        strength = json.loads(edge.strength_json)
    assert provenance == {"cooccurrence", "citation", "similarity"}
    # R4: graded strength persisted per doc-pair token; this saturated toy graph → 1.0.
    # Co-occurrence is the base fact and carries no strength entry.
    assert strength == {"citation": 1.0, "similarity": 1.0}


def test_build_is_byte_identical_on_rerun(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    kwargs = dict(apply=True, min_cooccurrence=1, presence_loader=_fake_presence(ids))

    build_concept_skeleton(skeleton_dir=skeleton_dir, **kwargs)
    first = (skeleton_dir / "skeleton.json").read_text(encoding="utf-8")
    build_concept_skeleton(skeleton_dir=skeleton_dir, **kwargs)
    second = (skeleton_dir / "skeleton.json").read_text(encoding="utf-8")

    assert first == second  # timestamp-free graph_version → byte-identical rebuild
    assert _count(ConceptEdge) == 1  # replace-not-append (no row duplication)
    assert _count(ConceptPresenceRow) == 3


def test_force_rebuild_keeps_curated_drops_derived(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    build_concept_skeleton(
        apply=True,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )
    curated_concepts = _count(Concept)
    curated_aliases = _count(ConceptAlias)

    build_concept_skeleton(
        apply=True,
        force=True,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )

    # Curated vocabulary survives a --force rebuild; derived rows are regenerated.
    assert _count(Concept) == curated_concepts == 2
    assert _count(ConceptAlias) == curated_aliases == 1
    assert _count(ConceptEdge) == 1
    assert _count(ConceptPresenceRow) == 3


def test_dry_run_writes_nothing(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    result = build_concept_skeleton(
        apply=False,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )
    assert result.applied is False
    assert result.n_edges == 1  # computed...
    assert not (skeleton_dir / "skeleton.json").exists()  # ...but nothing written
    assert _count(ConceptEdge) == 0
    assert _count(ConceptPresenceRow) == 0


def test_build_never_touches_chunk_store(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    build_concept_skeleton(
        apply=True,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )
    # Sidecar-only: no Chroma dir created, Document rows unchanged.
    assert not (env / "chroma").exists()
    assert not (env / "chroma_pc").exists()
    assert _count(Document) == 2
