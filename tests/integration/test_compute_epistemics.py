"""Integration test for the epistemics build (Feature 7d) — the impure orchestration.

Drives ``epistemics.build_epistemics`` against a temp SQLite (for the
``chunk_epistemics`` sidecar), a temp concept-graph ``graph.json``, and a **stubbed**
``load_doc_chunks`` (so it never reads the real Chroma store). Asserts projection →
rows written, marker derivation, the unique-source-stays-quiet rule, idempotency
(replace not append), dry-run writes nothing, and a missing graph raises.

Deterministic + offline: no network, no real LLM, no Chroma.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import concept_graph as cg
from doc_assistant import epistemics
from doc_assistant.db.models import Base, ChunkEpistemics, Document
from doc_assistant.db.session import session_scope
from doc_assistant.epistemics import MARKER_SUPERSEDED, build_epistemics, load_epistemics_index


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Temp SQLite bound to the global session machinery; `load_doc_chunks` stubbed."""
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    monkeypatch.setattr(epistemics, "load_doc_chunks", lambda: list(_CHUNKS))
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


# colbert↔ranking: old supports (2005), new supersedes (2022) → superseded_trend.
# hyde↔prompting: a single supporting source → unique → never marked.
_GRAPH_EXTRACTIONS = [
    cg.DocExtraction(
        "old",
        "hold",
        "old.pdf",
        ["colbert", "ranking"],
        [cg.Triple("colbert", "uses", "ranking", "supports")],
        year=2005,
    ),
    cg.DocExtraction(
        "new",
        "hnew",
        "new.pdf",
        ["colbert", "ranking"],
        [cg.Triple("colbert", "uses", "ranking", "supersedes")],
        year=2022,
    ),
    cg.DocExtraction(
        "z",
        "hz",
        "z.pdf",
        ["hyde", "prompting"],
        [cg.Triple("hyde", "uses", "prompting", "supports")],
        year=2020,
    ),
]

_CHUNKS = [
    ("doc-colbert", 0, "this chunk explains colbert and ranking together"),
    ("doc-hyde", 0, "a hyde prompting trick that nobody else covers"),
    ("doc-empty", 0, "totally unrelated prose with none of the concepts"),
]


def _write_graph(root: Path) -> None:
    graph = cg.assemble_graph(_GRAPH_EXTRACTIONS)
    root.mkdir(parents=True, exist_ok=True)
    (root / cg.GRAPH_NAME).write_text(json.dumps(cg.graph_to_dict(graph)), encoding="utf-8")


def _seed_docs(doc_ids: list[str]) -> None:
    with session_scope() as session:
        for did in doc_ids:
            session.add(
                Document(
                    id=did,
                    filename=f"{did}.pdf",
                    source_original=f"{did}.pdf",
                    doc_hash=f"h-{did}",
                    format="pdf",
                )
            )


def _count_rows() -> int:
    with session_scope() as session:
        return int(session.execute(select(func.count()).select_from(ChunkEpistemics)).scalar_one())


def test_apply_writes_rows_and_marks_superseded(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_graph(env / "graph")

    result = build_epistemics(apply=True, graph_dir=env / "graph")
    assert result.applied
    assert result.n_superseded_nodes >= 1

    with session_scope() as session:
        rows = list(session.execute(select(ChunkEpistemics)).scalars())
    by_key = {f"{r.document_id}:{r.chunk_index}": r for r in rows}
    # Both chunks that carry a claim get a row; the concept-free chunk does not.
    assert set(by_key) == {"doc-colbert:0", "doc-hyde:0"}
    assert by_key["doc-colbert:0"].n_superseded_trend >= 1
    assert by_key["doc-hyde:0"].n_contested == 0  # the unique-source chunk stays clean


def test_marker_index_returns_only_marked_chunks(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_graph(env / "graph")
    build_epistemics(apply=True, graph_dir=env / "graph")

    index = load_epistemics_index()
    assert MARKER_SUPERSEDED in index.get("doc-colbert:0", [])
    assert "doc-hyde:0" not in index  # quiet-on-clean: unique-source chunk not surfaced


def test_rebuild_is_idempotent_replaces_not_appends(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_graph(env / "graph")
    build_epistemics(apply=True, graph_dir=env / "graph")
    build_epistemics(apply=True, graph_dir=env / "graph")
    assert _count_rows() == 2  # two claim-bearing chunks, not four


def test_dry_run_writes_nothing(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_graph(env / "graph")
    result = build_epistemics(apply=False, graph_dir=env / "graph")
    assert not result.applied
    assert result.rows  # computed in-memory...
    assert _count_rows() == 0  # ...but nothing persisted


def test_missing_graph_raises(env: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_epistemics(apply=True, graph_dir=env / "no-graph-here")


def test_no_document_mutation(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_graph(env / "graph")
    build_epistemics(apply=True, graph_dir=env / "graph")
    with session_scope() as session:
        doc_count = session.execute(select(func.count()).select_from(Document)).scalar_one()
    assert doc_count == 2  # the sidecar never touches the documents table
