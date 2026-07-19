"""Integration test for the epistemics build (Feature 7d) — the impure orchestration.

Drives ``epistemics.build_epistemics`` against a temp SQLite (for the
``chunk_epistemics`` sidecar), a temp concept-skeleton ``skeleton.json``, and a
**stubbed** ``load_doc_chunks`` (so it never reads the real Chroma store). Asserts
projection → rows written, marker derivation, the unique-source-stays-quiet rule,
idempotency (replace not append), dry-run writes nothing, and a missing skeleton raises.

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
from doc_assistant.db.models import Base, ChunkEpistemics, Document
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge import concept_skeleton as cs
from doc_assistant.knowledge import epistemics
from doc_assistant.knowledge.epistemics import (
    MARKER_CONTESTED,
    MARKER_SUPERSEDED,
    build_epistemics,
    load_epistemics_index,
)


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


# colbert<->ranking: "old" supports, "new" contradicts → contested (the skeleton
# carries no publication years, so superseded_trend isn't reachable — see
# concept_skeleton.node_weights_for_epistemics).
# hyde<->prompting: a single supporting source → unique → never marked.
_COOC = frozenset({"cooccurrence"})
_SKELETON_NODES = [
    cs.ConceptNode("colbert", "colbert", ("old", "new"), 0, -1),
    cs.ConceptNode("ranking", "ranking", ("old", "new"), 0, -1),
    cs.ConceptNode("hyde", "hyde", ("z",), 0, -1),
    cs.ConceptNode("prompting", "prompting", ("z",), 0, -1),
]
_SKELETON_EDGES = [
    cs.SkeletonEdge(
        "colbert",
        "ranking",
        _COOC,
        cs.edge_weight(_COOC, 2),
        2,
        stance_by_doc=(("new", "contradicts"), ("old", "supports")),
    ),
    cs.SkeletonEdge(
        "hyde", "prompting", _COOC, cs.edge_weight(_COOC, 1), 1, stance_by_doc=(("z", "supports"),)
    ),
]

_CHUNKS = [
    ("doc-colbert", 0, "this chunk explains colbert and ranking together"),
    ("doc-hyde", 0, "a hyde prompting trick that nobody else covers"),
    ("doc-empty", 0, "totally unrelated prose with none of the concepts"),
]


def _write_skeleton(root: Path) -> None:
    skeleton = cs.analyze_skeleton(_SKELETON_NODES, _SKELETON_EDGES, seed=42)
    root.mkdir(parents=True, exist_ok=True)
    (root / cs.SKELETON_NAME).write_text(
        json.dumps(cs.skeleton_to_dict(skeleton)), encoding="utf-8"
    )


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


def test_apply_writes_rows_and_marks_contested(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_skeleton(env / "skeleton")

    result = build_epistemics(apply=True, skeleton_dir=env / "skeleton")
    assert result.applied
    assert result.n_contested_nodes >= 1

    with session_scope() as session:
        rows = list(session.execute(select(ChunkEpistemics)).scalars())
    by_key = {f"{r.document_id}:{r.chunk_index}": r for r in rows}
    # Both chunks that carry a claim get a row; the concept-free chunk does not.
    assert set(by_key) == {"doc-colbert:0", "doc-hyde:0"}
    assert by_key["doc-colbert:0"].n_contested >= 1
    assert by_key["doc-hyde:0"].n_contested == 0  # the unique-source chunk stays clean


def test_superseded_row_and_marker(env: Path) -> None:
    """G3 + G6: a contested node whose contradicting docs are, in aggregate, newer than its
    supporting docs surfaces the superseded_trend marker end-to-end (skeleton.json meta ->
    build_epistemics -> chunk_epistemics row -> marker index). Two dated docs per side (the
    G6 confidence floor) so the fire isn't the demoted single-doc case — see
    ``test_superseded_marker_requires_two_per_side`` below for that. Own skeleton fixture (not
    the module-level ``_SKELETON_*`` one) so it doesn't perturb the plain-contested tests above.
    """
    _seed_docs(["doc-colbert", "doc-hyde"])
    nodes = [
        cs.ConceptNode("colbert", "colbert", ("old1", "old2", "new1", "new2"), 0, -1),
        cs.ConceptNode("ranking", "ranking", ("old1", "old2", "new1", "new2"), 0, -1),
    ]
    edges = [
        cs.SkeletonEdge(
            "colbert",
            "ranking",
            _COOC,
            cs.edge_weight(_COOC, 2),
            2,
            stance_by_doc=(
                ("old1", "supports"),
                ("old2", "supports"),
                ("new1", "contradicts"),
                ("new2", "contradicts"),
            ),
        ),
    ]
    doc_years = {"old1": 2017, "old2": 2018, "new1": 2023, "new2": 2024}
    skeleton = cs.analyze_skeleton(nodes, edges, seed=42, meta_extra={"doc_years": doc_years})
    root = env / "skeleton"
    root.mkdir(parents=True, exist_ok=True)
    (root / cs.SKELETON_NAME).write_text(
        json.dumps(cs.skeleton_to_dict(skeleton)), encoding="utf-8"
    )

    result = build_epistemics(apply=True, skeleton_dir=root)
    assert result.n_superseded_nodes == 2  # both colbert and ranking are superseded_trend

    index = load_epistemics_index()
    assert MARKER_SUPERSEDED in index.get("doc-colbert:0", [])


def test_superseded_marker_requires_two_per_side(env: Path) -> None:
    """G6 (SPRINT-006): the exact 1-v-1 fixture that fired superseded_trend end-to-end under
    G3 alone now stays contested — demoted, not deleted, coverage. `MARKER_SUPERSEDED` must
    NOT appear; `MARKER_CONTESTED` still does (the underlying dispute is real, just thin)."""
    _seed_docs(["doc-colbert", "doc-hyde"])
    nodes = [
        cs.ConceptNode("colbert", "colbert", ("old", "new"), 0, -1),
        cs.ConceptNode("ranking", "ranking", ("old", "new"), 0, -1),
    ]
    edges = [
        cs.SkeletonEdge(
            "colbert",
            "ranking",
            _COOC,
            cs.edge_weight(_COOC, 2),
            2,
            stance_by_doc=(("new", "contradicts"), ("old", "supports")),
        ),
    ]
    skeleton = cs.analyze_skeleton(
        nodes, edges, seed=42, meta_extra={"doc_years": {"old": 2018, "new": 2024}}
    )
    root = env / "skeleton"
    root.mkdir(parents=True, exist_ok=True)
    (root / cs.SKELETON_NAME).write_text(
        json.dumps(cs.skeleton_to_dict(skeleton)), encoding="utf-8"
    )

    result = build_epistemics(apply=True, skeleton_dir=root)
    assert result.n_superseded_nodes == 0
    assert result.n_contested_nodes == 2

    index = load_epistemics_index()
    assert MARKER_CONTESTED in index.get("doc-colbert:0", [])
    assert MARKER_SUPERSEDED not in index.get("doc-colbert:0", [])


def test_uuid_id_node_is_attributed_via_label_end_to_end(env: Path) -> None:
    """KI-15, end-to-end: a curated-skeleton node id is an opaque UUID (never present in any
    chunk text) — attribution must go through the node's LABEL, not its id, or this node's
    claim never reaches a chunk and the marker never surfaces, no matter how correct its
    coverage/direction weights are."""
    _seed_docs(["doc-colbert", "doc-hyde"])
    uuid_a = "00688507-0351-442b-b156-00521129a344"
    uuid_b = "007cc904-194d-408d-b847-13f9c79b1a2d"
    nodes = [
        cs.ConceptNode(uuid_a, "colbert", ("old", "new"), 0, -1),
        cs.ConceptNode(uuid_b, "ranking", ("old", "new"), 0, -1),
    ]
    edges = [
        cs.SkeletonEdge(
            uuid_a,
            uuid_b,
            _COOC,
            cs.edge_weight(_COOC, 2),
            2,
            stance_by_doc=(("new", "contradicts"), ("old", "supports")),
        ),
    ]
    skeleton = cs.analyze_skeleton(nodes, edges, seed=42)
    root = env / "skeleton"
    root.mkdir(parents=True, exist_ok=True)
    (root / cs.SKELETON_NAME).write_text(
        json.dumps(cs.skeleton_to_dict(skeleton)), encoding="utf-8"
    )

    result = build_epistemics(apply=True, skeleton_dir=root)
    assert result.n_contested_nodes == 2
    # The whole point: rows/markers exist even though no chunk text ever contains a UUID.
    assert len(result.rows) >= 1

    index = load_epistemics_index()
    assert MARKER_CONTESTED in index.get("doc-colbert:0", [])


def test_marker_index_returns_only_marked_chunks(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_skeleton(env / "skeleton")
    build_epistemics(apply=True, skeleton_dir=env / "skeleton")

    index = load_epistemics_index()
    assert MARKER_CONTESTED in index.get("doc-colbert:0", [])
    assert "doc-hyde:0" not in index  # quiet-on-clean: unique-source chunk not surfaced


def test_rebuild_is_idempotent_replaces_not_appends(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_skeleton(env / "skeleton")
    build_epistemics(apply=True, skeleton_dir=env / "skeleton")
    build_epistemics(apply=True, skeleton_dir=env / "skeleton")
    assert _count_rows() == 2  # two claim-bearing chunks, not four


def test_dry_run_writes_nothing(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_skeleton(env / "skeleton")
    result = build_epistemics(apply=False, skeleton_dir=env / "skeleton")
    assert not result.applied
    assert result.rows  # computed in-memory...
    assert _count_rows() == 0  # ...but nothing persisted


def test_missing_skeleton_raises(env: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_epistemics(apply=True, skeleton_dir=env / "no-skeleton-here")


def test_no_document_mutation(env: Path) -> None:
    _seed_docs(["doc-colbert", "doc-hyde"])
    _write_skeleton(env / "skeleton")
    build_epistemics(apply=True, skeleton_dir=env / "skeleton")
    with session_scope() as session:
        doc_count = session.execute(select(func.count()).select_from(Document)).scalar_one()
    assert doc_count == 2  # the sidecar never touches the documents table
