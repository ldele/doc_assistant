"""Integration tests for the concept-graph read model (PR-G1 — ADR-017, feature-concept-graph.md).

Covers the loader (round-trip, absent, corrupt), the gap sidecar read, the staleness verdict, the
per-concept presence read, and the four routes (graph 200/404, presence, rebuild 202/409 + status).

Temp file-backed SQLite + a temp skeleton dir; no LLM, no network, no real build. The rebuild route
is driven through the ``rebuild_graph_fn`` seam (cpc §13) so no real 7s Node-A run is triggered.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.concept_graph_view import load_concept_presence, load_graph_view
from doc_assistant.concept_skeleton import (
    Community,
    ConceptNode,
    ConceptSkeleton,
    SkeletonEdge,
    load_skeleton,
    skeleton_to_dict,
)
from doc_assistant.db.models import Base, Concept, ConceptPresenceRow, Document, GapRow
from doc_assistant.db.session import session_scope

_A = "11111111-1111-4111-8111-111111111111"
_B = "22222222-2222-4222-8222-222222222222"


class _FakeController:
    def chunk_count(self) -> int:
        return 0


def _skeleton() -> ConceptSkeleton:
    """A 2-node / 1-edge toy skeleton — the smallest thing with a real edge + community."""
    return ConceptSkeleton(
        nodes=(
            ConceptNode(id=_A, label="Embeddings", doc_ids=("d1", "d2"), degree=1, community=0),
            ConceptNode(id=_B, label="BM25", doc_ids=("d1",), degree=1, community=0),
        ),
        edges=(
            SkeletonEdge(
                source_concept_id=_A,
                target_concept_id=_B,
                provenance=frozenset({"cooccurrence", "similarity"}),
                weight=2.5,
                n_cooccurrence_chunks=4,
            ),
        ),
        communities=(Community(id=0, label="Embeddings", node_ids=(_A, _B), size=2),),
        meta={"graph_version": "cafef00d", "n_concepts": 2, "n_edges": 1},
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Temp DB + a temp skeleton dir, so nothing touches the real corpus."""
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    skel_dir = tmp_path / "skeleton"
    skel_dir.mkdir()
    monkeypatch.setattr("doc_assistant.config.CONCEPT_SKELETON_DIR", skel_dir)
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _write_skeleton_json(env: Path, skeleton: ConceptSkeleton) -> None:
    (env / "skeleton" / "skeleton.json").write_text(
        json.dumps(skeleton_to_dict(skeleton)), encoding="utf-8"
    )


def _seed_concepts(*ids_labels: tuple[str, str]) -> None:
    with session_scope() as s:
        s.add_all(
            Concept(id=i, label=lab, source="manual", graph_include=True) for i, lab in ids_labels
        )


def _seed_presence(concept_id: str, document_id: str, chunk_keys: list[str], n: int) -> None:
    """``concept_presence`` FKs both ``concepts.id`` and ``documents.id``, so the referents must
    exist first — the schema enforcing that is correct, not an obstacle to route around.

    The referent is committed in its **own** session: these models carry no ``relationship()``, so
    a single flush does not reliably order the parent insert before the child and the FK trips.
    """
    with session_scope() as s:
        s.add(
            Document(
                id=document_id,
                filename=f"{document_id}.pdf",
                source_original=f"/tmp/{document_id}.pdf",
                doc_hash=document_id,
                format="pdf",
            )
        )
    with session_scope() as s:
        s.add(
            ConceptPresenceRow(
                concept_id=concept_id,
                document_id=document_id,
                chunk_keys_json=json.dumps(chunk_keys),
                n_mentions=n,
                graph_version="cafef00d",
            )
        )


# ---------- load_skeleton ----------


def test_load_skeleton_round_trips(env: Path) -> None:
    """The loader is the exact inverse of the writer — the artifact survives a round-trip."""
    _write_skeleton_json(env, _skeleton())
    loaded = load_skeleton()
    assert loaded is not None
    assert skeleton_to_dict(loaded) == skeleton_to_dict(_skeleton())


def test_load_skeleton_absent_is_none_not_an_error(env: Path) -> None:
    """A fresh clone has no skeleton.json (it is gitignored + regenerable). That is the NORMAL
    first run, so the loader returns None rather than raising."""
    assert load_skeleton() is None


def test_load_skeleton_corrupt_raises(env: Path) -> None:
    """A present-but-unparseable artifact is NOT 'never built' — silently returning None would
    invite a rebuild that masks a real problem."""
    (env / "skeleton" / "skeleton.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        load_skeleton()


# ---------- the view + staleness ----------


def test_graph_view_none_when_never_built(env: Path) -> None:
    assert load_graph_view() is None


def test_graph_view_assembles_skeleton_gaps_and_staleness(env: Path) -> None:
    _write_skeleton_json(env, _skeleton())
    _seed_concepts((_A, "Embeddings"), (_B, "BM25"))
    with session_scope() as s:
        s.add(
            GapRow(
                concept_id=_B,
                tier="t1",
                determinism="deterministic",
                kind="single_source",
                evidence_json=json.dumps(["d1"]),
                status="surfaced",
                graph_version="cafef00d",
            )
        )
    view = load_graph_view()
    assert view is not None
    assert len(view.skeleton.nodes) == 2
    assert len(view.gaps) == 1
    assert view.gaps[0].kind == "single_source"
    assert view.gaps[0].evidence.fact_ids == ("d1",)
    assert view.staleness.stale is False


def test_graph_view_empty_gap_sidecar_is_legitimate(env: Path) -> None:
    """A present skeleton with no gaps means build_gaps hasn't run (or found nothing) — not a
    broken graph."""
    _write_skeleton_json(env, _skeleton())
    _seed_concepts((_A, "Embeddings"), (_B, "BM25"))
    view = load_graph_view()
    assert view is not None
    assert view.gaps == ()


def test_staleness_fires_when_a_concept_was_curated_since_the_build(env: Path) -> None:
    """The load-bearing staleness case: the Manage-keywords view writes Concept rows live, so a
    new family makes the graph lag immediately."""
    _write_skeleton_json(env, _skeleton())
    _seed_concepts(
        (_A, "Embeddings"), (_B, "BM25"), ("33333333-3333-4333-8333-333333333333", "RAG")
    )
    view = load_graph_view()
    assert view is not None
    assert view.staleness.stale is True
    assert view.staleness.n_concepts_in_db == 3
    assert view.staleness.n_concepts_in_skeleton == 2
    assert view.staleness.added_labels == ("RAG",)
    assert view.staleness.removed_ids == ()


def test_staleness_fires_when_a_concept_was_deleted_since_the_build(env: Path) -> None:
    _write_skeleton_json(env, _skeleton())
    _seed_concepts((_A, "Embeddings"))  # _B deleted since
    view = load_graph_view()
    assert view is not None
    assert view.staleness.stale is True
    assert view.staleness.removed_ids == (_B,)


# ---------- presence ----------


def test_concept_presence_returns_docs_and_chunk_keys(env: Path) -> None:
    _seed_concepts((_A, "Embeddings"))
    _seed_presence(_A, "d1", ["d1:p0", "d1:p3"], 7)
    rows = load_concept_presence(_A)
    assert len(rows) == 1
    assert rows[0].document_id == "d1"
    assert rows[0].chunk_keys == ("d1:p0", "d1:p3")
    assert rows[0].n_mentions == 7


def test_concept_presence_unknown_concept_is_empty(env: Path) -> None:
    assert load_concept_presence("nope") == []


# ---------- routes ----------


def _client(**kw: object) -> TestClient:
    return TestClient(create_app(controller=_FakeController(), **kw))  # type: ignore[arg-type]


def test_route_graph_200_shape(env: Path) -> None:
    _write_skeleton_json(env, _skeleton())
    _seed_concepts((_A, "Embeddings"), (_B, "BM25"))
    r = _client().get("/api/concepts/graph")
    assert r.status_code == 200
    body = r.json()
    assert body["graph_version"] == "cafef00d"
    assert len(body["nodes"]) == 2
    assert len(body["edges"]) == 1
    assert len(body["communities"]) == 1
    assert body["staleness"]["stale"] is False
    # One id space: edge endpoints are concept UUIDs that resolve against node ids (KI-15).
    node_ids = {n["id"] for n in body["nodes"]}
    edge = body["edges"][0]
    assert {edge["source"], edge["target"]} <= node_ids
    assert edge["provenance"] == ["cooccurrence", "similarity"]
    assert edge["relation"] is None  # Node B never run — no stance on the wire


def test_route_graph_404_when_never_built(env: Path) -> None:
    """The empty state is a 404 with a rebuild hint — not a 500, and not a fake empty graph."""
    r = _client().get("/api/concepts/graph")
    assert r.status_code == 404
    assert "not built yet" in r.json()["detail"]


def test_route_presence(env: Path) -> None:
    _seed_concepts((_A, "Embeddings"))
    _seed_presence(_A, "d1", ["d1:p0"], 2)
    r = _client().get(f"/api/concepts/{_A}/presence")
    assert r.status_code == 200
    assert r.json() == [{"document_id": "d1", "chunk_keys": ["d1:p0"], "n_mentions": 2}]


def test_route_rebuild_202_and_status_reports_done(env: Path) -> None:
    """202 + poll, mirroring /api/ingest. The seam means no real Node-A build runs."""
    client = _client(rebuild_graph_fn=lambda: "newversion")
    r = client.post("/api/concepts/graph/rebuild")
    assert r.status_code == 202
    assert r.json()["state"] in {"running", "done"}
    # The worker is a daemon thread; poll the status route until it settles.
    for _ in range(200):
        body = client.get("/api/concepts/graph/rebuild/status").json()
        if body["state"] == "done":
            break
    assert body["state"] == "done"
    assert body["graph_version"] == "newversion"


def test_route_rebuild_reports_error_without_killing_the_thread(env: Path) -> None:
    def _boom() -> str:
        raise RuntimeError("node A exploded")

    client = _client(rebuild_graph_fn=_boom)
    assert client.post("/api/concepts/graph/rebuild").status_code == 202
    for _ in range(200):
        body = client.get("/api/concepts/graph/rebuild/status").json()
        if body["state"] == "error":
            break
    assert body["state"] == "error"
    assert "node A exploded" in body["message"]


def test_route_rebuild_409_while_one_is_running(env: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def _slow() -> str:
        started.set()
        release.wait(timeout=5)
        return "v2"

    client = _client(rebuild_graph_fn=_slow)
    assert client.post("/api/concepts/graph/rebuild").status_code == 202
    assert started.wait(timeout=5)
    try:
        r = client.post("/api/concepts/graph/rebuild")
        assert r.status_code == 409
        assert "already running" in r.json()["detail"]
    finally:
        release.set()
