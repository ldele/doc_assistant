"""Integration test for the concept-graph build (Feature 7) — impure orchestration.

Drives ``concept_graph.build_concept_graph`` against a temp SQLite seeded with
documents, a temp ``CONCEPT_GRAPH_DIR``, a **fake** extraction client, and a
stubbed ``sample_doc_text`` (so it never reads the real Chroma store). Asserts
extraction → graph.json + per-doc cache written, dry-run does nothing,
idempotency (cache hit, no LLM), ``--force`` re-extracts, per-doc isolation, and
that the chunk store is never touched.

Deterministic + offline: no network, no real LLM, no Chroma.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.concept_graph as cg
import doc_assistant.db.session as session_mod
from doc_assistant.concept_graph import GRAPH_NAME, build_concept_graph
from doc_assistant.db.models import Base, Document
from doc_assistant.db.session import session_scope

_CANNED = (
    '{"concepts": ["RAG", "DPR"], '
    '"relations": [{"subject": "RAG", "relation": "uses", "object": "DPR"}]}'
)


class _FakeClient:
    """An ``LLMClient``-shaped fake: ``.complete`` returns a canned extraction JSON."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls = 0
        self._fail_on = fail_on

    def complete(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
        self.calls += 1
        content = messages[0]["content"]
        if self._fail_on and self._fail_on in content:
            raise RuntimeError("simulated transport failure")
        return _CANNED


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Temp SQLite bound to the global session machinery; `sample_doc_text` stubbed."""
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    # Never touch the real Chroma store during extraction.
    monkeypatch.setattr(cg, "sample_doc_text", lambda *a, **k: ["excerpt one", "excerpt two"])
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed(names: tuple[str, ...] = ("a", "b", "c")) -> dict[str, str]:
    ids: dict[str, str] = {}
    with session_scope() as session:
        for name in names:
            doc = Document(
                filename=f"{name}.pdf",
                source_original=f"{name}.pdf",
                doc_hash=f"h{name}",
                format="pdf",
            )
            session.add(doc)
            session.flush()
            ids[name] = str(doc.id)
    return ids


def test_apply_extracts_writes_graph_and_cache(env: Path) -> None:
    _seed()
    root = env / "graph"
    fake = _FakeClient()
    result = build_concept_graph(apply=True, client=fake, graph_dir=root)

    assert fake.calls == 3  # one extraction per document
    assert result.extracted == 3 and result.cached == 0 and result.errors == 0
    # Graph artifact written, with the merged nodes/edge.
    graph_json = root / GRAPH_NAME
    assert graph_json.exists()
    payload = json.loads(graph_json.read_text(encoding="utf-8"))
    assert {n["id"] for n in payload["nodes"]} == {"rag", "dpr"}
    assert payload["edges"][0]["integrity"] == "EXTRACTED"
    assert payload["edges"][0]["weight"] == 3  # supported by all 3 docs
    # Per-doc extraction cache written, keyed by doc_hash.
    caches = {p.name for p in (root / "extractions").glob("*.json")}
    assert caches == {"ha.json", "hb.json", "hc.json"}


def test_dry_run_makes_no_calls_and_writes_nothing(env: Path) -> None:
    _seed()
    root = env / "graph"
    fake = _FakeClient()
    result = build_concept_graph(apply=False, client=fake, graph_dir=root)
    assert fake.calls == 0
    assert result.extracted == 0 and result.skipped == 3
    assert result.graph.nodes == []  # nothing cached → empty graph
    assert not (root / GRAPH_NAME).exists()


def test_second_run_is_cached_no_llm(env: Path) -> None:
    _seed()
    root = env / "graph"
    build_concept_graph(apply=True, client=_FakeClient(), graph_dir=root)
    second_fake = _FakeClient()
    result = build_concept_graph(apply=True, client=second_fake, graph_dir=root)
    assert second_fake.calls == 0  # all served from cache
    assert result.cached == 3 and result.extracted == 0
    assert (root / GRAPH_NAME).exists()  # graph still (re)written from cache


def test_force_reextracts(env: Path) -> None:
    _seed()
    root = env / "graph"
    build_concept_graph(apply=True, client=_FakeClient(), graph_dir=root)
    fake = _FakeClient()
    result = build_concept_graph(apply=True, force=True, client=fake, graph_dir=root)
    assert fake.calls == 3 and result.extracted == 3 and result.cached == 0


def test_per_doc_isolation_one_failure_does_not_abort(env: Path) -> None:
    _seed()
    root = env / "graph"
    fake = _FakeClient(fail_on="b.pdf")  # the prompt for doc b names b.pdf
    result = build_concept_graph(apply=True, client=fake, graph_dir=root)
    assert result.errors == 1
    assert result.extracted == 2  # a and c still extracted
    assert (root / GRAPH_NAME).exists()
    caches = {p.name for p in (root / "extractions").glob("*.json")}
    assert caches == {"ha.json", "hc.json"}  # the failed doc left no cache


def test_no_chunk_store_mutation(env: Path) -> None:
    _seed()
    root = env / "graph"
    build_concept_graph(apply=True, client=_FakeClient(), graph_dir=root)
    assert not (env / "chroma").exists()
    with session_scope() as session:
        from sqlalchemy import func, select

        doc_count = session.execute(select(func.count()).select_from(Document)).scalar_one()
    assert doc_count == 3  # unchanged
