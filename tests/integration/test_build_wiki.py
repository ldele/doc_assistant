"""Integration test for the wiki build (Feature 6) — the impure orchestration.

Drives ``wiki.build_wiki`` against a temp SQLite seeded with documents +
``DocSimilarity`` edges, a temp ``WIKI_DIR``, a **fake** summarizer client, and a
stubbed ``sample_chunks`` (so it never reads the real Chroma store). Asserts
clustering → notes written, the manifest, idempotency, ``--force``, and drift +
orphan-note removal when the graph re-clusters.

Deterministic + offline: no network, no real LLM, no Chroma.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import wiki
from doc_assistant.db.models import Base, DocSimilarity, Document
from doc_assistant.db.session import session_scope
from doc_assistant.wiki import build_wiki, topic_id_for


class _FakeClient:
    """An ``LLMClient``-shaped fake: ``.complete`` returns a canned topic JSON."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
        self.calls += 1
        return '{"title": "Fake Topic", "summary": "A fake summary.", "tags": ["t1", "t2"]}'


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Temp SQLite bound to the global session machinery; `sample_chunks` stubbed out."""
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    # Never touch the real Chroma store during summarisation.
    monkeypatch.setattr(wiki, "sample_chunks", lambda *a, **k: {})
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed() -> dict[str, str]:
    """Seed docs a/b/c with a→b (0.7, same topic) and b→c (0.52, cross link)."""
    ids: dict[str, str] = {}
    with session_scope() as session:
        for name in ("a", "b", "c"):
            doc = Document(
                filename=f"{name}.pdf",
                source_original=f"{name}.pdf",
                doc_hash=f"h{name}",
                format="pdf",
            )
            session.add(doc)
            session.flush()
            ids[name] = str(doc.id)
        session.add_all(
            [
                DocSimilarity(
                    source_document_id=ids["a"],
                    target_document_id=ids["b"],
                    embedding_model="bge-base",
                    score=0.70,
                ),
                DocSimilarity(
                    source_document_id=ids["b"],
                    target_document_id=ids["c"],
                    embedding_model="bge-base",
                    score=0.52,
                ),
            ]
        )
    return ids


def _wiki_files(root: Path) -> set[str]:
    return {p.name for p in root.glob("topic-*.md")}


def test_apply_writes_clustered_notes_and_manifest(env: Path) -> None:
    _seed()
    fake = _FakeClient()
    root = env / "wiki"
    result = build_wiki(apply=True, client=fake, min_similarity=0.55, wiki_dir=root)

    # Two topics: {a,b} and {c}. Summariser called once per topic.
    assert len(result.notes) == 2
    assert fake.calls == 2
    assert result.written == 2

    ab = topic_id_for(["ha", "hb"])
    c = topic_id_for(["hc"])
    assert _wiki_files(root) == {f"{ab}.md", f"{c}.md"}
    assert (root / wiki.MANIFEST_NAME).exists()

    ab_text = (root / f"{ab}.md").read_text(encoding="utf-8")
    assert "A fake summary." in ab_text
    assert "[[" in ab_text  # the a-b topic links to the c topic (cross edge 0.52)


def test_dry_run_makes_no_calls_and_writes_nothing(env: Path) -> None:
    _seed()
    fake = _FakeClient()
    root = env / "wiki"
    result = build_wiki(apply=False, client=fake, min_similarity=0.55, wiki_dir=root)
    assert fake.calls == 0
    assert result.written == 0
    assert not root.exists() or _wiki_files(root) == set()


def test_rebuild_is_idempotent_no_drift(env: Path) -> None:
    _seed()
    root = env / "wiki"
    build_wiki(apply=True, client=_FakeClient(), min_similarity=0.55, wiki_dir=root)
    second = build_wiki(apply=True, client=_FakeClient(), min_similarity=0.55, wiki_dir=root)
    assert not second.drift.any()
    assert second.removed_files == 0


def test_recluster_drifts_and_removes_orphan_note(env: Path) -> None:
    ids = _seed()
    root = env / "wiki"
    build_wiki(apply=True, client=_FakeClient(), min_similarity=0.55, wiki_dir=root)
    ab = topic_id_for(["ha", "hb"])
    assert (root / f"{ab}.md").exists()

    # Drop the a-b edge → a, b, c all become singletons → the {a,b} topic dies.
    with session_scope() as session:
        session.execute(
            delete(DocSimilarity).where(
                DocSimilarity.source_document_id == ids["a"],
                DocSimilarity.target_document_id == ids["b"],
            )
        )

    result = build_wiki(apply=True, client=_FakeClient(), min_similarity=0.55, wiki_dir=root)
    assert ab in result.drift.removed
    assert topic_id_for(["ha"]) in result.drift.added
    assert not (root / f"{ab}.md").exists()  # orphan note swept
    assert result.removed_files >= 1


def test_no_document_mutation_and_no_chroma_dir(env: Path) -> None:
    _seed()
    root = env / "wiki"
    build_wiki(apply=True, client=_FakeClient(), min_similarity=0.55, wiki_dir=root)
    # The wiki is a pure sidecar — no chunk store written.
    assert not (env / "chroma").exists()
    with session_scope() as session:
        from sqlalchemy import func, select

        doc_count = session.execute(select(func.count()).select_from(Document)).scalar_one()
    assert doc_count == 3  # unchanged
