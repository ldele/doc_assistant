"""Integration tests for selective ingestion (feature-selective-ingestion.md, S1).

All offline, tmp-dir, no corpus, no network, no real embeddings (cpc §13). A tmp SQLite (like
`test_api_library.py`) plus tmp source/cache dirs let us exercise the stat-only registry, the
selection resolution, the dry-run plan, and the HTTP wiring without loading a model or opening
Chroma. Ingested state is *simulated* (a `Document` row + a fresh cache file), never really run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield
    engine.dispose()


@pytest.fixture
def sources_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, temp_db: None) -> Path:
    """Tmp source + cache dirs wired into config, and a source dir the API resolves to."""
    from doc_assistant import app_settings, config

    src = tmp_path / "sources"
    src.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(config, "DOCS_PATH", src)
    monkeypatch.setattr(config, "CACHE_PATH", cache)
    monkeypatch.setattr(app_settings, "get_source_dir", lambda: src)
    return src


def _status(views: list, rel: str) -> str:
    return next(v.status for v in views if v.rel_path == rel)


def _mark_ingested(file: Path) -> None:
    """Simulate a completed ingest: a `Document` row + a cache file newer than the source."""
    from doc_assistant.ingest.cache import get_cache_path

    with session_scope() as s:
        s.add(
            Document(
                filename=file.name,
                source_original=str(file),
                doc_hash=f"h-{file.name}",
                format="pdf",
            )
        )
    cache = get_cache_path(file)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("cached", encoding="utf-8")
    sm = file.stat().st_mtime
    os.utime(cache, (sm + 100, sm + 100))  # cache newer than source → fresh


def _touch_newer_than_cache(file: Path) -> None:
    from doc_assistant.ingest.cache import get_cache_path

    cm = get_cache_path(file).stat().st_mtime
    os.utime(file, (cm + 100, cm + 100))  # source newer than cache → changed


# --- registry scan lifecycle ------------------------------------------------------------


def test_scan_lifecycle_new_ingested_changed_missing(sources_env: Path) -> None:
    from doc_assistant.ingest import registry

    src = sources_env
    a = src / "a.pdf"
    a.write_text("hello")

    with session_scope() as s:
        assert _status(registry.scan_sources(s, src), "a.pdf") == "new"

    _mark_ingested(a)
    with session_scope() as s:
        assert _status(registry.scan_sources(s, src), "a.pdf") == "ingested"

    _touch_newer_than_cache(a)
    with session_scope() as s:
        assert _status(registry.scan_sources(s, src), "a.pdf") == "changed"

    a.unlink()
    with session_scope() as s:
        # the row survives the file's disappearance and derives `missing`
        assert _status(registry.scan_sources(s, src), "a.pdf") == "missing"


def test_scan_is_stat_only_and_persists_rows(sources_env: Path) -> None:
    from doc_assistant.db.models import SourceFile
    from doc_assistant.ingest import registry

    src = sources_env
    (src / "a.pdf").write_text("x")
    (src / "b.pdf").write_text("y")
    with session_scope() as s:
        views = registry.scan_sources(s, src)
    assert {v.rel_path for v in views} == {"a.pdf", "b.pdf"}
    # rows were committed, and doc_type is dormant (always null)
    with session_scope() as s:
        rows = s.query(SourceFile).all()
        assert {r.rel_path for r in rows} == {"a.pdf", "b.pdf"}
        assert all(r.doc_type is None and r.excluded is False for r in rows)


# --- resolve_selection ------------------------------------------------------------------


def test_resolve_selection_excludes_then_explicit_overrides(sources_env: Path) -> None:
    from doc_assistant.ingest import registry

    src = sources_env
    (src / "a.pdf").write_text("x")
    (src / "b.pdf").write_text("y")
    with session_scope() as s:
        registry.scan_sources(s, src)
        registry.set_source_meta(s, "a.pdf", excluded=True)

    with session_scope() as s:  # implicit walk skips the excluded file
        assert {p.name for p in registry.resolve_selection(s, src, None)} == {"b.pdf"}

    with session_scope() as s:  # an explicit pick overrides the exclusion
        assert {p.name for p in registry.resolve_selection(s, src, ["a.pdf"])} == {"a.pdf"}


def test_resolve_selection_rejects_bad_paths(sources_env: Path) -> None:
    from doc_assistant.ingest import registry

    src = sources_env
    (src / "a.pdf").write_text("x")
    with session_scope() as s, pytest.raises(registry.InvalidSelection) as ei:
        registry.resolve_selection(s, src, ["../evil.pdf", "ghost.pdf"])
    assert "../evil.pdf" in ei.value.offenders["traversal"]
    assert "ghost.pdf" in ei.value.offenders["unknown"]


# --- dry run: reports the plan, never loads embeddings ----------------------------------


def test_main_dry_run_reports_plan_without_embeddings(
    sources_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doc_assistant.ingest as ingest_pkg

    src = sources_env
    (src / "a.pdf").write_text("x")
    (src / "b.pdf").write_text("y")

    def _trap(*_a: object, **_k: object) -> object:
        raise AssertionError("get_embeddings must not be called during a dry run")

    monkeypatch.setattr(ingest_pkg, "get_embeddings", _trap)
    plan = ingest_pkg.main(dry_run=True)
    assert plan == {"would_add": 2, "would_reembed": 0, "skip_unchanged": 0, "excluded": 0}


def test_main_dry_run_counts_excluded(sources_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import doc_assistant.ingest as ingest_pkg
    from doc_assistant.ingest import registry

    src = sources_env
    (src / "a.pdf").write_text("x")
    (src / "b.pdf").write_text("y")
    with session_scope() as s:
        registry.scan_sources(s, src)
        registry.set_source_meta(s, "a.pdf", excluded=True)

    def _trap(*_a: object, **_k: object) -> object:
        raise AssertionError("no embeddings in a dry run")

    monkeypatch.setattr(ingest_pkg, "get_embeddings", _trap)
    plan = ingest_pkg.main(dry_run=True)
    assert plan["excluded"] == 1
    assert plan["would_add"] == 1  # only b.pdf remains


# --- API flow ---------------------------------------------------------------------------


class _FakeController:
    def chunk_count(self) -> int:
        return 0


def _poll_status(client: TestClient, *, state: str, tries: int = 60) -> dict:
    import time

    for _ in range(tries):
        st = client.get("/api/ingest/status").json()
        if st["state"] == state:
            return st
        time.sleep(0.05)
    return client.get("/api/ingest/status").json()


def test_api_sources_scan_patch_and_selective_ingest(sources_env: Path) -> None:
    src = sources_env
    (src / "a.pdf").write_text("x")
    (src / "b.pdf").write_text("y")
    seen: dict[str, object] = {}

    def fake_ingest(
        *, scope: str | None = None, files: list[Path] | None = None
    ) -> dict[str, int]:
        seen["scope"] = scope
        seen["files"] = files
        return {"added": 1, "skipped": 0, "error": 0}

    app = create_app(
        controller=_FakeController(),  # type: ignore[arg-type]
        ingest_fn=fake_ingest,
        controller_factory=_FakeController,
    )
    client = TestClient(app)

    # GET scans the dir → both files new, doc_type null, not excluded
    rows = client.get("/api/sources").json()
    assert {r["rel_path"]: r["status"] for r in rows} == {"a.pdf": "new", "b.pdf": "new"}
    assert all(r["doc_type"] is None and r["excluded"] is False for r in rows)

    # PATCH excludes a; the echoed view reflects it
    r = client.patch("/api/sources", json={"rel_path": "a.pdf", "excluded": True})
    assert r.status_code == 200 and r.json()["excluded"] is True

    # PATCH an unknown rel_path → 404
    assert (
        client.patch("/api/sources", json={"rel_path": "ghost.pdf", "excluded": True}).status_code
        == 404
    )

    # POST /api/ingest {paths:[b]} → the worker gets an explicit files= selection (not scope)
    assert client.post("/api/ingest", json={"paths": ["b.pdf"]}).status_code == 202
    assert _poll_status(client, state="done")["state"] == "done"
    assert seen["scope"] is None
    got = seen["files"]
    assert isinstance(got, list) and [p.name for p in got] == ["b.pdf"]

    # POST with a traversal path → 400 (nothing starts)
    assert client.post("/api/ingest", json={"paths": ["../evil.pdf"]}).status_code == 400


def test_api_ingest_no_body_still_works(sources_env: Path) -> None:
    """No-body POST keeps the whole-dir behavior — the scope path, not a selection."""
    src = sources_env
    (src / "a.pdf").write_text("x")
    seen: dict[str, object] = {}

    def fake_ingest(
        *, scope: str | None = None, files: list[Path] | None = None
    ) -> dict[str, int]:
        seen["scope"] = scope
        seen["files"] = files
        return {"added": 1, "skipped": 0, "error": 0}

    app = create_app(
        controller=_FakeController(),  # type: ignore[arg-type]
        ingest_fn=fake_ingest,
        controller_factory=_FakeController,
    )
    client = TestClient(app)
    assert client.post("/api/ingest").status_code == 202
    _poll_status(client, state="done")
    assert seen["files"] is None and seen["scope"] == str(src.resolve())
