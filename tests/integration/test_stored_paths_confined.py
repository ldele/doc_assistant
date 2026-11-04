"""No write or served file through a stored path leaves the open library (ADR-061 choice 4).

The review's must-hold G1 for 0.7.2's piece 5, and `docs/security.md` step S-7. The library here is
a copy: its rows still name the files of the library it was copied from, in another folder, which
is what a library copied with a file manager looks like. Each path that deletes, overwrites or
serves a file through a stored path is driven against it, and the original's files must come
through untouched. The figure crops and the figure route have their cases beside their own tests
(`test_reingest_figure_crops.py`, `ingest/test_figures_extract.py`).

Offline: a temp SQLite, temp folders, fake vector stores. No model.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import app_settings, config, reingest
from doc_assistant.db.models import Base, Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest import figures as figures_mod
from doc_assistant.ingest.cache import spacing_record_path

ORIGINAL_TEXT = "the original library's file"


class FakeChroma:
    """The two calls a delete and the orphan sweep make, over fixed metadata."""

    def __init__(self, metadatas: list[dict[str, Any]]) -> None:
        self.metadatas = metadatas
        self.deleted: list[Any] = []

    def get(self, **kwargs: Any) -> dict[str, Any]:
        return {"ids": [f"c{i}" for i in range(len(self.metadatas))], "metadatas": self.metadatas}

    def delete(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs)


@pytest.fixture
def copied_library(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SimpleNamespace]:
    """An open library at `tmp_path/library` whose one document names the original's files."""
    lib = tmp_path / "library"
    for sub in ("sources", "cache", "figures"):
        (lib / sub).mkdir(parents=True)
    monkeypatch.setattr(config, "DOCS_PATH", lib / "sources")
    monkeypatch.setattr(config, "CACHE_PATH", lib / "cache")
    monkeypatch.setattr(figures_mod, "FIGURE_DIR", lib / "figures")
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", lib / "settings.json")
    monkeypatch.delenv("DOC_SOURCE_DIR", raising=False)

    original = tmp_path_factory.mktemp("original-library")
    source = original / "sources" / "paper.md"
    cache = original / "cache" / "paper.md"
    for path in (source, cache, spacing_record_path(cache)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ORIGINAL_TEXT, encoding="utf-8")

    engine = create_engine(f"sqlite:///{lib / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)
    with session_scope() as session:
        doc = Document(
            filename="paper.md",
            source_original=str(source),
            source_cache=str(cache),
            doc_hash="h-paper",
            format="md",
        )
        session.add(doc)
        session.flush()
        doc_id = str(doc.id)
    try:
        yield SimpleNamespace(lib=lib, source=source, cache=cache, doc_id=doc_id)
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _untouched(env: SimpleNamespace) -> None:
    for path in (env.source, env.cache, spacing_record_path(env.cache)):
        assert path.exists(), f"{path.name} was removed from the original library"
        assert path.read_text(encoding="utf-8") == ORIGINAL_TEXT


def test_a_delete_leaves_the_originals_source_and_cache_alone(
    copied_library: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant.library import delete_document

    monkeypatch.setattr(
        "send2trash.send2trash", lambda p: pytest.fail(f"sent {p} to the Recycle Bin")
    )
    result = delete_document(copied_library.doc_id, FakeChroma([]), delete_file=True)

    assert result is not None and result.trashed_file is False
    _untouched(copied_library)


def test_the_viewer_says_the_file_is_outside_this_library(copied_library: SimpleNamespace) -> None:
    from doc_assistant.library.source_view import get_source_view

    view = get_source_view(copied_library.doc_id)

    assert view is not None and view.available is False
    assert "outside this library's folders" in (view.reason or "")


def test_reveal_refuses_a_file_outside_this_library(
    copied_library: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant.library import documents

    monkeypatch.setattr(
        documents, "_reveal_in_file_manager", lambda p: pytest.fail(f"revealed {p}")
    )
    assert documents.reveal_document_source(copied_library.doc_id) is False


def test_the_copys_own_file_is_found_in_its_sources_folder(
    copied_library: SimpleNamespace,
) -> None:
    """The fallback a copy needs: its stored path names the original, its file is in sources/."""
    from doc_assistant.library.documents import resolve_source_path

    own = copied_library.lib / "sources" / "paper.md"
    own.write_text("the copy's file", encoding="utf-8")

    assert resolve_source_path(str(copied_library.source), "paper.md") == own


def test_a_text_rerun_refuses_to_delete_the_originals_cache(
    copied_library: SimpleNamespace,
) -> None:
    (copied_library.lib / "sources" / "paper.md").write_text("the copy's file", encoding="utf-8")

    result = reingest.rerun([copied_library.doc_id], ["text"])

    outcome = result.outcomes[0]
    assert outcome.status == "error"
    assert "outside this library" in outcome.detail
    _untouched(copied_library)


def test_the_orphan_sweep_deletes_only_caches_inside_this_library(
    copied_library: SimpleNamespace,
) -> None:
    from doc_assistant.ingest.cleanup import cleanup_orphans_chroma

    inside = copied_library.lib / "cache" / "gone.md"
    inside.write_text("a cache of this library's own", encoding="utf-8")
    gone_source = copied_library.lib / "sources" / "gone.md"  # never created: the source is gone
    db = FakeChroma(
        [
            {
                "doc_hash": "h-paper",
                "source_original": str(copied_library.lib / "sources" / "deleted.md"),
                "source_cache": str(copied_library.cache),
            },
            {
                "doc_hash": "h-gone",
                "source_original": str(gone_source),
                "source_cache": str(inside),
            },
        ]
    )

    cleanup_orphans_chroma(db, ["h-paper", "h-gone"], also_clean_cache=True)  # type: ignore[arg-type]

    assert not inside.exists()  # this library's orphan cache goes, as before
    _untouched(copied_library)  # the original's, named by a stored path, stays
