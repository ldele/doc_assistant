"""`ingest/registry.py` — the multi-root half (ADR-046, AD3b).

The registry was single-root by construction: `SourceFile` was keyed by `rel_path` alone, which
is why referencing a file outside the one source dir was a **schema** change rather than a UI one.
These tests pin the four properties that change is supposed to buy, and the two it must not break:

* the same relative path may exist under two roots, and they are two rows;
* a root that cannot be reached right now is reported as *unavailable*, not as a mass deletion;
* an unreachable root's rows survive it — nothing is deleted and `last_seen` is not touched;
* the bare-`rel_path` shorthand still means the library root, so pre-AD3b callers are unchanged.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def temp_database(monkeypatch, tmp_path):
    """Isolated engine + session factory, with the library root seeded as `init_db` would."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    from doc_assistant.db.models import LIBRARY_ROOT_ID, Base, SourceRoot

    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, future=True)() as s:
        s.add(SourceRoot(id=LIBRARY_ROOT_ID, path=str(tmp_path / "library"), kind="library"))
        s.commit()
    yield path
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


def _write(path: Path, data: bytes = b"%PDF-1.4 x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ============================================================
# The key helpers — pure, and the shorthand is the compatibility surface.
# ============================================================


def test_a_bare_rel_path_still_means_the_library_root():
    """Every caller written before AD3b passes a bare rel_path. It must keep working."""
    from doc_assistant.ingest.registry import split_key

    assert split_key("papers/rag.pdf", {"library", "abc"}) == ("library", "papers/rag.pdf")


def test_a_prefix_is_only_a_root_when_it_actually_is_one():
    """A POSIX filename may contain a colon; splitting blindly would invent a root from it."""
    from doc_assistant.ingest.registry import split_key

    # `my` is not a registered root, so the whole string is a library-root rel_path.
    assert split_key("my:notes.pdf", {"library"}) == ("library", "my:notes.pdf")
    # ...and when it is one, the split happens.
    assert split_key("abc:notes.pdf", {"library", "abc"}) == ("abc", "notes.pdf")


def test_source_key_round_trips():
    from doc_assistant.ingest.registry import source_key, split_key

    assert split_key(source_key("abc", "a/b.pdf"), {"abc"}) == ("abc", "a/b.pdf")


# ============================================================
# Roots
# ============================================================


def test_moving_the_library_updates_the_row_rather_than_replacing_it(temp_database, tmp_path):
    """The row's id is what every SourceFile.root_id points at — replacing it orphans them all."""
    from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceRoot
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import ensure_library_root

    moved = tmp_path / "new-library"
    moved.mkdir()
    with session_scope() as session:
        root = ensure_library_root(session, moved)
        assert root.id == LIBRARY_ROOT_ID
        assert Path(root.path) == moved.resolve()
        assert session.execute(select(func.count()).select_from(SourceRoot)).scalar_one() == 1


def test_registering_the_same_folder_twice_reuses_its_root(temp_database, tmp_path):
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import register_root

    folder = tmp_path / "zotero"
    folder.mkdir()
    with session_scope() as session:
        first = register_root(session, folder)
        second = register_root(session, folder)
        assert first.id == second.id


# ============================================================
# Scanning across roots
# ============================================================


def test_the_same_rel_path_under_two_roots_is_two_rows(temp_database, tmp_path):
    """The whole point of the key change: `paper.pdf` may exist in the library AND in Zotero."""
    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import ensure_library_root, register_root, scan_sources

    library = tmp_path / "library"
    zotero = tmp_path / "zotero"
    _write(library / "paper.pdf")
    _write(zotero / "paper.pdf")

    with session_scope() as session:
        ensure_library_root(session, library)
        register_root(session, zotero)
        views = scan_sources(session, library)

    assert [v.rel_path for v in views] == ["paper.pdf", "paper.pdf"]
    assert {v.root_kind for v in views} == {"library", "referenced"}
    with session_scope() as session:
        assert session.execute(select(func.count()).select_from(SourceFile)).scalar_one() == 2


def test_the_library_root_is_listed_first(temp_database, tmp_path):
    """Stable ordering, so the UI is not at the mercy of dict order."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import list_roots, register_root

    library = tmp_path / "library"
    library.mkdir()
    (tmp_path / "aaa-sorts-first").mkdir()
    with session_scope() as session:
        register_root(session, tmp_path / "aaa-sorts-first")
        roots = list_roots(session, library)
    assert roots[0].kind == "library"


# ============================================================
# The unavailable root — the distinction the honest-degradation contract asks for.
# ============================================================


def test_an_unreachable_root_reports_unavailable_rather_than_mass_deletion(
    temp_database, tmp_path
):
    """An unplugged drive must not be indistinguishable from the user losing 400 documents."""
    import shutil

    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import ensure_library_root, register_root, scan_sources

    library = tmp_path / "library"
    library.mkdir()
    drive = tmp_path / "external-drive"
    _write(drive / "paper.pdf")

    with session_scope() as session:
        ensure_library_root(session, library)
        register_root(session, drive)
        scan_sources(session, library)

    shutil.rmtree(drive)  # the drive is unplugged

    with session_scope() as session:
        views = scan_sources(session, library)

    (row,) = [v for v in views if v.root_kind == "referenced"]
    assert row.status == "missing", "the file genuinely is not readable right now"
    assert row.root_available is False, "and this is what says *why* — not a deletion"


def test_an_unreachable_root_keeps_its_rows_and_their_last_seen(temp_database, tmp_path):
    """The rows have to survive the drive being plugged back in, untouched while it is gone."""
    import shutil

    from doc_assistant.db.models import SourceFile
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import ensure_library_root, register_root, scan_sources

    library = tmp_path / "library"
    library.mkdir()
    drive = tmp_path / "external-drive"
    _write(drive / "paper.pdf")

    with session_scope() as session:
        ensure_library_root(session, library)
        register_root(session, drive)
        scan_sources(session, library)
    with session_scope() as session:
        before = session.execute(
            select(SourceFile.last_seen).where(SourceFile.rel_path == "paper.pdf")
        ).scalar_one()

    shutil.rmtree(drive)
    with session_scope() as session:
        scan_sources(session, library)

    with session_scope() as session:
        rows = session.execute(select(SourceFile)).scalars().all()
        assert len(rows) == 1, "an unavailable root must never delete its rows"
        assert rows[0].last_seen == before, "nor claim it saw the file while the drive was gone"


def test_an_unreachable_root_does_not_block_ingesting_the_library(temp_database, tmp_path):
    """One unplugged drive must not fail the whole selection — the library is still present."""
    import shutil

    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import ensure_library_root, register_root, resolve_selection

    library = tmp_path / "library"
    _write(library / "here.pdf")
    drive = tmp_path / "external-drive"
    _write(drive / "gone.pdf")

    with session_scope() as session:
        ensure_library_root(session, library)
        register_root(session, drive)

    shutil.rmtree(drive)

    with session_scope() as session:
        resolved = resolve_selection(session, library, None)

    assert [p.name for p in resolved] == ["here.pdf"]


# ============================================================
# Selection across roots
# ============================================================


def test_a_referenced_file_can_be_selected_by_its_key(temp_database, tmp_path):
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import (
        ensure_library_root,
        register_root,
        resolve_selection,
        source_key,
    )

    library = tmp_path / "library"
    _write(library / "here.pdf")
    zotero = tmp_path / "zotero"
    _write(zotero / "theirs.pdf")

    with session_scope() as session:
        ensure_library_root(session, library)
        root = register_root(session, zotero)
        resolved = resolve_selection(session, library, [source_key(root.id, "theirs.pdf")])

    assert [p.name for p in resolved] == ["theirs.pdf"]


def test_traversal_is_still_caught_when_a_root_prefix_is_involved(temp_database, tmp_path):
    """Regression guard. Validating the composite key instead of the rel_path defeats this:
    `PurePosixPath("library:../evil.pdf").parts` is `("library:..", "evil.pdf")`, so `".." in
    parts` is False and the traversal walks straight through."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.registry import InvalidSelection, ensure_library_root

    library = tmp_path / "library"
    _write(library / "here.pdf")

    from doc_assistant.ingest.registry import resolve_selection

    with session_scope() as session:
        ensure_library_root(session, library)
        for attempt in ("../evil.pdf", "library:../evil.pdf"):
            with pytest.raises(InvalidSelection) as ei:
                resolve_selection(session, library, [attempt])
            assert ei.value.offenders["traversal"], f"{attempt!r} must read as traversal"


# ============================================================
# Exclusions survive path normalisation (review finding 2)
# ============================================================


def test_an_exclusion_still_applies_when_the_walk_root_is_a_short_path(temp_database, tmp_path):
    """The two sides of the exclusion test must meet in the same normalised form.

    `SourceRoot.path` is stored resolved; `registry.pathkey` normalises case and separators but
    does **not** expand 8.3 short names, junctions or symlinks. When `_resolve_walk_root` returned
    `config.DOCS_PATH` raw, a data dir reached through any of those produced keys that could never
    match, and every standing exclusion was silently ignored — `skipped=0`, no error, no warning.

    Windows-only because 8.3 aliasing is the cheap way to make `abspath` and `resolve` disagree
    without creating a symlink (which needs privileges).
    """
    import ctypes

    from doc_assistant import config
    from doc_assistant.db.models import LIBRARY_ROOT_ID, SourceFile, SourceRoot
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest import _drop_excluded, _resolve_walk_root

    if os.name != "nt":
        pytest.skip("8.3 short names are a Windows filesystem feature")

    sources = tmp_path / "A Folder With A Long Name"
    sources.mkdir()
    (sources / "excluded.pdf").write_bytes(b"%PDF-1.4 x")

    buf = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(sources), buf, 1024):
        pytest.skip("8.3 name generation is disabled on this volume")
    short = Path(buf.value)
    if short == sources:
        pytest.skip("no distinct 8.3 alias for this path")

    with session_scope() as session:
        # The fixture already seeded the root, as `init_db` would; point it at this test's folder
        # in the same *resolved* form every writer of `SourceRoot.path` uses.
        root = session.get(SourceRoot, LIBRARY_ROOT_ID)
        assert root is not None
        root.path = str(sources.resolve())
        session.flush()
        session.add(
            SourceFile(
                root_id=LIBRARY_ROOT_ID,
                rel_path="excluded.pdf",
                format="pdf",
                size=10,
                mtime=0.0,
                excluded=True,
            )
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "DOCS_PATH", short)
        walk_root = _resolve_walk_root(None)
        walked = [p for p in walk_root.rglob("*") if p.is_file()]
        kept, skipped = _drop_excluded(walked)

    assert skipped == 1, "the exclusion must be honoured through the short-name alias"
    assert kept == []
