"""ADR-026 — ``document_meta.document_id`` gets the foreign key it shipped without.

Without an FK the override outlived its document: the orphan sweep and the old ``--rebuild`` bulk
delete (KI-24) removed ``Document`` rows without touching this table, leaving rows nothing could
read and nothing would clean. ``delete_document`` compensated by deleting the override by hand,
which is exactly the kind of correctness-by-remembering an FK exists to retire.

SQLite cannot ``ALTER`` a constraint, so this is the project's first **rebuild** migration
(create → copy → drop → rename). These tests build a genuinely pre-migration table — the real
``CREATE TABLE`` the old model produced, no FK — and drive ``init_db`` over it.

Temp file-backed SQLite; no LLM, no model load, no network.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.migrations import init_db
from doc_assistant.db.models import Base, Document, DocumentMeta
from doc_assistant.db.session import session_scope

# Verbatim what ``create_all`` produced for the pre-ADR-026 model: same columns, same
# nullability, no FOREIGN KEY clause.
_LEGACY_DDL = """
CREATE TABLE document_meta (
    document_id VARCHAR NOT NULL,
    title_override TEXT,
    authors_override TEXT,
    year_override INTEGER,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (document_id)
)
"""


@pytest.fixture
def legacy_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """A database whose ``document_meta`` predates the foreign key."""
    db_path = tmp_path / "library.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)

    # Everything except document_meta, which is then created in its pre-FK shape.
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE document_meta"))
        conn.execute(text(_LEGACY_DDL))

    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    monkeypatch.setattr(session_mod, "get_engine", lambda: engine)
    monkeypatch.setattr("doc_assistant.db.migrations.get_engine", lambda: engine)
    monkeypatch.setattr("doc_assistant.db.migrations.SQLITE_PATH", str(db_path))
    try:
        yield engine
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed_document(doc_id: str, filename: str) -> None:
    with session_scope() as session:
        session.add(
            Document(
                id=doc_id,
                filename=filename,
                source_original=f"/tmp/{filename}",
                doc_hash=f"hash-{filename}",
                format="pdf",
            )
        )


def _insert_override(engine: Engine, doc_id: str, title: str) -> None:
    """Insert straight through SQL — the ORM would refuse an orphan once the FK exists."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO document_meta (document_id, title_override, updated_at) "
                "VALUES (:d, :t, '2026-07-20 00:00:00')"
            ).bindparams(d=doc_id, t=title)
        )


def _has_fk(engine: Engine) -> bool:
    return any(
        fk.get("referred_table") == "documents"
        for fk in inspect(engine).get_foreign_keys("document_meta")
    )


def _overrides(engine: Engine) -> dict[str, str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT document_id, title_override FROM document_meta")).all()
    return {str(d): str(t) for d, t in rows}


# --- the migration ----------------------------------------------------------------------------- #


def test_legacy_table_really_has_no_fk(legacy_db: Engine) -> None:
    """Non-vacuity: the fixture must reproduce the broken shape, or nothing below proves
    anything."""
    assert _has_fk(legacy_db) is False


def test_init_db_adds_the_fk_and_keeps_live_overrides(legacy_db: Engine) -> None:
    _seed_document("doc-live", "live.pdf")
    _insert_override(legacy_db, "doc-live", "My own title")

    changed = init_db()

    assert _has_fk(legacy_db) is True
    assert _overrides(legacy_db) == {"doc-live": "My own title"}
    assert any("document_meta.document_id FK" in c for c in changed)


def test_orphaned_overrides_are_dropped_and_counted(legacy_db: Engine) -> None:
    """They are what the constraint forbids, and unrescuable — an override records no filename."""
    _seed_document("doc-live", "live.pdf")
    _insert_override(legacy_db, "doc-live", "keep me")
    _insert_override(legacy_db, "doc-vanished", "its document is long gone")

    changed = init_db()

    assert _overrides(legacy_db) == {"doc-live": "keep me"}
    assert any("1 orphan row(s) dropped" in c for c in changed)


def test_migration_is_idempotent(legacy_db: Engine) -> None:
    _seed_document("doc-live", "live.pdf")
    _insert_override(legacy_db, "doc-live", "My own title")

    init_db()
    second = init_db()

    assert second == []  # nothing left to change
    assert _has_fk(legacy_db) is True
    assert _overrides(legacy_db) == {"doc-live": "My own title"}


# --- what the FK buys -------------------------------------------------------------------------- #


def test_deleting_a_document_now_cascades_the_override(legacy_db: Engine) -> None:
    """The point of the whole change: no caller has to remember any more.

    ``delete_document`` still removes the override explicitly (ADR-014 path), but the bulk paths —
    the orphan sweep, ``_sweep_rebuild_rows`` — never did, and that is how orphans were made.
    """
    _seed_document("doc-live", "live.pdf")
    _insert_override(legacy_db, "doc-live", "My own title")
    init_db()

    with session_scope() as session:
        doc = session.get(Document, "doc-live")
        assert doc is not None
        session.delete(doc)  # the bulk-path shape: no explicit override cleanup

    assert _overrides(legacy_db) == {}


def test_orphan_sweep_no_longer_leaves_an_override_behind(legacy_db: Engine) -> None:
    """The live path that was still creating orphans after KI-24: ``cleanup_orphans_sqlite``."""
    _seed_document("doc-gone", "gone.pdf")
    _insert_override(legacy_db, "doc-gone", "notes on a file I deleted")
    init_db()

    from doc_assistant.ingest.cleanup import cleanup_orphans_sqlite

    class _FakeChroma:
        def get(self, **_: object) -> dict[str, list[dict[str, str]]]:
            meta = {"doc_hash": "hash-gone.pdf", "source_original": "/tmp/gone.pdf"}
            return {"metadatas": [meta]}

    removed = cleanup_orphans_sqlite(_FakeChroma())  # type: ignore[arg-type]

    assert removed == ["hash-gone.pdf"]
    assert _overrides(legacy_db) == {}


def test_an_override_for_an_unknown_document_is_now_rejected(legacy_db: Engine) -> None:
    """Structurally impossible, not merely tidied up afterwards."""
    init_db()

    with pytest.raises(IntegrityError), session_scope() as session:
        session.add(DocumentMeta(document_id="never-existed", title_override="nope"))


def test_a_failed_rebuild_leaves_the_old_table_intact(legacy_db: Engine) -> None:
    """The swap is atomic: a failure must not leave the table dropped-but-not-renamed.

    Forced with a lenient older variant of the table (``updated_at`` nullable) holding a NULL the
    model forbids — the migration refuses rather than inventing a value, and rolls the whole swap
    back.
    """
    with legacy_db.begin() as conn:
        conn.execute(text("DROP TABLE document_meta"))
        lenient = _LEGACY_DDL.replace("updated_at DATETIME NOT NULL", "updated_at DATETIME")
        conn.execute(text(lenient))
        conn.execute(
            text(
                "INSERT INTO document_meta (document_id, title_override, updated_at) "
                "VALUES ('doc-live', 'unmigratable', NULL)"
            )
        )
    _seed_document("doc-live", "live.pdf")

    with pytest.raises(IntegrityError):
        init_db()

    assert _has_fk(legacy_db) is False  # unchanged
    assert _overrides(legacy_db) == {"doc-live": "unmigratable"}  # the row is still there
    assert "document_meta__rebuild" not in inspect(legacy_db).get_table_names()
