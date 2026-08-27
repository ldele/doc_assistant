"""ADR-046 / AD3b — ``source_files`` is re-keyed on ``(root_id, rel_path)``.

The project's **second** rebuild migration (create → copy → drop → rename), and the first one to
carry data users cannot regenerate: `source_files` holds the ``excluded`` flags they set by hand.
SQLite refuses to ``ALTER`` a ``REFERENCES`` column onto a live table under
``PRAGMA foreign_keys=ON``, so the whole table is rebuilt — and a rebuild that goes wrong is not
recoverable from inside the app.

It shipped with **no test at all**, which is how it was found: the review's pre-merge check ran it
by hand against `data/library.db.bak-20260824-preroots`, a real 97-document pre-AD3b database. It
passed, but a one-off run proves the migration works *once*, on *one* machine. These tests build
the genuinely pre-AD3b table — verbatim the ``CREATE TABLE`` that database holds, old ``UNIQUE`` on
``rel_path`` and all — and drive ``init_db`` over it.

The sharpest assertion here is `test_an_index_added_by_an_earlier_migration_survives_the_rebuild`.
``CreateTable`` renders the table only, and the old indexes die with the dropped table, so before
this branch a rebuild returned a correctly-shaped but **unindexed** table. It was latent for
ADR-026 (``document_meta`` declares no index) and stopped being latent here, because AD2's
``ix_source_files_source_sha256`` is exactly such an index.

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
from doc_assistant.db.models import Base

# Verbatim from `data/library.db.bak-20260824-preroots`: no `root_id`, no FK, and `rel_path`
# unique on its own. `source_sha256` and `origin` are present because AD2 landed before AD3b —
# this is the real intermediate state a user upgrading actually passes through.
_LEGACY_DDL = """
CREATE TABLE source_files (
    id VARCHAR NOT NULL,
    rel_path VARCHAR NOT NULL,
    format VARCHAR NOT NULL,
    size INTEGER NOT NULL,
    mtime FLOAT NOT NULL,
    doc_type VARCHAR,
    excluded BOOLEAN NOT NULL,
    first_seen DATETIME NOT NULL,
    last_seen DATETIME NOT NULL,
    source_sha256 VARCHAR,
    origin VARCHAR NOT NULL DEFAULT 'copied',
    PRIMARY KEY (id)
)
"""
_LEGACY_INDEXES = (
    "CREATE UNIQUE INDEX ix_source_files_rel_path ON source_files (rel_path)",
    "CREATE INDEX ix_source_files_source_sha256 ON source_files (source_sha256)",
)

_ROWS = [
    ("s1", "papers/rag.pdf", "pdf", 100, 1.0, 0, "abc123"),
    ("s2", "papers/bm25.pdf", "pdf", 200, 2.0, 1, None),  # excluded — a user's own decision
    ("s3", "notes/reading.md", "md", 300, 3.0, 0, "def456"),
]


@pytest.fixture
def legacy_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """A database whose ``source_files`` predates the multi-root key."""
    db_path = tmp_path / "library.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE source_files"))
        conn.execute(text("DROP TABLE source_roots"))
        conn.execute(text(_LEGACY_DDL))
        for ddl in _LEGACY_INDEXES:
            conn.execute(text(ddl))
        for row in _ROWS:
            conn.execute(
                text(
                    "INSERT INTO source_files (id, rel_path, format, size, mtime, excluded,"
                    " source_sha256, origin, first_seen, last_seen) VALUES (:i, :r, :f, :s, :m,"
                    " :e, :h, 'copied', '2026-01-01', '2026-01-01')"
                ),
                {
                    "i": row[0],
                    "r": row[1],
                    "f": row[2],
                    "s": row[3],
                    "m": row[4],
                    "e": row[5],
                    "h": row[6],
                },
            )

    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    monkeypatch.setattr("doc_assistant.db.migrations.get_engine", lambda: engine)
    monkeypatch.setattr("doc_assistant.db.migrations.SQLITE_PATH", str(db_path))
    yield engine
    session_mod._engine, session_mod._SessionLocal = orig_engine, orig_factory
    engine.dispose()


def _indexes(engine: Engine) -> dict[str, tuple[bool, list[str]]]:
    with engine.connect() as conn:
        out: dict[str, tuple[bool, list[str]]] = {}
        for row in conn.execute(text("PRAGMA index_list(source_files)")):
            name, unique = row[1], bool(row[2])
            members = [r[2] for r in conn.execute(text(f"PRAGMA index_info('{name}')"))]
            out[name] = (unique, members)
        return out


def test_the_legacy_table_really_is_pre_ad3b(legacy_db: Engine) -> None:
    """Guard the fixture: had this table already carried `root_id`, every test below is vacuous."""
    cols = {c["name"] for c in inspect(legacy_db).get_columns("source_files")}
    assert "root_id" not in cols
    assert _indexes(legacy_db)["ix_source_files_rel_path"][0] is True, "rel_path was unique alone"


def test_every_row_survives_and_is_backfilled_to_the_library_root(legacy_db: Engine) -> None:
    """The rebuild copies data the user cannot regenerate — `excluded` is their own decision."""
    init_db()

    with legacy_db.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, rel_path, root_id, excluded, source_sha256 FROM source_files"
                " ORDER BY id"
            )
        ).fetchall()
    assert [r[0] for r in rows] == ["s1", "s2", "s3"], "a row was lost in the rebuild"
    assert {r[2] for r in rows} == {"library"}, "root_id was not backfilled"
    assert [bool(r[3]) for r in rows] == [False, True, False], "an exclusion was lost"
    assert [r[4] for r in rows] == ["abc123", None, "def456"], "a cached hash was lost"


def test_the_key_becomes_the_pair_and_the_old_one_goes(legacy_db: Engine) -> None:
    """The whole point: the same `papers/rag.pdf` may now exist under two roots."""
    init_db()
    idx = _indexes(legacy_db)

    composite = [n for n, (u, m) in idx.items() if u and sorted(m) == ["rel_path", "root_id"]]
    assert composite, f"no UNIQUE index on (root_id, rel_path): {idx}"
    still_unique_alone = [n for n, (u, m) in idx.items() if u and m == ["rel_path"]]
    assert not still_unique_alone, f"the old rel_path UNIQUE survived: {still_unique_alone}"


def test_an_index_added_by_an_earlier_migration_survives_the_rebuild(legacy_db: Engine) -> None:
    """`CreateTable` renders the table only, and the dropped table takes its indexes with it.

    Latent for ADR-026 (`document_meta` declares none) and not latent here: AD2's additive
    migration created `ix_source_files_source_sha256` on live databases, and every duplicate
    lookup in `library/add.py` reads through it. Without the rebuild recreating the model's
    indexes, this migration silently returned a correctly-shaped but unindexed table.
    """
    assert "ix_source_files_source_sha256" in _indexes(legacy_db), "fixture precondition"
    init_db()
    assert "ix_source_files_source_sha256" in _indexes(legacy_db), (
        "the AD2 index was dropped with the old table and never recreated"
    )


def test_the_new_key_and_the_foreign_key_are_actually_enforced(legacy_db: Engine) -> None:
    """A constraint that exists in the schema but is not enforced is decoration."""
    init_db()

    with pytest.raises(IntegrityError), legacy_db.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(
            text(
                "INSERT INTO source_files (id, root_id, rel_path, format, size, mtime,"
                " excluded, origin, first_seen, last_seen) VALUES ('dup', 'library',"
                " 'papers/rag.pdf', 'pdf', 1, 1.0, 0, 'copied', '2026-01-01', '2026-01-01')"
            )
        )

    with pytest.raises(IntegrityError), legacy_db.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(
            text(
                "INSERT INTO source_files (id, root_id, rel_path, format, size, mtime,"
                " excluded, origin, first_seen, last_seen) VALUES ('fk', 'no-such-root',"
                " 'x.pdf', 'pdf', 1, 1.0, 0, 'copied', '2026-01-01', '2026-01-01')"
            )
        )


def test_the_same_rel_path_under_two_roots_is_now_legal(legacy_db: Engine) -> None:
    """The complement of the test above — the migration exists to *allow* this."""
    init_db()

    with legacy_db.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(
            text(
                "INSERT INTO source_roots (id, path, kind, added_at) VALUES"
                " ('zotero', 'C:/zotero', 'referenced', '2026-01-01')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO source_files (id, root_id, rel_path, format, size, mtime, excluded,"
                " origin, first_seen, last_seen) VALUES ('z1', 'zotero', 'papers/rag.pdf', 'pdf',"
                " 1, 1.0, 0, 'referenced', '2026-01-01', '2026-01-01')"
            )
        )
    with legacy_db.connect() as conn:
        n = conn.execute(
            text("SELECT count(*) FROM source_files WHERE rel_path = 'papers/rag.pdf'")
        ).scalar_one()
    assert n == 2


def test_the_migration_is_idempotent(legacy_db: Engine) -> None:
    """It runs on every `init_db` — every app start. A second pass must change nothing."""
    init_db()
    first_idx, first_applied = _indexes(legacy_db), init_db()

    assert not [a for a in first_applied if "source_files.root_id" in a], (
        "the second run re-reported the migration as applied"
    )
    assert _indexes(legacy_db) == first_idx
    with legacy_db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM source_files")).scalar_one() == 3
        assert conn.execute(text("SELECT count(*) FROM source_roots")).scalar_one() == 1


def test_no_foreign_key_is_left_violated(legacy_db: Engine) -> None:
    """`_rebuild_table` checks this itself and raises — assert the end state, don't trust it."""
    init_db()
    with legacy_db.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_key_check")).fetchall() == []
