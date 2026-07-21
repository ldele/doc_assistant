"""Database schema creation and migrations.

``create_all`` is the migration for a *new table* (additive, idempotent — this
is how the `figures` table landed). It does NOT add a *new column* to a table
that already exists, so an additive column on a pre-existing table needs an
explicit ``ALTER TABLE`` here (SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we
check the live schema first). Keep this list append-only.

**Rebuild migrations (ADR-026).** SQLite cannot ``ALTER`` a *constraint*, so a table whose shape
must change — a missing foreign key, a widened primary key — is rebuilt and swapped instead:
create the correct table from the model, copy the rows worth keeping, drop, rename. That is a
strictly bigger hammer than ``ADD COLUMN``, so it is rare by policy: each one is a named function
below, is idempotent (it inspects the live schema and no-ops when already correct), and must be
justified in an ADR. See ``_rebuild_document_meta_fk``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import structlog
from sqlalchemy import MetaData, Table, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable

from doc_assistant.config import SQLITE_PATH
from doc_assistant.db.models import Base, DocumentMeta
from doc_assistant.db.session import get_engine

log = structlog.get_logger(__name__)

# Additive columns on PRE-EXISTING tables: (table, column, column_ddl, index_or_None).
# create_all() never alters an existing table, so each is applied by hand below,
# guarded by a live-schema check so re-running is a no-op. Append-only.
_ADDITIVE_COLUMNS: list[tuple[str, str, str, str | None]] = [
    # Chunk 2c — categorical failure tag on the (pre-existing) reviewer table.
    ("answer_reviews", "failure_tag", "VARCHAR", "ix_answer_reviews_failure_tag"),
    # Glossary — curated definition gloss on the (pre-existing) concepts table.
    ("concepts", "definition", "TEXT", None),
    # R4 — graded per-token provenance strength on the (pre-existing) concept_edges table.
    ("concept_edges", "strength_json", "TEXT", None),
    # Conversation rename — user-set title on the (pre-existing) conversation_meta table.
    ("conversation_meta", "title_override", "TEXT", None),
    # ADR-018 — graph-vocabulary opt-in on the (pre-existing) concepts table. Lands NULL
    # on every existing row, which reads as excluded; scripts/backfill_graph_include.py
    # sets the policy (source == "manual" opts in). Indexed: load_concepts filters on it
    # on every skeleton build.
    ("concepts", "graph_include", "BOOLEAN", "ix_concepts_graph_include"),
    # ADR-025 F2 — the folder retrieval scope a turn ran under, on the (pre-existing)
    # answer_records table. NULL on every existing row, which reads correctly as "unscoped".
    ("answer_records", "retrieval_scope_json", "TEXT", None),
    # E1.1 (KI-8) — the segmentation-agnostic marker join key on the (regenerable)
    # chunk_epistemics table. NULL on existing rows, which read_epistemics_index falls back to
    # `{document_id}:{chunk_index}`; the next `compute_epistemics --apply` fills it (incl. the new
    # `{doc}:p{parent_index}` parent rows). Indexed: the live marker join is a per-turn lookup.
    ("chunk_epistemics", "chunk_key", "VARCHAR", "ix_chunk_epistemics_chunk_key"),
]


def _apply_additive_columns(engine: Engine) -> list[str]:
    """Idempotently add any missing additive columns (+ their index) to existing tables.

    Returns the ``table.column`` names actually added, so a caller can say what it changed
    rather than leaving a schema drift to be discovered at the first failed write (KI-23).
    """
    added: list[str] = []
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, column, ddl, index in _ADDITIVE_COLUMNS:
            if table not in tables:
                continue  # create_all will have made it with the column already present
            columns = {c["name"] for c in inspector.get_columns(table)}
            if column in columns:
                continue
            # Identifiers are hardcoded constants in _ADDITIVE_COLUMNS, never user input.
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))  # nosec B608
            if index:
                conn.execute(
                    text(f"CREATE INDEX IF NOT EXISTS {index} ON {table} ({column})")  # nosec B608
                )
            log.info("added_column", table=table, column=column)
            added.append(f"{table}.{column}")
    return added


def _rebuild_table(engine: Engine, model_table: Table, *, keep_where: str | None = None) -> None:
    """Replace a live table with the model's version of it, carrying the rows over.

    The SQLite-documented table-rebuild dance: foreign keys off, one transaction, create the new
    shape under a temp name, copy, drop, rename, ``foreign_key_check``, commit. ``keep_where`` is a
    SQL predicate selecting the rows to carry (rows that would violate the new constraints must be
    left behind, or the rebuilt table would be lying about them).

    Only columns present in **both** the live table and the model are copied, so a rebuild is safe
    to run against a schema that predates an additive column. Executed in ``AUTOCOMMIT`` because
    ``PRAGMA foreign_keys`` is a no-op inside a transaction, with an explicit ``BEGIN``/``COMMIT``
    around the destructive middle so a crash cannot leave the table dropped-but-not-renamed.
    """
    name = model_table.name
    temp = f"{name}__rebuild"
    live_columns = {c["name"] for c in inspect(engine).get_columns(name)}
    copied = [c.name for c in model_table.columns if c.name in live_columns]

    # Render the new shape from the model. The staging MetaData needs the tables this one points
    # at, or the FK clause cannot resolve a target at compile time — only the temp table's DDL is
    # ever executed, so the copies are render-time scaffolding, nothing more.
    staging = MetaData()
    for fk in model_table.foreign_keys:
        referenced = fk.column.table
        if referenced.name != name and referenced.name not in staging.tables:
            referenced.to_metadata(staging)
    new_table = model_table.to_metadata(staging, name=temp)
    create_sql = str(CreateTable(new_table).compile(engine))
    columns_sql = ", ".join(copied)
    where_sql = f" WHERE {keep_where}" if keep_where else ""

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        try:
            conn.execute(text("BEGIN"))
            # Belt and braces: the whole swap is transactional, so a leftover temp table
            # should be impossible — but if one ever survived, every future run would wedge.
            conn.execute(text(f"DROP TABLE IF EXISTS {temp}"))  # nosec B608
            conn.execute(text(create_sql))
            # Identifiers come from the SQLAlchemy model and this module's own constants,
            # never from user input.
            copy_sql = (
                f"INSERT INTO {temp} ({columns_sql}) "  # nosec B608
                f"SELECT {columns_sql} FROM {name}{where_sql}"
            )
            conn.execute(text(copy_sql))
            conn.execute(text(f"DROP TABLE {name}"))  # nosec B608
            conn.execute(text(f"ALTER TABLE {temp} RENAME TO {name}"))  # nosec B608
            violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
            if violations:
                raise RuntimeError(f"rebuilt {name} still violates a foreign key: {violations!r}")
            conn.execute(text("COMMIT"))
        except Exception:
            conn.execute(text("ROLLBACK"))
            raise
        finally:
            conn.execute(text("PRAGMA foreign_keys=ON"))


def _rebuild_document_meta_fk(engine: Engine) -> str | None:
    """Give ``document_meta.document_id`` the foreign key it shipped without (ADR-026).

    Without it an override outlived its document: the orphan sweep and the old ``--rebuild`` bulk
    delete (KI-24) removed ``Document`` rows without touching this table, leaving rows nothing
    could read and nothing would clean. The FK makes that structurally impossible.

    Already-orphaned rows cannot be carried over — they are exactly what the constraint forbids,
    and they cannot be rescued either (an override records no filename or hash, only a dead id).
    They are dropped, and **logged in full first** so the migration never removes something a user
    typed without leaving a trace. Returns a one-line description of what changed, or None when
    the schema was already correct.
    """
    inspector = inspect(engine)
    if "document_meta" not in set(inspector.get_table_names()):
        return None  # create_all just made it, with the FK
    keys = inspector.get_foreign_keys("document_meta")
    if any(fk.get("referred_table") == "documents" for fk in keys):
        return None

    with engine.connect() as conn:
        orphans: Sequence[Any] = conn.execute(
            text(
                "SELECT document_id, title_override, authors_override, year_override "
                "FROM document_meta "
                "WHERE document_id NOT IN (SELECT id FROM documents)"
            )
        ).fetchall()
    if orphans:
        log.warning(
            "dropping_orphaned_document_meta",
            count=len(orphans),
            rows=[dict(r._mapping) for r in orphans[:20]],
            hint="metadata overrides whose document no longer exists; unreadable already, "
            "and unrescuable (an override records no filename or hash). Logged here in full.",
        )

    _rebuild_table(
        engine,
        cast(Table, DocumentMeta.__table__),
        keep_where="document_id IN (SELECT id FROM documents)",
    )
    log.info(
        "rebuilt_table", table="document_meta", added="FK document_id -> documents.id CASCADE"
    )
    return f"document_meta.document_id FK (+{len(orphans)} orphan row(s) dropped)"


def init_db(reset: bool = False) -> list[str]:
    """Create all tables + apply additive column migrations. Safe to run repeatedly.

    Returns the additive columns added by this call (empty when the schema was already
    current), so an entrypoint can log a schema change instead of silently drifting (KI-23).

    Args:
        reset: If True, drops all tables first. WARNING: destroys data.
    """
    db_path = Path(SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine()

    if reset:
        log.warning("resetting_database", path=str(db_path))
        Base.metadata.drop_all(engine)

    log.info("creating_tables", path=str(db_path))
    Base.metadata.create_all(engine)
    added = _apply_additive_columns(engine)

    # Rebuild migrations run last: they replace a whole table, so they must see the schema the
    # additive pass has already brought up to date (ADR-026).
    rebuilt = _rebuild_document_meta_fk(engine)
    if rebuilt is not None:
        added.append(rebuilt)

    # Verify
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    log.info("tables_present", tables=sorted(tables))
    return added


if __name__ == "__main__":
    import argparse

    # Program entrypoint (`python -m doc_assistant.db.migrations`): configure logging
    # here so the table-creation events are visible (ADR-003). Library imports never do.
    from doc_assistant.config import LOG_JSON, LOG_LEVEL
    from doc_assistant.logging_config import configure_logging

    configure_logging(json=LOG_JSON, level=LOG_LEVEL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="Drop all tables before creating. Destroys data."
    )
    args = parser.parse_args()
    init_db(reset=args.reset)
