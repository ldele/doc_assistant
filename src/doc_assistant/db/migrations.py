"""Database schema creation and migrations.

``create_all`` is the migration for a *new table* (additive, idempotent — this
is how the `figures` table landed). It does NOT add a *new column* to a table
that already exists, so an additive column on a pre-existing table needs an
explicit ``ALTER TABLE`` here (SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we
check the live schema first). Keep this list append-only.
"""

from pathlib import Path

import structlog
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from doc_assistant.config import SQLITE_PATH
from doc_assistant.db.models import Base
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
]


def _apply_additive_columns(engine: Engine) -> None:
    """Idempotently add any missing additive columns (+ their index) to existing tables."""
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


def init_db(reset: bool = False) -> None:
    """Create all tables + apply additive column migrations. Safe to run repeatedly.

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
    _apply_additive_columns(engine)

    # Verify
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    log.info("tables_present", tables=sorted(tables))


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
