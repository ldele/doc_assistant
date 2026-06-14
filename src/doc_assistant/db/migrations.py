"""Database schema creation and migrations.

``create_all`` is the migration for a *new table* (additive, idempotent — this
is how the `figures` table landed). It does NOT add a *new column* to a table
that already exists, so an additive column on a pre-existing table needs an
explicit ``ALTER TABLE`` here (SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we
check the live schema first). Keep this list append-only.
"""

from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from doc_assistant.config import SQLITE_PATH
from doc_assistant.db.models import Base
from doc_assistant.db.session import get_engine

# Additive columns on PRE-EXISTING tables: (table, column, column_ddl, index_or_None).
# create_all() never alters an existing table, so each is applied by hand below,
# guarded by a live-schema check so re-running is a no-op. Append-only.
_ADDITIVE_COLUMNS: list[tuple[str, str, str, str | None]] = [
    # Chunk 2c — categorical failure tag on the (pre-existing) reviewer table.
    ("answer_reviews", "failure_tag", "VARCHAR", "ix_answer_reviews_failure_tag"),
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
            print(f"  + added column {table}.{column}")


def init_db(reset: bool = False) -> None:
    """Create all tables + apply additive column migrations. Safe to run repeatedly.

    Args:
        reset: If True, drops all tables first. WARNING: destroys data.
    """
    db_path = Path(SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine()

    if reset:
        print(f"Resetting database at {db_path}")
        Base.metadata.drop_all(engine)

    print(f"Creating tables in {db_path}")
    Base.metadata.create_all(engine)
    _apply_additive_columns(engine)

    # Verify
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables present: {sorted(tables)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="Drop all tables before creating. Destroys data."
    )
    args = parser.parse_args()
    init_db(reset=args.reset)
