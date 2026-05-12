"""Database schema creation and migrations.

For Phase 3, we just have schema creation. As the schema evolves in later 
phases, this module grows to include actual migrations.
"""
from pathlib import Path

from doc_assistant.config import SQLITE_PATH
from doc_assistant.db.models import Base
from doc_assistant.db.session import get_engine


def init_db(reset: bool = False) -> None:
    """Create all tables. Safe to run multiple times.

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

    # Verify
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables present: {sorted(tables)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Drop all tables before creating. Destroys data.")
    args = parser.parse_args()
    init_db(reset=args.reset)