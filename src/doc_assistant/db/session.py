"""SQLAlchemy engine and session management."""
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from doc_assistant.config import SQLITE_URL, SQLITE_PATH


# Engine — created once at import time
_engine = create_engine(
    SQLITE_URL,
    echo=False,  # set True to log all SQL (verbose)
    future=True,
)

# Session factory
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


# Enable foreign key constraints on every connection.
# SQLite has FKs off by default; we have to enable them per-connection.
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> Engine:
    return _engine


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as session:
            doc = Document(...)
            session.add(doc)
            # commit happens automatically on success
            # rollback happens automatically on exception
    """
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Return a new session. Caller is responsible for committing and closing.

    Prefer session_scope() when possible.
    """
    return _SessionLocal()