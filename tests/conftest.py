"""Suite-wide fixtures.

Three jobs. First, keep the **real install's credential file** out of the test run. ADR-034 lets a
user save an API key in the app, which lands at ``<data home>/credentials.json`` — the same data
home the test process resolves. Without this fixture, saving a key in the desktop app would
silently flip every assertion about a keyless provider (``provider_available`` reads the store as
well as the env), and the suite's verdict would depend on machine state. Autouse + tmp_path is the
whole fix: every test resolves credentials against an empty directory unless it writes one itself.

Second, let Starlette's ``TestClient`` through the API's host guard (security S-5): it sends
``Host: testserver``, which the guard refuses by design. ``test_api_host_guard.py`` sets its own
list to test the guard itself.

Third, keep the **working library's database** out of the test run. ``db.session`` binds its
engine to the configured ``library.db`` at import, so a test that reaches the database without
swapping the engine reaches the developer's own library. Two route test files enter the app's
lifespan that way, and the lifespan's first act is the schema migration: on 2026-10-01 a full run
added a new column to a working library (KI-60). The session fixture below binds the default
engine to a throwaway file instead, so an unredirected test lands where it would on a fresh
checkout — an empty database — whatever the machine holds.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import credentials


@pytest.fixture(scope="session", autouse=True)
def _keep_the_working_library_out_of_the_suite(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Bind the database layer's default engine to a throwaway file for the whole run.

    One file for the session, not one per test: it stands where the configured database stood,
    which the unredirected tests already shared. A test that swaps the engine itself (most do)
    saves and restores this one."""
    scratch = tmp_path_factory.mktemp("unredirected-db") / "library.db"
    engine = create_engine(
        f"sqlite:///{scratch}",
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},  # as db.session: handlers run in threads
    )
    working = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    try:
        yield
    finally:
        session_mod._engine, session_mod._SessionLocal = working
        engine.dispose()


@pytest.fixture(autouse=True)
def _isolate_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the credential store at a per-test temp file (never the user's real one)."""
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", tmp_path / "credentials.json")


@pytest.fixture(autouse=True)
def _allow_the_test_client_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve ``Host: testserver`` (the TestClient's) alongside the loopback defaults (S-5)."""
    monkeypatch.setenv("DOC_API_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver")
