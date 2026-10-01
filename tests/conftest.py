"""Suite-wide setup: the tests run on a data directory of their own.

**The data directory.** Everything the application keeps lives under one data home
(``config.DATA_PATH``): the library database, the vector stores, the extraction cache, the
session logs, the settings, the graph sidecars. On a developer's machine that is the working
library in ``data/``. This file points the whole run at a fresh, empty directory instead, by
setting ``DOC_DATA_DIR`` before ``doc_assistant.config`` is first imported. A test that reaches a
store without redirecting it lands on an empty one, as it does on a fresh checkout, whatever the
machine holds. A subprocess a test starts inherits the variable, so it lands there too.

It is not left to each test because none of this fails a test. On 2026-10-01 a full run applied a
schema migration to a working library (two route tests enter the app's lifespan, whose first act
is ``init_db()``), and every run left extraction-cache files and session logs under ``data/`` and
opened the working vector store (KI-60).

The directory is made under ``.pytest-data/`` in the repository and not in the system temp
directory, for one reason: ``config`` moves the vector stores to ``%PROGRAMDATA%`` when the data
path is not ASCII (KI-11), and the temp path of a Windows account with an accented name is not.
In the repository the path is ASCII whenever the checkout is, so everything a run creates stays
in one directory, which is removed when the run ends.

It **fails closed**. If ``doc_assistant.config`` was imported before this file could set the
variable, the run stops here rather than test against the working library.

Two smaller jobs follow, both about machine state leaking into a verdict. Each test gets its own
credential file: ADR-034 lets a user save an API key in the app, and a key saved by one test must
not flip the next test's assertion about a keyless provider. And Starlette's ``TestClient`` is let
through the API's host guard (security S-5): it sends ``Host: testserver``, which the guard
refuses by design. ``test_api_host_guard.py`` sets its own list to test the guard itself.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where each run's data directory is made. Gitignored.
DATA_HOME_ROOT = REPO_ROOT / ".pytest-data"

#: A run directory this old was left by a run that never finished (killed, or a store still open
#: at exit on Windows). The next run removes it. A day, so a session paused in a debugger is safe.
STALE_AFTER_SECONDS = 24 * 60 * 60

#: ``<pid>:<path>`` of the data directory this process made, so loading this file a second time in
#: one interpreter reuses it. A child process sees another pid and makes its own if it runs pytest.
_OWNER_VAR = "DOC_TESTS_DATA_HOME"

#: Set to anything to keep the run's data directory and print its path, to see what the tests
#: wrote: ``DOC_TESTS_KEEP_DATA_HOME=1 pytest tests/integration/test_x.py``. Remove it afterwards.
KEEP_VAR = "DOC_TESTS_KEEP_DATA_HOME"


def sweep_stale(root: Path, now: float) -> None:
    """Remove the run directories under ``root`` that are older than ``STALE_AFTER_SECONDS``."""
    for leftover in root.glob("run-*"):
        try:
            stale = now - leftover.stat().st_mtime > STALE_AFTER_SECONDS
        except OSError:
            continue
        if stale:
            shutil.rmtree(leftover, ignore_errors=True)


def _data_home() -> Path:
    """This run's empty data directory, with ``DOC_DATA_DIR`` pointing at it."""
    pid, _, made = os.environ.get(_OWNER_VAR, "").partition(":")
    if pid == str(os.getpid()) and made and Path(made).is_dir():
        return Path(made)
    DATA_HOME_ROOT.mkdir(exist_ok=True)
    sweep_stale(DATA_HOME_ROOT, time.time())
    home = DATA_HOME_ROOT / f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    home.mkdir()
    # Assigned, not defaulted: a developer who exports DOC_DATA_DIR to run the app against a
    # library elsewhere must not hand that library to the tests.
    os.environ["DOC_DATA_DIR"] = str(home)
    os.environ[_OWNER_VAR] = f"{os.getpid()}:{home}"
    return home


def _remove_data_home() -> None:
    """Remove this run's data directory. Best effort: on Windows a file still held open cannot
    be deleted, and what remains is swept by a later run once it is a day old."""
    shutil.rmtree(DATA_HOME, ignore_errors=True)
    with contextlib.suppress(OSError):
        DATA_HOME_ROOT.rmdir()  # succeeds only when no other run has a directory in it


DATA_HOME = _data_home()

# Imported only now, on purpose: `config` resolves the data directory when it is first imported.
from doc_assistant import config as app_config  # noqa: E402
from doc_assistant import credentials  # noqa: E402

if DATA_HOME.resolve() != app_config.DATA_PATH:
    _remove_data_home()
    raise RuntimeError(
        "tests/conftest.py could not give the suite its own data directory: "
        f"doc_assistant.config was imported before it and resolved {app_config.DATA_PATH}. "
        "Nothing was run. Find what imports doc_assistant ahead of the tests' conftest "
        "(a plugin, a conftest above tests/, a sitecustomize) and remove it."
    )


def _release_open_stores() -> None:
    """Close what the run left open under the data directory, so Windows lets its files go.

    The default database engine holds ``library.db``. A Chroma client is kept alive by chromadb's
    own cache of systems, and its ``chroma.sqlite3`` stays locked until the system is stopped —
    dropping the client is not enough (measured on chromadb 1.5.9). The cache is a private
    attribute, read defensively: if it moves, the files stay and a later run sweeps them."""
    session_mod = sys.modules.get("doc_assistant.db.session")
    if session_mod is not None:
        session_mod.get_engine().dispose()
    shared = sys.modules.get("chromadb.api.shared_system_client")
    if shared is not None:
        client = shared.SharedSystemClient
        for system in list(getattr(client, "_identifier_to_system", {}).values()):
            with contextlib.suppress(Exception):
                system.stop()
        client.clear_system_cache()


def pytest_report_header(config: pytest.Config) -> str:
    """Say where the run's data lives, in the header of every non-quiet run."""
    return f"data directory: {DATA_HOME} (the suite's own, removed when the run ends)"


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove the run's data directory once the session is over, unless asked to keep it."""
    _release_open_stores()
    if os.environ.get(KEEP_VAR, "").strip():
        sys.stderr.write(f"\ntests' data directory kept ({KEEP_VAR}): {DATA_HOME}\n")
        return
    _remove_data_home()


@pytest.fixture(autouse=True)
def _isolate_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the credential store at a per-test temp file, so no test sees another's key."""
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", tmp_path / "credentials.json")


@pytest.fixture(autouse=True)
def _allow_the_test_client_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve ``Host: testserver`` (the TestClient's) alongside the loopback defaults (S-5)."""
    monkeypatch.setenv("DOC_API_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver")
