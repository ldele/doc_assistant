"""Suite-wide fixtures.

Two jobs. First, keep the **real install's credential file** out of the test run. ADR-034 lets a
user save an API key in the app, which lands at ``<data home>/credentials.json`` — the same data
home the test process resolves. Without this fixture, saving a key in the desktop app would
silently flip every assertion about a keyless provider (``provider_available`` reads the store as
well as the env), and the suite's verdict would depend on machine state. Autouse + tmp_path is the
whole fix: every test resolves credentials against an empty directory unless it writes one itself.

Second, let Starlette's ``TestClient`` through the API's host guard (security S-5): it sends
``Host: testserver``, which the guard refuses by design. ``test_api_host_guard.py`` sets its own
list to test the guard itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant import credentials


@pytest.fixture(autouse=True)
def _isolate_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the credential store at a per-test temp file (never the user's real one)."""
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", tmp_path / "credentials.json")


@pytest.fixture(autouse=True)
def _allow_the_test_client_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve ``Host: testserver`` (the TestClient's) alongside the loopback defaults (S-5)."""
    monkeypatch.setenv("DOC_API_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver")
