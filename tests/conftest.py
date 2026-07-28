"""Suite-wide fixtures.

One job today: keep the **real install's credential file** out of the test run. ADR-034 lets a user
save an API key in the app, which lands at ``<data home>/credentials.json`` — the same data home
the test process resolves. Without this fixture, saving a key in the desktop app would silently
flip every assertion about a keyless provider (``provider_available`` reads the store as well as
the env), and the suite's verdict would depend on machine state. Autouse + tmp_path is the whole
fix: every test resolves credentials against an empty directory unless it writes one itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant import credentials


@pytest.fixture(autouse=True)
def _isolate_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the credential store at a per-test temp file (never the user's real one)."""
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", tmp_path / "credentials.json")
