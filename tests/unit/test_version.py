"""`doc_assistant.__version__` must not drift from `pyproject.toml` (ADR-044).

The update check compares the running version against the newest published release. If the
constant the app reports is stale, the comparison is against a lie — the app would keep offering
an "update" it already is, or stay quiet about one it is not. `release_preflight` checks all six
places at release time; this test catches the drift at commit time, which is where it starts.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

from doc_assistant import __version__

ROOT = Path(__file__).resolve().parents[2]


def test_version_constant_matches_pyproject() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == pyproject["project"]["version"], (
        "src/doc_assistant/__init__.py and pyproject.toml disagree — bump both "
        "(docs/RELEASE.md §1 lists all six places)"
    )
