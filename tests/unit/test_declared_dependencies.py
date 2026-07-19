"""Guard: every declared *runtime* dependency is actually installed in the environment.

Motivation (2026-07-19): ``send2trash>=2.1.0`` was declared in ``pyproject.toml`` (a base dep) but
absent from the venv. Because ``library.delete_document`` imports it lazily *inside* the function,
the gap was invisible until call time — ``DELETE /api/library/documents/{id}`` 500'd on every call,
and the six ``test_document_delete.py`` failures surfaced only as a cryptic ``ModuleNotFoundError``
raised from pytest's monkeypatch machinery. That cryptic shape got repeatedly misread across
sessions as "pre-existing venv drift, unrelated" rather than "a shipped feature is broken" (KI-22).

This test closes that gap: it fails *loudly, by package name*, the moment a declared base dep
is missing — so the next drift can't be mistaken for test-infra noise. It checks installation via
``importlib.metadata`` (dist metadata), NOT by importing, so it stays fast and doesn't drag in
torch / chromadb, and it needs no dist-name -> import-name mapping.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import tomllib

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _distribution_name(requirement: str) -> str:
    """The bare distribution name from a PEP 508 requirement string.

    ``uvicorn[standard]>=0.30`` -> ``uvicorn``; ``foo ; python_version < '3.13'`` -> ``foo``.
    We only need the name, so split off the first version/extra/marker/whitespace delimiter.
    """
    for i, ch in enumerate(requirement):
        if ch in "<>=!~;[(@ \t":
            return requirement[:i].strip()
    return requirement.strip()


def _declared_runtime_dependencies() -> list[str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    return [_distribution_name(d) for d in deps]


_RUNTIME_DEPS = _declared_runtime_dependencies()


def test_dependency_list_is_non_empty() -> None:
    """Sanity guard on the parser — a silently-empty list would make the check below vacuous."""
    assert "send2trash" in _RUNTIME_DEPS, _RUNTIME_DEPS


@pytest.mark.parametrize("dist_name", _RUNTIME_DEPS)
def test_declared_runtime_dependency_is_installed(dist_name: str) -> None:
    """Each base dep in pyproject.toml resolves to an installed distribution.

    ``importlib.metadata.version`` normalizes names (PEP 503), so ``rank_bm25`` / ``python-docx`` /
    ``beautifulsoup4`` all resolve without a hand-maintained alias table. A miss means the venv
    does not satisfy the declared contract — run ``just sync``.
    """
    try:
        version(dist_name)
    except PackageNotFoundError:
        pytest.fail(
            f"declared runtime dependency {dist_name!r} is not installed in this environment "
            f"(pyproject.toml [project].dependencies). The venv does not satisfy the declared "
            f"contract — missing-dependency drift, not a test-infra flake. Fix: `just sync`."
        )


def test_send2trash_import_form_used_by_delete_document() -> None:
    """Pin the exact import ``delete_document`` does: ``from send2trash import send2trash``.

    Regression guard for KI-22: the six safe-delete tests monkeypatch ``send2trash.send2trash``,
    so a broken *real* import can hide behind the patch. This asserts the callable is importable.
    """
    from send2trash import send2trash

    assert callable(send2trash)
