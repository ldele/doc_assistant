"""*Reveal in folder* runs Windows' own file manager by its full path (S-10, S11).

A bare ``explorer`` is looked up on PATH, so a program of that name earlier on PATH would run
with the document's path instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from doc_assistant.library import documents


def test_explorer_is_named_by_its_full_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WINDIR", r"D:\Win")
    assert documents._explorer() == Path(r"D:\Win") / "explorer.exe"
    monkeypatch.delenv("WINDIR")
    monkeypatch.setenv("SYSTEMROOT", r"E:\Sys")
    assert documents._explorer() == Path(r"E:\Sys") / "explorer.exe"


def test_reveal_runs_that_path_and_no_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(documents.sys, "platform", "win32")
    monkeypatch.setenv("WINDIR", r"C:\Windows")
    monkeypatch.setattr(documents.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    documents._reveal_in_file_manager(Path(r"C:\lib\paper.pdf"))
    [(args, kw)] = calls
    assert args[0] == str(Path(r"C:\Windows") / "explorer.exe")
    assert args[1] == r"/select,C:\lib\paper.pdf"
    assert not kw.get("shell")
