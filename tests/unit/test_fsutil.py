"""Unit tests for ``fsutil.atomic_write_text`` — the shared crash-safe cache writer.

The cached ``.md`` is the source-of-truth the next ingest re-hashes, and several writers
overwrite it in place (T1: the Marker/pdfplumber table splices + the initial extraction).
A crash mid-write must leave the original intact, not a half-written file ``is_cache_fresh``
would then trust. Pure filesystem tests — no DB / Chroma / LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.fsutil import atomic_write_text


def test_writes_new_file(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    atomic_write_text(target, "hello\nworld\n")
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"


def test_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("old content", encoding="utf-8")
    atomic_write_text(target, "new content")
    assert target.read_text(encoding="utf-8") == "new content"


def test_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "note.md"
    atomic_write_text(target, "x")
    assert target.read_text(encoding="utf-8") == "x"


def test_leaves_no_temp_file_on_success(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    atomic_write_text(target, "content")
    # The write-temp-then-rename leaves only the target — no stray ``.tmp``.
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_failure_keeps_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "note.md"
    target.write_text("ORIGINAL", encoding="utf-8")

    # Simulate a crash during the atomic swap (after the temp file is fully written).
    def boom(src: object, dst: object) -> None:
        raise OSError("simulated crash during replace")

    monkeypatch.setattr("doc_assistant.fsutil.os.replace", boom)

    with pytest.raises(OSError, match="simulated crash"):
        atomic_write_text(target, "NEW — must not land")

    # The original is byte-for-byte intact (no partial write), and no temp lingers.
    assert target.read_text(encoding="utf-8") == "ORIGINAL"
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]
