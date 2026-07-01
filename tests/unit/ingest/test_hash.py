"""Tests for document hashing."""

from doc_assistant.ingest import doc_hash


def test_hash_is_deterministic():
    """Same input always produces same hash."""
    h1 = doc_hash("hello world")
    h2 = doc_hash("hello world")
    assert h1 == h2


def test_hash_changes_with_content():
    """Different content produces different hash."""
    h1 = doc_hash("hello world")
    h2 = doc_hash("goodbye world")
    assert h1 != h2


def test_hash_is_path_independent():
    """Same content at different paths produces the SAME hash.

    Content-only hashing: documents survive moves/renames without
    creating orphan rows in SQLite or Chroma.
    """
    h1 = doc_hash("hello world")
    h2 = doc_hash("hello world")
    assert h1 == h2


def test_hash_length():
    """Hash should be 16 hex chars."""
    h = doc_hash("anything")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
