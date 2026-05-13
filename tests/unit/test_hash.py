"""Tests for document hashing."""
from doc_assistant.ingest import doc_hash


def test_hash_is_deterministic():
    """Same input always produces same hash."""
    h1 = doc_hash("hello world", "/path/to/file.pdf")
    h2 = doc_hash("hello world", "/path/to/file.pdf")
    assert h1 == h2


def test_hash_changes_with_content():
    """Different content produces different hash."""
    h1 = doc_hash("hello world", "/path/to/file.pdf")
    h2 = doc_hash("goodbye world", "/path/to/file.pdf")
    assert h1 != h2


def test_hash_changes_with_path():
    """Same content at different paths produces different hash.
    
    Note: this is the path+content hashing we currently use.
    Phase 2.9 deferred work would change this to content-only.
    """
    h1 = doc_hash("hello world", "/a.pdf")
    h2 = doc_hash("hello world", "/b.pdf")
    assert h1 != h2


def test_hash_length():
    """Hash should be 16 hex chars."""
    h = doc_hash("anything", "/x.pdf")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)