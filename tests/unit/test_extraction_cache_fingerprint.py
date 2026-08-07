"""The extraction cache invalidates when the EXTRACTOR changes, not just the source file (KI-40).

Freshness used to compare mtimes only, so every extraction improvement this project shipped was
invisible to anyone who had already ingested — KI-14, KI-29 and the 2026-08-07 text-layer fallback
all had that hole. These pin the fix and, more importantly, pin the *reasons* each component of the
fingerprint is in it: each one below corresponds to a way the cache could otherwise go quietly
stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.ingest import cache as cache_mod
from doc_assistant.ingest.cache import (
    _fingerprint_path,
    extraction_fingerprint,
    is_cache_fresh,
)


@pytest.fixture(autouse=True)
def _clear_fingerprint_cache():
    """`extraction_fingerprint` is lru_cached for the process; these tests move its inputs."""
    extraction_fingerprint.cache_clear()
    yield
    extraction_fingerprint.cache_clear()


def _write_pair(tmp_path: Path, *, fingerprint: str | None) -> tuple[Path, Path]:
    """A source file and a cache entry newer than it, optionally fingerprinted."""
    original = tmp_path / "paper.pdf"
    original.write_bytes(b"%PDF-1.4 fake")
    cached = tmp_path / "paper.md"
    cached.write_text("# extracted", encoding="utf-8")
    if fingerprint is not None:
        _fingerprint_path(cached).write_text(fingerprint, encoding="utf-8")
    return original, cached


def test_matching_fingerprint_is_fresh(tmp_path: Path) -> None:
    original, cached = _write_pair(tmp_path, fingerprint=extraction_fingerprint())
    assert is_cache_fresh(original, cached) is True


def test_a_cache_with_no_fingerprint_is_stale(tmp_path: Path) -> None:
    """Caches written before this existed hold output no current version would produce.

    Treating them as fresh would preserve exactly the bug KI-40 describes — the upgrade would be
    silently inert for every existing library."""
    original, cached = _write_pair(tmp_path, fingerprint=None)
    assert is_cache_fresh(original, cached) is False


def test_a_different_fingerprint_is_stale(tmp_path: Path) -> None:
    original, cached = _write_pair(tmp_path, fingerprint="0000000000000000")
    assert is_cache_fresh(original, cached) is False


def test_source_newer_than_cache_still_wins(tmp_path: Path) -> None:
    """The original mtime rule must survive: a fingerprint match cannot vouch for stale content."""
    original, cached = _write_pair(tmp_path, fingerprint=extraction_fingerprint())
    import os
    import time

    future = time.time() + 60
    os.utime(original, (future, future))
    assert is_cache_fresh(original, cached) is False


def test_changing_a_tunable_changes_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module-level constants are referenced by NAME, so their values never reach `co_code`.

    `_TEXT_LAYER_KEPT_MIN` is exactly such a knob and it changes extraction output, so it has to be
    hashed explicitly — bytecode hashing alone would miss it."""
    from doc_assistant import extractors

    before = extraction_fingerprint()
    extraction_fingerprint.cache_clear()
    monkeypatch.setattr(extractors, "_TEXT_LAYER_KEPT_MIN", 0.9)
    assert extraction_fingerprint() != before


def test_changing_the_selected_extractor_changes_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = extraction_fingerprint()
    extraction_fingerprint.cache_clear()
    monkeypatch.setattr(cache_mod.config, "PDF_EXTRACTOR", "something-else")
    assert extraction_fingerprint() != before


def test_changing_extractor_logic_changes_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bump-free half: editing an extractor must invalidate without touching a constant."""
    from doc_assistant import extractors

    before = extraction_fingerprint()
    extraction_fingerprint.cache_clear()

    def _different(md: str) -> str:
        return md + "changed"

    monkeypatch.setattr(extractors, "strip_image_placeholders", _different)
    assert extraction_fingerprint() != before


def test_fingerprint_is_stable_across_calls() -> None:
    """It keys a cache; an unstable value would re-extract the library on every run."""
    a = extraction_fingerprint()
    extraction_fingerprint.cache_clear()
    assert extraction_fingerprint() == a


def test_fingerprint_path_is_a_sibling_not_a_header(tmp_path: Path) -> None:
    """The .md bytes ARE the document text and are hashed into doc_hash — a header inside it would
    change every document's identity."""
    cached = tmp_path / "paper.md"
    fp = _fingerprint_path(cached)
    assert fp.parent == cached.parent
    assert fp.name == "paper.md.fp"
