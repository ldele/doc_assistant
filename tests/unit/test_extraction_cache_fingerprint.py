"""The extraction cache invalidates when the EXTRACTOR changes, not just the source file (KI-40).

Freshness used to compare mtimes only, so every extraction improvement this project shipped was
invisible to anyone who had already ingested — KI-14, KI-29 and the 2026-08-07 text-layer fallback
all had that hole. These pin the fix and, more importantly, pin the *reasons* each component of the
fingerprint is in it: each one below corresponds to a way the cache could otherwise go quietly
stale.
"""

from __future__ import annotations

import os
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
    original, cached = _write_pair(tmp_path, fingerprint=extraction_fingerprint(".pdf"))
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
    original, cached = _write_pair(tmp_path, fingerprint=extraction_fingerprint(".pdf"))
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

    before = extraction_fingerprint(".pdf")
    extraction_fingerprint.cache_clear()
    monkeypatch.setattr(extractors, "_TEXT_LAYER_KEPT_MIN", 0.9)
    assert extraction_fingerprint(".pdf") != before


def test_changing_the_selected_extractor_changes_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = extraction_fingerprint(".pdf")
    extraction_fingerprint.cache_clear()
    monkeypatch.setattr(cache_mod.config, "PDF_EXTRACTOR", "something-else")
    assert extraction_fingerprint(".pdf") != before


def test_changing_extractor_logic_changes_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bump-free half: editing an extractor must invalidate without touching a constant."""
    from doc_assistant import extractors

    before = extraction_fingerprint(".pdf")
    extraction_fingerprint.cache_clear()

    def _different(md: str) -> str:
        return md + "changed"

    monkeypatch.setattr(extractors, "strip_image_placeholders", _different)
    assert extraction_fingerprint(".pdf") != before


def test_fingerprint_is_stable_across_calls() -> None:
    """It keys a cache; an unstable value would re-extract the library on every run."""
    a = extraction_fingerprint(".pdf")
    extraction_fingerprint.cache_clear()
    assert extraction_fingerprint(".pdf") == a


def test_fingerprint_path_is_a_sibling_not_a_header(tmp_path: Path) -> None:
    """The .md bytes ARE the document text and are hashed into doc_hash — a header inside it would
    change every document's identity."""
    cached = tmp_path / "paper.md"
    fp = _fingerprint_path(cached)
    assert fp.parent == cached.parent
    assert fp.name == "paper.md.fp"


# ============================================================
# Cache paths for referenced files (ADR-046, AD3b).
# ============================================================


def test_a_library_file_keeps_the_mirror_cache_layout(tmp_path, monkeypatch):
    """The path every already-extracted document depends on. Changing it re-extracts the corpus."""
    from doc_assistant import config
    from doc_assistant.ingest.cache import get_cache_path

    monkeypatch.setattr(config, "DOCS_PATH", tmp_path / "sources")
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")

    got = get_cache_path(tmp_path / "sources" / "a" / "b.pdf")
    assert got == tmp_path / "cache" / "a" / "b.md"


def test_a_referenced_file_outside_the_library_still_resolves(tmp_path, monkeypatch):
    """AD3b regression guard: this raised ValueError, so a referenced file could not be ingested.

    `get_cache_path` is called unguarded on the ingest path, so the crash was not hypothetical —
    and `registry._cache_is_fresh` swallowed the same error, which made every referenced file read
    `new` forever.
    """
    from doc_assistant import config
    from doc_assistant.ingest.cache import get_cache_path

    monkeypatch.setattr(config, "DOCS_PATH", tmp_path / "sources")
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")

    got = get_cache_path(tmp_path / "zotero" / "theirs.pdf")
    assert (tmp_path / "cache") in got.parents
    assert got.name == "theirs.md"
    assert got.suffix == ".md"


def test_the_same_referenced_file_always_resolves_to_the_same_entry(tmp_path, monkeypatch):
    """Otherwise the cache never hits and every listing re-extracts."""
    from doc_assistant import config
    from doc_assistant.ingest.cache import get_cache_path

    monkeypatch.setattr(config, "DOCS_PATH", tmp_path / "sources")
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")

    one = get_cache_path(tmp_path / "zotero" / "theirs.pdf")
    two = get_cache_path(tmp_path / "zotero" / "theirs.pdf")
    assert one == two


def test_same_name_in_two_folders_does_not_collide(tmp_path, monkeypatch):
    """Two papers may share a filename; one cache entry for both would serve the wrong text."""
    from doc_assistant import config
    from doc_assistant.ingest.cache import get_cache_path

    monkeypatch.setattr(config, "DOCS_PATH", tmp_path / "sources")
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")

    a = get_cache_path(tmp_path / "zotero" / "paper.pdf")
    b = get_cache_path(tmp_path / "dropbox" / "paper.pdf")
    assert a != b


def test_a_referenced_cache_entry_can_go_stale_like_any_other(tmp_path, monkeypatch):
    """The freshness contract must not quietly differ by root — it decides re-extraction."""
    from doc_assistant import config
    from doc_assistant.ingest.cache import get_cache_path, is_cache_fresh, write_cache

    monkeypatch.setattr(config, "DOCS_PATH", tmp_path / "sources")
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")

    src = tmp_path / "zotero" / "theirs.html"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("<html><body><p>x</p></body></html>", encoding="utf-8")

    cached = get_cache_path(src)
    cached.parent.mkdir(parents=True, exist_ok=True)
    write_cache(cached, "# x", source=src)
    assert is_cache_fresh(src, cached), "a just-written cache is fresh"

    os.utime(src, (cached.stat().st_mtime + 10, cached.stat().st_mtime + 10))
    assert not is_cache_fresh(src, cached), "a newer source must invalidate it"


# ============================================================
# Per-format scoping (KI-48). Two failure modes, opposite directions:
# under-invalidating serves text no current version would produce (KI-40 again), while
# over-invalidating re-extracts a corpus for a change that could not have touched it.
# ============================================================

_FORMATS = (".pdf", ".epub", ".html", ".docx", ".rtf", ".odt", ".txt")


def _fingerprints() -> dict[str, str]:
    from doc_assistant.ingest.cache import extraction_fingerprint

    extraction_fingerprint.cache_clear()
    return {suffix: extraction_fingerprint(suffix) for suffix in _FORMATS}


def test_every_format_has_its_own_identity():
    """If two formats share a fingerprint, one of them is invalidated by the other's changes."""
    fps = _fingerprints()
    assert len(set(fps.values())) == len(_FORMATS), fps


def test_a_change_to_a_formats_entry_point_moves_only_that_format(monkeypatch):
    """The KI-40 reintroduction guard, rewritten so it can actually fail.

    It used to hand `_extraction_closure` the entry point as a *seed* and then assert the seed
    came back in the result — which every seed does, unconditionally, by construction. The
    tautology hid the invariant it was named for: that a format's own extractor is hashed into
    that format's fingerprint, and into no other's.

    Stated behaviourally instead. Every non-PDF format is dispatched through the `_EXTRACTORS`
    dict, a *runtime* lookup a static walk cannot follow, so without the seeds `extract_epub` is
    unreachable from `extract_to_markdown` and an EPUB fix would not invalidate one EPUB cache.
    Measured before the seeds existed: the `.epub` closure held two functions and neither was
    `extract_epub`.
    """
    from doc_assistant import extractors

    # Formats sharing one entry function move together, and that is correct — `.txt` and `.md`
    # are both `extract_text`. Grouped so "the others" means the ones that genuinely differ.
    by_entry: dict[str, list[str]] = {}
    for suffix in _FORMATS:
        entry = extractors._EXTRACTORS.get(suffix)
        by_entry.setdefault(entry.__name__ if entry else "extract_pdf_pymupdf", []).append(suffix)

    for name, owned in by_entry.items():
        before = _fingerprints()
        original = getattr(extractors, name)

        def patched(*args, _o=original, **kwargs):
            return _o(*args, **kwargs)

        # Impersonate the function it replaces, so this models *editing that extractor* rather
        # than rebinding the name to something foreign. Without it the closure records the
        # wrapper's own `__name__`, the hashing loop's `getattr(extractors, name)` misses, and
        # the whole thing falls back to the module-wide fingerprint — which moves every format
        # and would make this test pass for the wrong reason.
        patched.__name__ = name
        patched.__qualname__ = name
        patched.__module__ = extractors.__name__

        monkeypatch.setattr(extractors, name, patched)
        after = _fingerprints()
        monkeypatch.undo()

        for suffix in owned:
            assert after[suffix] != before[suffix], (
                f"{suffix}: a change to its own entry point {name} left its fingerprint alone — "
                "that cache would serve text no current version produces (KI-40)"
            )
        for suffix in _FORMATS:
            if suffix not in owned:
                assert after[suffix] == before[suffix], (
                    f"{suffix} was invalidated by a change to {name}, which it does not run"
                )


def test_changing_one_formats_helper_leaves_the_others_alone(monkeypatch):
    """The whole point of KI-48: an EPUB/HTML fix must not re-extract 97 PDFs."""
    from doc_assistant import extractors

    before = _fingerprints()
    original = extractors._soup_to_markdown

    def patched(soup):
        return original(soup) + ""

    monkeypatch.setattr(extractors, "_soup_to_markdown", patched)
    after = _fingerprints()

    assert after[".epub"] != before[".epub"], "EPUB uses the helper and must notice"
    assert after[".html"] != before[".html"], "HTML uses the helper and must notice"
    for untouched in (".pdf", ".docx", ".rtf", ".odt", ".txt"):
        assert after[untouched] == before[untouched], f"{untouched} does not use it"


def test_a_pdf_only_tunable_moves_only_pdf(monkeypatch):
    """`_TEXT_LAYER_KEPT_MIN` is referenced by name, so its VALUE never reaches co_code."""
    from doc_assistant import extractors

    before = _fingerprints()
    monkeypatch.setattr(extractors, "_TEXT_LAYER_KEPT_MIN", 0.75)
    after = _fingerprints()

    assert after[".pdf"] != before[".pdf"]
    for untouched in (".epub", ".html", ".docx", ".rtf", ".odt", ".txt"):
        assert after[untouched] == before[untouched]


def test_a_shared_exit_step_moves_every_format(monkeypatch):
    """`strip_image_placeholders` runs for all of them, so narrowing must not lose it (KI-14)."""
    from doc_assistant import extractors

    before = _fingerprints()
    original = extractors.strip_image_placeholders

    def patched(md):
        return original(md)

    monkeypatch.setattr(extractors, "strip_image_placeholders", patched)
    after = _fingerprints()

    for suffix in _FORMATS:
        assert after[suffix] != before[suffix], f"{suffix} missed a change to the shared exit"


def test_the_fingerprint_does_not_embed_a_memory_address():
    """Determinism across processes. `_EXTRACTORS` reprs its functions **with their addresses**,
    so hashing that repr would give a different answer every run and no cache would ever read
    fresh again. Caught by measurement, not review."""
    from doc_assistant import extractors
    from doc_assistant.ingest.cache import _stable_repr

    rendered = _stable_repr(extractors._EXTRACTORS)
    assert " at 0x" not in rendered and " object at " not in rendered, rendered
    assert _stable_repr(extractors._EXTRACTORS) == rendered, "must be stable within a run too"


def test_a_regex_constant_is_hashed_by_its_pattern():
    """The pattern is the part that changes behaviour; `re.compile` objects repr stably anyway,
    but pinning it means a flags-only change is still visible."""
    import re as _re

    from doc_assistant.ingest.cache import _stable_repr

    assert _stable_repr(_re.compile("a+", _re.I)) != _stable_repr(_re.compile("a+"))
    assert _stable_repr(_re.compile("a+")) != _stable_repr(_re.compile("b+"))


def test_scoping_failure_falls_back_to_the_whole_module(monkeypatch):
    """Over-invalidating costs CPU; under-invalidating serves stale text.

    The fallback picks CPU.
    """
    from doc_assistant.ingest import cache as cache_mod

    def boom(*_a, **_k):
        raise RuntimeError("walk exploded")

    monkeypatch.setattr(cache_mod, "_extraction_closure", boom)
    cache_mod.extraction_fingerprint.cache_clear()
    assert cache_mod.extraction_fingerprint(".pdf") == cache_mod._whole_module_fingerprint()
    cache_mod.extraction_fingerprint.cache_clear()


def test_the_cache_pair_round_trips_per_format(tmp_path, monkeypatch):
    """End to end: what `write_cache` records is what `is_cache_fresh` accepts, per format."""
    from doc_assistant import config
    from doc_assistant.ingest.cache import get_cache_path, is_cache_fresh, write_cache

    monkeypatch.setattr(config, "DOCS_PATH", tmp_path / "sources")
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")

    for name in ("a.pdf", "b.epub", "c.html", "d.txt"):
        src = tmp_path / "sources" / name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"x")
        cached = get_cache_path(src)
        cached.parent.mkdir(parents=True, exist_ok=True)
        write_cache(cached, "# x", source=src)
        assert is_cache_fresh(src, cached), f"{name} must read fresh right after being written"


def test_a_renamed_extractor_does_not_silently_re_extract_the_corpus(monkeypatch):
    """A function whose `__name__` differs from its attribute must not collapse the scope.

    The closure returned function *names*, and the caller then re-resolved each one with
    `getattr(extractors, name)`. For an alias — or any decorator that does not carry
    `functools.wraps` — that lookup raised, the blanket `except` caught it, and the fingerprint
    silently became the **whole-module** one. Safe in direction (it over-invalidates) and
    expensive in fact: a corpus-wide re-extraction, 61 min for 97 documents and ~55 h projected
    at 10,000, triggered by a rename that could not change a single byte of output.

    The tell is precise: under the defect the format's fingerprint *equals* `suffix=None`'s.
    """
    import structlog

    from doc_assistant import extractors
    from doc_assistant.ingest.cache import extraction_fingerprint

    original = extractors.extract_docx

    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    # Deliberately NOT impersonating: `__name__` stays "wrapper" while the attribute is
    # "extract_docx". This is what an alias or a bare decorator looks like.
    wrapper.__module__ = extractors.__name__
    monkeypatch.setattr(extractors, "extract_docx", wrapper)

    with structlog.testing.capture_logs() as logs:
        extraction_fingerprint.cache_clear()
        docx = extraction_fingerprint(".docx")
        whole = extraction_fingerprint(None)

    assert docx != whole, "the scope collapsed to whole-module — every format would re-extract"
    assert not [entry for entry in logs if entry.get("event") == "extraction_fingerprint_fallback"]


def test_the_other_formats_are_untouched_by_a_renamed_extractor(monkeypatch):
    """The complement: a `.docx`-only change must still leave the other six alone."""
    from doc_assistant import extractors

    before = _fingerprints()
    original = extractors.extract_docx

    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    wrapper.__module__ = extractors.__name__
    monkeypatch.setattr(extractors, "extract_docx", wrapper)
    after = _fingerprints()

    assert after[".docx"] != before[".docx"], "its own change must still be noticed"
    for untouched in (".pdf", ".epub", ".html", ".rtf", ".odt", ".txt"):
        assert after[untouched] == before[untouched], f"{untouched} re-extracts for nothing"
