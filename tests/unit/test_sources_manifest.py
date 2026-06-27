"""Pure-core tests for the private sources manifest (merge / enrich / (de)serialise)."""

from __future__ import annotations

from doc_assistant.sources_manifest import (
    SourceEntry,
    enrich_with_public_corpus,
    entry_from_dict,
    entry_to_dict,
    merge_entries,
    parse_manifest,
    render_manifest,
)


def test_entry_roundtrip_maps_bytes_key() -> None:
    entry = SourceEntry(filename="a.pdf", url="https://x/a.pdf", sha256="abc", size_bytes=10)
    as_dict = entry_to_dict(entry)
    assert as_dict["bytes"] == 10  # on-disk key is `bytes`, field is `size_bytes`
    assert entry_from_dict(as_dict) == entry


def test_parse_manifest_empty_and_malformed() -> None:
    assert parse_manifest("") == []
    assert parse_manifest("documents:\n") == []
    assert parse_manifest("- just\n- a\n- list\n") == []  # non-dict top level


def test_render_parse_roundtrip_and_header() -> None:
    entries = [
        SourceEntry("paper.pdf", "https://arxiv.org/x.pdf", "deadbeef", 123),
        SourceEntry("private.pdf", None, "cafef00d", 456),
    ]
    text = render_manifest(entries)
    assert "GITIGNORED" in text  # the warning header survives
    assert parse_manifest(text) == entries


def test_merge_preserves_url_refreshes_hash() -> None:
    existing = [SourceEntry("a.pdf", "https://x/a.pdf", "oldsha", 1)]
    scanned = [SourceEntry("a.pdf", None, "newsha", 2)]
    [merged] = merge_entries(existing, scanned)
    assert merged.url == "https://x/a.pdf"  # curated url kept
    assert merged.sha256 == "newsha" and merged.size_bytes == 2  # content pin refreshed


def test_merge_appends_new_and_keeps_absent() -> None:
    existing = [SourceEntry("gone.pdf", "https://x/gone.pdf", "s", 1)]
    scanned = [SourceEntry("fresh.pdf", None, "s2", 2)]
    merged = merge_entries(existing, scanned)
    names = [e.filename for e in merged]
    assert names == ["gone.pdf", "fresh.pdf"]  # absent kept first, new appended
    assert merged[1].url is None


def test_enrich_by_sha_then_filename() -> None:
    public = [
        SourceEntry("rag.pdf", "https://arxiv/rag.pdf", "shaRAG", 9),
        SourceEntry("dpr.pdf", "https://arxiv/dpr.pdf", "shaDPR", 9),
    ]
    entries = [
        SourceEntry("renamed.pdf", None, "shaRAG", 9),  # matches by checksum, different name
        SourceEntry("dpr.pdf", None, "different", 9),  # matches by filename fallback
        SourceEntry("mine.pdf", None, "nope", 9),  # no match
    ]
    filled = enrich_with_public_corpus(entries, public)
    assert filled == 2
    assert entries[0].url == "https://arxiv/rag.pdf"
    assert entries[1].url == "https://arxiv/dpr.pdf"
    assert entries[2].url is None


def test_enrich_skips_already_filled() -> None:
    public = [SourceEntry("a.pdf", "https://public/a.pdf", "sha", 1)]
    entries = [SourceEntry("a.pdf", "https://mine/a.pdf", "sha", 1)]
    assert enrich_with_public_corpus(entries, public) == 0
    assert entries[0].url == "https://mine/a.pdf"  # untouched
