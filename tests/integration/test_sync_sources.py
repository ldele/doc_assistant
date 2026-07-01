"""Integration tests for the sources-manifest boundary (scan / build / download / verify).

No network: ``_http_get`` is monkeypatched. No real config paths: every helper takes
explicit temp paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from doc_assistant import sources_manifest as sm


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_scan_sources_supported_nested_excludes_unsupported(tmp_path: Path) -> None:
    _write(tmp_path / "top.pdf", b"%PDF-1.4 top")
    _write(tmp_path / "notes.md", b"# notes")
    _write(tmp_path / "ignore.xyz", b"binary junk")
    _write(tmp_path / "sub" / "nested.pdf", b"%PDF-1.4 nested")

    entries = sm.scan_sources(tmp_path)
    names = {e.filename for e in entries}

    assert "ignore.xyz" not in names  # unsupported extension dropped
    assert "sub/nested.pdf" in names  # nested file keyed by POSIX relative path
    top = next(e for e in entries if e.filename == "top.pdf")
    assert top.sha256 == hashlib.sha256(b"%PDF-1.4 top").hexdigest()
    assert top.size_bytes == len(b"%PDF-1.4 top")
    assert top.url is None


def test_scan_missing_dir_is_empty(tmp_path: Path) -> None:
    assert sm.scan_sources(tmp_path / "does-not-exist") == []


def test_build_manifest_autofills_from_public_corpus_and_persists(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    public_data = b"%PDF public corpus paper"
    _write(sources / "rag.pdf", public_data)  # in the public corpus
    _write(sources / "private.pdf", b"%PDF my private book")  # not

    public_manifest = tmp_path / "corpus_manifest.yaml"
    public_manifest.write_text(
        sm.render_manifest(
            [
                sm.SourceEntry(
                    filename="rag.pdf",
                    url="https://arxiv.org/pdf/2005.11401v4.pdf",
                    sha256=hashlib.sha256(public_data).hexdigest(),
                    size_bytes=len(public_data),
                )
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "sources_manifest.yaml"

    result = sm.build_manifest(
        sources_dir=sources,
        manifest_path=manifest_path,
        public_corpus_path=public_manifest,
    )

    assert result.total == 2
    assert result.new == 2
    assert result.enriched == 1  # rag.pdf got its url for free
    assert result.missing_url == 1  # private.pdf still needs one

    sm.write_manifest(manifest_path, result.entries)
    reloaded = {e.filename: e for e in sm.load_manifest(manifest_path)}
    assert reloaded["rag.pdf"].url == "https://arxiv.org/pdf/2005.11401v4.pdf"
    assert reloaded["private.pdf"].url is None


def test_build_preserves_user_filled_url_on_rebuild(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    _write(sources / "book.pdf", b"%PDF v1")
    manifest_path = tmp_path / "m.yaml"
    empty_public = tmp_path / "none.yaml"

    first = sm.build_manifest(
        sources_dir=sources, manifest_path=manifest_path, public_corpus_path=empty_public
    )
    # user fills the url by hand, then the file content changes + a rebuild runs
    first.entries[0].url = "https://mine/book.pdf"
    sm.write_manifest(manifest_path, first.entries)
    _write(sources / "book.pdf", b"%PDF v2 revised")

    second = sm.build_manifest(
        sources_dir=sources, manifest_path=manifest_path, public_corpus_path=empty_public
    )
    entry = second.entries[0]
    assert entry.url == "https://mine/book.pdf"  # curated url survives the rebuild
    assert entry.sha256 == hashlib.sha256(b"%PDF v2 revised").hexdigest()  # pin refreshed


def test_download_missing_fetches_verifies_and_skips_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"%PDF downloaded bytes"
    monkeypatch.setattr(sm, "_http_get", lambda url: payload)
    dest = tmp_path / "sources"
    entry = sm.SourceEntry(
        filename="paper.pdf",
        url="https://example.org/paper.pdf",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )

    [outcome] = sm.download_missing([entry], dest)
    assert outcome.status == "downloaded"
    assert (dest / "paper.pdf").read_bytes() == payload

    [again] = sm.download_missing([entry], dest)  # idempotent — already on disk
    assert again.status == "present"


def test_download_missing_flags_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sm, "_http_get", lambda url: b"wrong bytes")
    entry = sm.SourceEntry("p.pdf", "https://x/p.pdf", sha256="0" * 64, size_bytes=5)
    [outcome] = sm.download_missing([entry], tmp_path)
    assert outcome.status == "mismatch"
    assert (tmp_path / "p.pdf").exists()  # file still written for inspection


def test_download_missing_no_url_and_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(url: str) -> bytes:
        raise AssertionError("network must not be touched")

    monkeypatch.setattr(sm, "_http_get", _boom)
    no_url = sm.SourceEntry("private.pdf", None, "s", 1)
    has_url = sm.SourceEntry("pub.pdf", "https://x/pub.pdf", "s", 1)

    results = sm.download_missing([no_url, has_url], tmp_path, dry_run=True)
    outcomes = {o.filename: o for o in results}
    assert outcomes["private.pdf"].status == "no_url"
    assert outcomes["pub.pdf"].status == "would_download"
    assert not (tmp_path / "pub.pdf").exists()  # dry-run writes nothing


def test_verify_present_detects_tamper(tmp_path: Path) -> None:
    good = b"%PDF intact"
    _write(tmp_path / "a.pdf", good)
    entry = sm.SourceEntry("a.pdf", None, hashlib.sha256(good).hexdigest(), len(good))

    [ok] = sm.verify_present([entry], tmp_path)
    assert ok.status == "present"

    (tmp_path / "a.pdf").write_bytes(b"tampered")
    [bad] = sm.verify_present([entry], tmp_path)
    assert bad.status == "mismatch"
