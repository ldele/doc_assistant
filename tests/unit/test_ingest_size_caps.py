"""Ingest size caps — docs/security.md S4, step S-1.

A file over the byte cap, and an EPUB/DOCX/ODT whose archive expands past its cap or is shaped like
a zip bomb, is refused **before it is opened**: with a sentence in the add review sheet
(`get_format_status`), and with `IngestRefusedError` just before extraction (`load_or_extract`).
Real documents of every archive format must pass, and the check must stay outside the extraction
fingerprint, or adding it would have re-extracted the whole corpus (KI-48).
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from doc_assistant import config, extractors
from doc_assistant.extractors import IngestRefusedError, get_format_status, ingest_refusal
from doc_assistant.ingest import cache
from tests.unit.test_extractors_formats import _write_docx, _write_epub, _write_odt

MiB = 1 << 20


def _zip(path: Path, entries: dict[str, bytes], method: int = zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", compression=method, compresslevel=9) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def _bomb(path: Path) -> Path:
    """10 MiB of zeros deflates to ~10 KB: ~1030:1, deflate's ceiling — no real text gets there."""
    return _zip(path, {"mimetype": b"application/epub+zip", "OEBPS/pad.xhtml": bytes(10 * MiB)})


def test_a_file_over_the_byte_cap_is_refused_by_name_in_the_review_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big = tmp_path / "notes.md"
    big.write_bytes(b"x" * 4096)
    monkeypatch.setattr(config, "MAX_INGEST_FILE_BYTES", 1024)
    supported, advisory = get_format_status(big)
    assert supported is False
    assert advisory is not None
    assert "4.0 KB" in advisory and "1.0 KB limit" in advisory


def test_a_zip_bomb_is_refused_by_name_in_the_review_sheet(tmp_path: Path) -> None:
    supported, advisory = get_format_status(_bomb(tmp_path / "book.epub"))
    assert supported is False
    assert advisory is not None
    assert "EPUB" in advisory and "zip bombs" in advisory


def test_an_archive_that_expands_past_its_cap_is_refused(tmp_path: Path) -> None:
    """Stored (uncompressed) entries: the ratio is 1, so only the expanded total can catch it."""
    doc = _zip(tmp_path / "report.docx", {"a.xml": b"a" * 3000}, method=zipfile.ZIP_STORED)
    sentence = ingest_refusal(doc, max_expanded=2048)
    assert sentence is not None
    assert "DOCX would expand to" in sentence


def test_a_damaged_archive_is_refused_with_a_sentence_not_a_traceback(tmp_path: Path) -> None:
    fake = tmp_path / "paper.odt"
    fake.write_bytes(b"this is not a zip archive")
    sentence = ingest_refusal(fake)
    assert sentence is not None
    assert "damaged or is not a real ODT file" in sentence


@pytest.mark.parametrize(
    ("suffix", "write"), [(".docx", _write_docx), (".odt", _write_odt), (".epub", _write_epub)]
)
def test_real_documents_of_every_archive_format_are_not_refused(
    tmp_path: Path, suffix: str, write: object
) -> None:
    path = write(tmp_path / f"real{suffix}")  # type: ignore[operator]
    assert ingest_refusal(path) is None
    assert get_format_status(path) == (True, None)


def test_non_archive_formats_under_the_cap_are_not_refused(tmp_path: Path) -> None:
    for name in ("a.pdf", "b.md", "c.txt", "d.html", "e.rtf"):
        p = tmp_path / name
        p.write_bytes(b"%PDF-1.4 small" if name.endswith(".pdf") else b"small")
        assert ingest_refusal(p) is None, name


def test_extraction_refuses_before_opening_a_bomb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ingest path — a file dropped into the source folder never passes the review sheet."""
    bomb = _bomb(tmp_path / "book.epub")
    monkeypatch.setattr(cache, "get_cache_path", lambda _p: tmp_path / "cache" / "book.md")

    def must_not_run(*_a: object, **_k: object) -> str:
        raise AssertionError("extract_to_markdown opened a refused file")

    monkeypatch.setattr(cache, "extract_to_markdown", must_not_run)
    with pytest.raises(IngestRefusedError, match="zip bombs"):
        cache.load_or_extract(bomb)
    assert not (tmp_path / "cache").exists()


def test_the_refusal_is_a_value_error_so_ingest_reports_it_per_file() -> None:
    """`ingest` turns any exception into a per-file `error` with its message; this keeps the
    refusal inside that path instead of needing a handler of its own."""
    assert issubclass(IngestRefusedError, ValueError)


def test_the_size_check_is_outside_every_formats_extraction_fingerprint() -> None:
    """KI-48: the fingerprint hashes what `extract_to_markdown` can reach. Were the check
    reachable, adding or tuning it would mark every cached document stale and re-extract the
    corpus."""
    seeds = tuple(fn.__name__ for fn in extractors._EXTRACTORS.values())
    functions, constants, _ = cache._extraction_closure(frozenset(), seeds)
    for name in ("ingest_refusal", "get_format_status", "_size_label"):
        assert name not in functions
    assert "_ARCHIVE_FORMATS" not in constants


def test_zipfile_never_decompresses_past_an_entrys_declared_size(tmp_path: Path) -> None:
    """The assumption the declared-size cap rests on, pinned: an archive that LIES about an entry's
    size (declares 1 KB, holds 1 MiB) is truncated at the declared size, then fails its CRC.
    If CPython ever stopped truncating, the cap would stop bounding memory — this test says so."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", bytes(MiB))
    raw = bytearray(buf.getvalue())
    # Central directory header: signature PK\x01\x02, uncompressed size at offset 24 (4 bytes).
    cd = raw.rfind(b"PK\x01\x02")
    assert cd > 0
    struct.pack_into("<I", raw, cd + 24, 1024)
    liar = tmp_path / "liar.docx"
    liar.write_bytes(bytes(raw))

    read = 0
    with (
        pytest.raises(zipfile.BadZipFile),
        zipfile.ZipFile(liar) as z,
        z.open("word/document.xml") as f,
    ):
        while chunk := f.read(4096):
            read += len(chunk)
    assert read <= 1024
