"""The letter-spacing ignore list reaches one document's extraction and fingerprint (ADR-056).

The list is the user's own file in the data home: per PDF (by sha256, with its size beside it),
the spaced forms to keep, or ``*`` for the whole document. These pin what it may cost and what it
may move: nothing at all without a list, one document's re-extraction when an entry changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from doc_assistant import config
from doc_assistant.ingest import cache as cache_mod
from doc_assistant.ingest.cache import (
    document_fingerprint,
    extraction_fingerprint,
    is_cache_fresh,
    read_spacing_record,
    rejoin_ignore_entries,
    rejoin_overrides,
    spacing_record_path,
    write_cache,
)


@pytest.fixture(autouse=True)
def _ignore_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test gets its own (absent) ignore list and a cold memo."""
    path = tmp_path / "rejoin_ignore.json"
    monkeypatch.setattr(config, "REJOIN_IGNORE_PATH", path)
    cache_mod._ignore_list_memo.clear()
    extraction_fingerprint.cache_clear()
    yield path
    cache_mod._ignore_list_memo.clear()
    extraction_fingerprint.cache_clear()


def _pdf(tmp_path: Path, name: str, body: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 " + body)
    return path


def _write_list(path: Path, *entries: tuple[Path, list[str]]) -> None:
    documents = [
        {
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "size": pdf.stat().st_size,
            "file": pdf.name,
            "keep": keep,
        }
        for pdf, keep in entries
    ]
    path.write_text(json.dumps({"version": 1, "documents": documents}), encoding="utf-8")
    cache_mod._ignore_list_memo.clear()


def test_without_a_list_a_document_carries_its_formats_fingerprint(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    assert rejoin_ignore_entries(pdf) == frozenset()
    assert document_fingerprint(pdf) == extraction_fingerprint(".pdf")


def test_an_entry_is_found_by_the_pdfs_hash(tmp_path: Path, _ignore_list: Path) -> None:
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    _write_list(_ignore_list, (pdf, ["A B C D"]))
    assert rejoin_ignore_entries(pdf) == frozenset({"A B C D"})


def test_an_entry_follows_the_file_through_a_rename(tmp_path: Path, _ignore_list: Path) -> None:
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    _write_list(_ignore_list, (pdf, ["*"]))
    moved = pdf.rename(tmp_path / "renamed.pdf")
    assert rejoin_ignore_entries(moved) == frozenset({"*"})


def test_a_pdf_of_the_same_size_but_other_bytes_has_no_entries(
    tmp_path: Path, _ignore_list: Path
) -> None:
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    twin = _pdf(tmp_path, "b.pdf", b"two")
    _write_list(_ignore_list, (pdf, ["A B C D"]))
    assert twin.stat().st_size == pdf.stat().st_size
    assert rejoin_ignore_entries(twin) == frozenset()


def test_a_pdf_whose_size_matches_no_entry_is_never_hashed(
    tmp_path: Path, _ignore_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    other = _pdf(tmp_path, "b.pdf", b"a longer body")
    _write_list(_ignore_list, (pdf, ["A B C D"]))

    def _no_hashing(path: Path) -> str:
        raise AssertionError(f"hashed {path.name}")

    monkeypatch.setattr("doc_assistant.sources_manifest.sha256_file", _no_hashing)
    assert rejoin_ignore_entries(other) == frozenset()


def test_other_formats_never_read_the_list(tmp_path: Path, _ignore_list: Path) -> None:
    epub = tmp_path / "a.epub"
    epub.write_bytes(b"one")
    _ignore_list.write_text("not json", encoding="utf-8")
    assert rejoin_ignore_entries(epub) == frozenset()


def test_a_malformed_list_is_read_as_empty_and_logged(
    tmp_path: Path, _ignore_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inform, don't block: ingest goes on, and the warning says the entries no longer apply."""
    warned: list[str] = []

    class _Recorder:
        def warning(self, event: str, **_: object) -> None:
            warned.append(event)

    # A recorder, not `capture_logs`: the app caches its loggers on first use, so once any earlier
    # test has configured logging, `capture_logs` sees nothing.
    monkeypatch.setattr(cache_mod, "log", _Recorder())
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    _ignore_list.write_text("{not json", encoding="utf-8")
    assert rejoin_ignore_entries(pdf) == frozenset()
    assert warned == ["rejoin_ignore_unreadable"]
    assert document_fingerprint(pdf) == extraction_fingerprint(".pdf")


@pytest.mark.parametrize("text", ['{"documents": null}', '{"documents": 3}', "[]"])
def test_a_list_of_the_wrong_shape_is_read_as_empty_and_logged(
    tmp_path: Path, _ignore_list: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """Valid JSON is not a valid list: every freshness check reads this, so it must not raise."""
    warned: list[str] = []

    class _Recorder:
        def warning(self, event: str, **_: object) -> None:
            warned.append(event)

    monkeypatch.setattr(cache_mod, "log", _Recorder())
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    _ignore_list.write_text(text, encoding="utf-8")
    assert rejoin_ignore_entries(pdf) == frozenset()
    assert warned == ["rejoin_ignore_unreadable"]


def test_an_entry_of_the_wrong_shape_is_skipped_and_the_others_apply(
    tmp_path: Path, _ignore_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warned: list[str] = []

    class _Recorder:
        def warning(self, event: str, **_: object) -> None:
            warned.append(event)

    monkeypatch.setattr(cache_mod, "log", _Recorder())
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    size = pdf.stat().st_size
    documents = [
        {"sha256": sha, "size": size, "file": pdf.name, "keep": None},
        "not an entry",
        {"sha256": sha, "size": size, "file": pdf.name, "keep": ["A B C"]},
    ]
    _ignore_list.write_text(json.dumps({"documents": documents}), encoding="utf-8")
    assert rejoin_ignore_entries(pdf) == frozenset({"A B C"})
    assert warned == ["rejoin_ignore_entries_skipped"]


def test_an_entry_moves_only_its_own_documents_fingerprint(
    tmp_path: Path, _ignore_list: Path
) -> None:
    named = _pdf(tmp_path, "named.pdf", b"one")
    other = _pdf(tmp_path, "other.pdf", b"a longer body")
    before = document_fingerprint(named), document_fingerprint(other)
    _write_list(_ignore_list, (named, ["A B C D"]))
    assert document_fingerprint(named) != before[0]
    assert document_fingerprint(named).startswith(extraction_fingerprint(".pdf") + "+keep:")
    assert document_fingerprint(other) == before[1]


def test_changing_an_entry_re_extracts_that_document_alone(
    tmp_path: Path, _ignore_list: Path
) -> None:
    named = _pdf(tmp_path, "named.pdf", b"one")
    other = _pdf(tmp_path, "other.pdf", b"a longer body")
    for pdf in (named, other):
        write_cache(tmp_path / f"{pdf.stem}.md", "# text", source=pdf)
    assert is_cache_fresh(named, tmp_path / "named.md")
    _write_list(_ignore_list, (named, ["A B C D"]))
    assert not is_cache_fresh(named, tmp_path / "named.md")
    assert is_cache_fresh(other, tmp_path / "other.md")
    write_cache(tmp_path / "named.md", "# text", source=named)
    assert is_cache_fresh(named, tmp_path / "named.md")
    _write_list(_ignore_list, (named, ["A B C D", "N E U"]))
    assert not is_cache_fresh(named, tmp_path / "named.md")


def test_your_spacing_reaches_the_overrides_and_the_fingerprint(
    tmp_path: Path, _ignore_list: Path
) -> None:
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    before = document_fingerprint(pdf)
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    entry = {
        "sha256": sha,
        "size": pdf.stat().st_size,
        "file": "a.pdf",
        "keep": [],
        "spacing": {"N E U": "NE U"},
    }
    _ignore_list.write_text(json.dumps({"version": 1, "documents": [entry]}), encoding="utf-8")
    cache_mod._ignore_list_memo.clear()
    assert rejoin_overrides(pdf) == (frozenset(), {"N E U": "NE U"})
    assert document_fingerprint(pdf) != before


def test_the_record_is_written_replaced_removed_or_left_alone(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    cached = tmp_path / "a.md"
    run = {"spaced": "N E U", "state": "closed", "page": 1}
    write_cache(cached, "# text", source=pdf, spacing_record=[run])
    assert read_spacing_record(cached) == [run]
    write_cache(cached, "# text again", source=pdf)  # a rewrite that does not extract
    assert read_spacing_record(cached) == [run]
    write_cache(cached, "# text", source=pdf, spacing_record=[])
    assert read_spacing_record(cached) == []
    assert not spacing_record_path(cached).exists()


def test_each_recorded_run_gets_its_span_in_the_text_written(tmp_path: Path) -> None:
    """Keyword selection checks ``text[start:end]`` before it drops a run (ADR-056 point 8), so
    a span must point at that run's own occurrence: on its page, not the body's use of the same
    words, and none at all for a run that is not there."""
    head = "NEURONS: STRUCTURE"
    text = (
        f"\n<!-- page:1 -->\nThe lecture says {head} in its body.\n"
        f"\n<!-- page:2 -->\n{head}\nbody\n"
        f"\n<!-- page:3 -->\n{head}\nbody\n"
    )
    runs = [
        {"written": head, "in_text": True, "page": 2},
        {"written": head, "in_text": True, "page": 3},
        {"written": "QRS", "in_text": False, "page": 3},
        {"written": "XYZ", "in_text": True, "page": 3},  # text rewritten since: not there
        {"written": head, "in_text": True, "page": 9},  # no such page
    ]
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    cached = tmp_path / "a.md"
    write_cache(cached, text, source=pdf, spacing_record=runs)
    spans = [(r["start"], r["end"]) for r in read_spacing_record(cached)]
    assert spans[0][0] > text.index("<!-- page:2") and spans[1][0] > text.index("<!-- page:3")
    assert [text[s:e] for s, e in spans[:2]] == [head, head]
    assert spans[2:] == [(None, None)] * 3


def test_two_runs_with_one_closed_form_on_a_page_get_one_occurrence_each() -> None:
    text = "\n<!-- page:1 -->\nABC, then ABC\n"
    runs = [{"written": "ABC", "in_text": True, "page": 1} for _ in range(2)]
    cache_mod.locate_spacing_runs(text, runs)
    assert [r["start"] for r in runs] == [text.index("ABC"), text.rindex("ABC")]


def test_the_list_is_read_once_per_version_of_the_file(
    tmp_path: Path, _ignore_list: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _pdf(tmp_path, "a.pdf", b"one")
    _write_list(_ignore_list, (pdf, ["A B C D"]))
    reads = 0
    real = json.loads

    def _counting(text: str) -> object:
        nonlocal reads
        reads += 1
        return real(text)

    monkeypatch.setattr(cache_mod.json, "loads", _counting)
    for _ in range(5):
        rejoin_ignore_entries(pdf)
    assert reads == 1
