"""Guard tests for the extraction record (`ingest.record`, spec `docs/specs/ingestion-record.md`).

A temp SQLite bound to the global session machinery, invented documents and cached text, no
Chroma, no model. The passes themselves are `reingest`'s and are tested there; what is pinned here
is what the record says about them: each state, each warning and its near miss, the carry-over of
a pass a run did not run, the bound on kept rows, and that a failing pass never loses the record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import reingest
from doc_assistant.db.models import Base, Document, ExtractionRecord
from doc_assistant.db.session import session_scope
from doc_assistant.ingest import cache, record
from doc_assistant.ingest.tables import ExtractedTable, splice_tables


@pytest.fixture
def lib(tmp_path: Path) -> Iterator[Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    (tmp_path / "cache").mkdir()
    try:
        yield tmp_path
    finally:
        session_mod._engine, session_mod._SessionLocal = orig_engine, orig_factory
        engine.dispose()


def _pages(texts: list[str]) -> str:
    return "".join(f"\n<!-- page:{i} -->\n{t}" for i, t in enumerate(texts, start=1))


def _doc(
    home: Path, name: str, text: str, *, fmt: str = "pdf", splice: Any = "ran", **kw: Any
) -> str:
    cached = home / "cache" / (Path(name).stem + ".md")
    cached.write_text(text, encoding="utf-8")
    if splice == "ran":
        splice = {
            "version": 1,
            "extracted": 0,
            "candidate_pages": [],
            "seconds": 0.1,
            "error": None,
        }
    if splice is not None:
        cache.tables_record_path(cached).write_text(json.dumps(splice), encoding="utf-8")
    fields: dict[str, Any] = {
        "filename": name,
        "source_original": str(home / name),
        "source_cache": str(cached),
        "doc_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "format": fmt,
        "title": "A title",
        "authors": "A. Author",
        "year": 2020,
        **kw,
    }
    with session_scope() as s:
        d = Document(**fields)
        s.add(d)
        s.flush()
        return str(d.id)


def _refs(
    entries: int, *, heading: bool = True, notes: list[str] | None = None
) -> tuple[str, dict[str, Any]]:
    return "ok", {
        "heading": heading,
        "entries": entries,
        "without_year": 0,
        "matched": 0,
        "notes": notes or [],
        "seconds": 0.0,
    }


def _figs(found: int, captioned: int) -> tuple[str, dict[str, Any]]:
    return "ok", {
        "found": found,
        "cropped": found,
        "with_caption": captioned,
        "caption_only": 0,
        "seconds": 0.0,
    }


def _ids(rec: dict[str, Any]) -> set[str]:
    return {w["id"] for w in rec["warnings"]}


def test_a_full_document_records_found_everywhere_and_warns_of_nothing(lib: Path) -> None:
    tables = [ExtractedTable(page=2, index=1, rows=[["a", "b"], ["1", "2"]])]
    text = splice_tables(_pages(["Intro", "Table 1: means\nrow"]), tables)
    doc = _doc(lib, "a.pdf", text)
    record.write_record(doc, "index", {"references": _refs(30), "figures": _figs(3, 3)})
    rec = record.current_record(doc)
    assert rec is not None
    assert rec["states"] == {"references": "found", "figures": "found", "tables": "found"}
    assert rec["text"]["pages"] == 2 and rec["tables"]["extracted"] == 1
    assert rec["warnings"] == []
    assert rec["summary"] == "30 references · 3 figures · 1 table"


def test_a_heading_with_no_entries_and_no_heading_are_two_different_warnings(lib: Path) -> None:
    a = _doc(lib, "a.pdf", _pages(["x"]))
    b = _doc(lib, "b.pdf", _pages(["y"]))
    record.write_record(a, "index", {"references": _refs(0, heading=True), "figures": _figs(0, 0)})
    record.write_record(
        b,
        "index",
        {
            "references": _refs(0, heading=False, notes=["no references heading detected"]),
            "figures": _figs(0, 0),
        },
    )
    ra, rb = record.current_record(a), record.current_record(b)
    assert ra and rb
    assert "references_heading_only" in _ids(ra) and "references_none" not in _ids(ra)
    assert "references_none" in _ids(rb) and "references_heading_only" not in _ids(rb)
    assert "no references heading detected" in next(
        w["text"] for w in rb["warnings"] if w["id"] == "references_none"
    )


def test_a_short_list_is_partial_and_a_full_one_is_not(lib: Path) -> None:
    a = _doc(lib, "a.pdf", _pages(["x"]))
    record.write_record(a, "index", {"references": _refs(2), "figures": _figs(0, 0)})
    rec = record.current_record(a)
    assert rec and rec["states"]["references"] == "partial"
    assert "references_few" in _ids(rec)
    record.write_record(
        a, "index", {"references": _refs(record.REFERENCES_FEW), "figures": _figs(0, 0)}
    )
    rec = record.current_record(a)
    assert rec and rec["states"]["references"] == "found" and "references_few" not in _ids(rec)


def test_a_page_naming_a_table_none_was_extracted_from_is_named(lib: Path) -> None:
    doc = _doc(
        lib, "a.pdf", _pages(["Table 1: results\nrow", "as Table 1 shows", "Table 2 means"])
    )
    record.write_record(doc, "index", {"references": _refs(10), "figures": _figs(0, 0)})
    rec = record.current_record(doc)
    assert rec and rec["states"]["tables"] == "none_found"
    assert rec["tables"]["pages_without_one"] == [1, 3]  # page 2 only cites it mid-line
    assert any(w["text"].startswith("Pages 1, 3 name a table") for w in rec["warnings"])


def test_a_non_pdf_skips_figures_and_tables_with_a_reason(lib: Path) -> None:
    doc = _doc(lib, "a.epub", "# A book\n\nText.", fmt="epub", splice=None)
    record.run_index_passes([doc])
    rec = record.current_record(doc)
    assert rec is not None
    assert rec["states"]["figures"] == "skipped" and rec["states"]["tables"] == "skipped"
    assert "PDF" in rec["figures"]["reason"] and "PDF" in rec["tables"]["reason"]
    assert rec["states"]["references"] == "none_found"  # read from its text, none in it


def test_a_pass_that_raises_leaves_the_record_with_its_reason(
    lib: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = _doc(lib, "a.pdf", _pages(["x"]))

    def boom(_doc: Any, _facts: Any = None) -> tuple[str, str]:
        raise RuntimeError("page 3 is unreadable")

    monkeypatch.setattr(reingest, "_rerun_figures", boom)
    assert record.run_index_passes([doc]) == 1
    rec = record.current_record(doc)
    assert rec and rec["states"]["figures"] == "error"
    assert "page 3 is unreadable" in rec["figures"]["reason"]
    assert rec["states"]["references"] == "none_found"  # the other pass still ran


def test_a_pass_this_run_did_not_run_is_carried_from_the_newest_record(lib: Path) -> None:
    doc = _doc(lib, "a.pdf", _pages(["x"]))
    record.write_record(doc, "index", {"references": _refs(20), "figures": _figs(4, 2)})
    record.write_record(doc, "rerun", {"references": _refs(25)})
    rec = record.current_record(doc)
    assert rec and rec["run_kind"] == "rerun"
    assert rec["references"]["entries"] == 25
    assert rec["states"]["figures"] == "found" and rec["figures"]["found"] == 4
    assert "carried_from" in rec["figures"]
    assert "figures_uncaptioned" in _ids(rec)


def test_never_looked_for_is_not_run_not_none_found(lib: Path) -> None:
    doc = _doc(lib, "a.pdf", _pages(["x"]), splice=None)
    record.write_record(doc, "rerun", {})
    rec = record.current_record(doc)
    assert rec and rec["states"] == {
        "references": "not_run",
        "figures": "not_run",
        "tables": "not_run",
    }
    assert rec["warnings"] == []


def test_only_the_newest_records_are_kept(lib: Path) -> None:
    doc = _doc(lib, "a.pdf", _pages(["x"]))
    for n in range(record.RECORDS_KEPT + 3):
        record.write_record(doc, "rerun", {"references": _refs(n + 5)})
    with session_scope() as s:
        kept = s.execute(select(func.count()).select_from(ExtractionRecord)).scalar_one()
    assert kept == record.RECORDS_KEPT
    rec = record.current_record(doc)
    assert rec and rec["references"]["entries"] == record.RECORDS_KEPT + 7


def test_a_record_of_an_earlier_text_reads_as_stale(lib: Path) -> None:
    doc = _doc(lib, "a.pdf", _pages(["x"]))
    record.write_record(doc, "index", {"references": _refs(10), "figures": _figs(0, 0)})
    with session_scope() as s:
        row = s.get(Document, doc)
        assert row is not None
        row.doc_hash = "0" * 16
    rec = record.current_record(doc)
    assert rec and rec["stale"] and "stale" in _ids(rec)


def test_latest_records_reads_one_row_per_document_and_none_for_the_unrecorded(lib: Path) -> None:
    a = _doc(lib, "a.pdf", _pages(["x"]))
    b = _doc(lib, "b.pdf", _pages(["y"]))
    _doc(lib, "c.pdf", _pages(["z"]))
    record.write_record(a, "index", {"references": _refs(1)})
    record.write_record(a, "rerun", {"references": _refs(9)})
    record.write_record(b, "index", {"references": _refs(0)})
    with session_scope() as s:
        latest = record.latest_records(s, None)
    assert set(latest) == {a, b}
    assert latest[a]["references"]["entries"] == 9


def test_an_unknown_document_has_no_record(lib: Path) -> None:
    assert record.write_record("nope", "index", {}) is None
    assert record.current_record("nope") is None


def test_load_or_extract_splices_a_pdfs_tables_and_records_the_splice(
    lib: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doc_assistant.ingest.regions as regions
    import doc_assistant.ingest.tables as tables

    source = lib / "paper.pdf"
    source.write_bytes(b"%PDF-1.4 not really")
    monkeypatch.setattr(cache, "get_cache_path", lambda _p: lib / "cache" / "paper.md")
    monkeypatch.setattr(cache, "ingest_refusal", lambda _p: None)
    monkeypatch.setattr(cache, "extract_to_markdown", lambda *_a, **_k: _pages(["Table 1: x"]))
    monkeypatch.setattr(regions, "table_candidate_pages", lambda _p: [1])
    monkeypatch.setattr(
        tables,
        "extract_tables_from_pages",
        lambda _p, _pages: [ExtractedTable(page=1, index=1, rows=[["a", "b"], ["1", "2"]])],
    )
    text = cache.load_or_extract(source)
    assert tables.has_spliced_tables(text)
    facts = cache.read_tables_record(lib / "cache" / "paper.md")
    assert facts is not None and facts["extracted"] == 1 and facts["error"] is None


def test_a_failing_splice_keeps_the_text_and_records_the_error(
    lib: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doc_assistant.ingest.regions as regions

    source = lib / "paper.pdf"
    source.write_bytes(b"%PDF-1.4 not really")
    monkeypatch.setattr(cache, "get_cache_path", lambda _p: lib / "cache" / "paper.md")
    monkeypatch.setattr(cache, "ingest_refusal", lambda _p: None)
    monkeypatch.setattr(cache, "extract_to_markdown", lambda *_a, **_k: _pages(["body"]))

    def boom(_p: str) -> list[int]:
        raise ValueError("bad xref")

    monkeypatch.setattr(regions, "table_candidate_pages", boom)
    assert "body" in cache.load_or_extract(source)
    facts = cache.read_tables_record(lib / "cache" / "paper.md")
    assert facts is not None and "bad xref" in facts["error"]


def test_a_change_to_the_table_splice_moves_the_pdf_fingerprint_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import doc_assistant.ingest.tables as tables

    def clear() -> None:
        cache._table_fingerprint.cache_clear()
        cache.extraction_fingerprint.cache_clear()  # type: ignore[attr-defined]

    clear()
    pdf, epub = cache.extraction_fingerprint(".pdf"), cache.extraction_fingerprint(".epub")

    def other(markdown: str) -> bool:
        return "changed" in markdown

    monkeypatch.setattr(tables, "has_spliced_tables", other)
    clear()
    try:
        assert cache.extraction_fingerprint(".pdf") != pdf
        assert cache.extraction_fingerprint(".epub") == epub
    finally:
        monkeypatch.undo()
        clear()


def test_the_real_figures_pass_fills_the_record(
    lib: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index run's first real use raised: the counts read committed (expired) rows."""
    import pymupdf

    import doc_assistant.ingest.figures as figures
    from doc_assistant.ingest.figures import FigureRegion

    pdf = lib / "a.pdf"
    handle = pymupdf.open()  # type: ignore[no-untyped-call]
    handle.new_page()  # type: ignore[no-untyped-call]
    handle.save(str(pdf))  # type: ignore[no-untyped-call]
    handle.close()  # type: ignore[no-untyped-call]
    doc = _doc(lib, "a.pdf", _pages(["Figure 1: a thing"]))
    monkeypatch.setattr(reingest, "_source_path", lambda _d: pdf)
    monkeypatch.setattr(figures, "figure_dir", lambda _h: lib / "figures")
    monkeypatch.setattr(
        figures,
        "detect_figure_regions",
        lambda _p: [
            FigureRegion(1, None, "figure", "Figure 1: a thing", None, "caption_only"),
            FigureRegion(1, None, "figure", None, None, "caption_only"),
        ],
    )
    record.run_index_passes([doc])
    rec = record.current_record(doc)
    assert rec and rec["states"]["figures"] == "found"
    assert rec["figures"]["found"] == 2 and rec["figures"]["with_caption"] == 1
    assert rec["figures"]["caption_only"] == 2
    assert "figures_uncaptioned" in _ids(rec)


def test_a_finished_row_gets_the_records_line_and_first_warning_and_others_do_not() -> None:
    from doc_assistant.ingest.run import DocumentResult, IngestRun

    run = IngestRun()
    run.begin([Path("C:/lib/a.pdf"), Path("C:/lib/b.pdf")])
    run.record(0, DocumentResult("added", passages=12))
    run.record(1, DocumentResult("error", reason="not a PDF"))
    run.found("C:/lib/a.pdf", "30 references · 3 figures · 1 table", "3 figures without a caption")
    run.found("C:/lib/b.pdf", "should not land", None)
    rows, _ = run.page(0, 10)
    by_name = {r.name: r for r in rows}
    assert by_name["a.pdf"].detail == "12 passages · 30 references · 3 figures · 1 table"
    assert by_name["a.pdf"].warning == "3 figures without a caption"
    assert by_name["b.pdf"].detail == "not a PDF" and by_name["b.pdf"].warning is None


def test_the_summary_counts_each_state_and_its_issue_opens_the_same_documents(lib: Path) -> None:
    from doc_assistant.library.summary import issue_document_ids, summarise_collection

    heading = _doc(lib, "h.pdf", _pages(["x"]))
    listed = _doc(lib, "l.pdf", _pages(["Table 1: x"]))
    _doc(lib, "old.pdf", _pages(["y"]), splice=None)  # never recorded: not looked for
    record.write_record(heading, "index", {"references": _refs(0), "figures": _figs(2, 1)})
    record.write_record(listed, "index", {"references": _refs(30), "figures": _figs(0, 0)})
    s = summarise_collection()
    assert s is not None
    r = s.found.record
    assert (r.recorded, r.references_listed, r.references_heading_only) == (2, 1, 1)
    assert r.references_not_looked_for == 1
    assert (r.figures_uncaptioned, r.figures_uncaptioned_documents) == (1, 1)
    assert (r.tables_missed, r.tables_not_looked_for) == (1, 1)
    assert issue_document_ids("references_heading_only") == {heading}
    assert issue_document_ids("tables_missed") == {listed}
    assert len(issue_document_ids("references_not_looked_for")) == 1


def test_the_summary_at_zero_documents_has_no_record_counts(lib: Path) -> None:
    from doc_assistant.library.summary import RecordCounts, summarise_collection

    s = summarise_collection()
    assert s is not None and s.documents == 0 and s.found.record == RecordCounts()
