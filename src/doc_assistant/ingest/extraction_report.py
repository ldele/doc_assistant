"""What the fast track's extraction finds in a document, and what it gets wrong (0.9.0, session 2).

Read-only by construction: it reads a fresh cache when there is one and otherwise extracts in
memory, runs the metadata, reference, figure and table passes on the result, and returns a
`DocumentReport`. It writes no cache, no row, no figure image and no table; the runner
(`scripts/extraction_report.py`) writes only its own report file. It is the seam for session 3's
record of what indexing found.

The fault flags are pure functions over the extracted text, each derived from the document itself
rather than from a list tuned to this corpus (the robustness contract):

- **running headers and footers**: a page's first or last line that recurs, digits ignored, on
  at least `RUNNING_MIN_PAGES` pages and `RUNNING_MIN_SHARE` of them;
- **a title that is a running header** (KI-54's journal name): the stored or extracted title is
  one of those lines;
- **an author line that reads as prose** (KI-89): `metadata_extractor._reads_as_prose`;
- **a page that repeats the page before** (KI-57's mechanism): a segment byte-identical to the
  previous one;
- **words split at a line-end hyphen**: ``exam-\\nple`` counted only when ``example`` occurs
  unbroken elsewhere in the same document, so ``well-\\nknown`` is not counted;
- **tables drawn as text**: a page whose text names a table (``Table 3``) where the table pass
  found none.

Cost: one open of the PDF per pass, as each pass opens it itself today; the runner times each
pass, which is the measurement session 3 needs (`docs/performance.md`).
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeVar

from doc_assistant.ingest.chunking import PAGE_MARKER

#: A running line must recur on at least this many pages... (structural: two pages can share a
#: line by chance, a section heading say; three in the same top or bottom slot is a header).
RUNNING_MIN_PAGES = 3
#: ...and on at least this share of the document's pages, so a 300-page book's chapter heading
#: repeated on its chapter's 4 pages is not called the book's running header.
RUNNING_MIN_SHARE = 0.2
#: Lines shorter than this are page numbers and stray marks once digits are folded; not a header.
RUNNING_MIN_CHARS = 4
#: How many lines at each end of a page are looked at for a running line.
RUNNING_EDGE_LINES = 2
#: How many examples a report keeps per flag (what a reader can check by hand).
EXAMPLES_KEPT = 5

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")
_MD_DECOR = re.compile(r"[#*_`~>|]+")
_HYPHEN_SPLIT = re.compile(r"\b([A-Za-z]{2,})-[ \t]*\n[ \t]*([a-z]{2,})\b")
_WORD = re.compile(r"[A-Za-z]{4,}")
_TABLE_MENTION = re.compile(r"(?m)^\W{0,6}Table\s+([0-9]+|[IVXLC]+)\b")


# ============================================================
# Pure flags over text (tested in tests/unit/test_extraction_report.py)
# ============================================================


def page_segments(markdown: str) -> list[tuple[int, str]]:
    """The cache's pages as ``(page number, text)``, in order. Text with no page marker (EPUB,
    HTML, DOCX) is one segment numbered 0."""
    marks = list(PAGE_MARKER.finditer(markdown))
    if not marks:
        return [(0, markdown)]
    out: list[tuple[int, str]] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(markdown)
        out.append((int(m.group(1)), markdown[m.end() : end]))
    return out


def _fold(line: str) -> str:
    """A line as compared across pages: markdown decoration, digits and case folded away."""
    text = _MD_DECOR.sub(" ", line)
    text = _DIGITS.sub("#", text)
    return _SPACES.sub(" ", text).strip().lower()


def _edge_lines(text: str) -> list[str]:
    lines = [ln for ln in (s.strip() for s in text.splitlines()) if ln]
    if len(lines) <= 2 * RUNNING_EDGE_LINES:
        return lines
    return lines[:RUNNING_EDGE_LINES] + lines[-RUNNING_EDGE_LINES:]


def running_lines(segments: Sequence[tuple[int, str]]) -> dict[str, int]:
    """Folded lines that recur at a page's top or bottom, with how many pages carry each."""
    pages = [s for s in segments if s[1].strip()]
    if len(pages) < RUNNING_MIN_PAGES:
        return {}
    counts: Counter[str] = Counter()
    for _, text in pages:
        counts.update({_fold(ln) for ln in _edge_lines(text)})
    floor = max(RUNNING_MIN_PAGES, RUNNING_MIN_SHARE * len(pages))
    return {
        line: n
        for line, n in counts.items()
        if n >= floor and len(line.replace("#", "").strip()) >= RUNNING_MIN_CHARS
    }


def title_is_running_line(title: str | None, running: dict[str, int]) -> bool:
    """KI-54: a title that is also a line repeated across the pages' edges (a journal's name)."""
    return bool(title) and _fold(title or "") in running


def repeated_pages(segments: Sequence[tuple[int, str]]) -> list[int]:
    """KI-57's mechanism: pages whose text is byte-identical to the page before (non-empty)."""
    return [
        page
        for (_, prev), (page, text) in pairwise(segments)
        if text.strip() and text.strip() == prev.strip()
    ]


def hyphen_splits(markdown: str) -> list[str]:
    """Words broken at a line-end hyphen whose joined form occurs unbroken in the same text."""
    words = {w.lower() for w in _WORD.findall(markdown)}
    found: list[str] = []
    for m in _HYPHEN_SPLIT.finditer(markdown):
        joined = (m.group(1) + m.group(2)).lower()
        if joined in words:
            found.append(f"{m.group(1)}-{m.group(2)}")
    return found


def table_mentions(segments: Sequence[tuple[int, str]]) -> dict[int, list[str]]:
    """Pages whose text opens a line with *Table N* (a caption or its label), by page."""
    out: dict[int, list[str]] = {}
    for page, text in segments:
        labels = sorted({m.group(1) for m in _TABLE_MENTION.finditer(text)})
        if labels:
            out[page] = labels
    return out


# ============================================================
# The report for one document
# ============================================================


@dataclass
class PassResult:
    """One pass: what it found, how long it took, and its error if it failed."""

    seconds: float = 0.0
    error: str | None = None


@dataclass
class DocumentReport:
    """Everything the fast track's passes found in one document. Plain data, JSON-ready."""

    path: str
    name: str
    format: str
    text_source: str = ""  # "cache" (fresh) or "extracted" (in memory, nothing written)
    pages: int = 0
    chars: int = 0
    empty_pages: list[int] = field(default_factory=list)
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    stored_title: str | None = None
    stored_authors: str | None = None
    references_found: bool = False
    reference_entries: int = 0
    reference_entries_without_year: int = 0
    reference_notes: list[str] = field(default_factory=list)
    figures: int = 0
    figures_caption_only: int = 0
    figures_without_caption: int = 0
    tables: int = 0
    table_candidate_pages: int = 0
    multi_column_pages: int = 0
    running_lines: dict[str, int] = field(default_factory=dict)
    title_is_running_line: bool = False
    stored_title_is_running_line: bool = False
    authors_read_as_prose: bool = False
    stored_authors_read_as_prose: bool = False
    repeated_pages: list[int] = field(default_factory=list)
    hyphen_splits: int = 0
    hyphen_examples: list[str] = field(default_factory=list)
    tables_as_text_pages: list[int] = field(default_factory=list)
    passes: dict[str, PassResult] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _timed(report: DocumentReport, name: str, fn: Callable[[], T]) -> T | None:
    """Run one pass, recording its time and, instead of raising, its error."""
    start = time.perf_counter()
    try:
        return fn()
    except Exception as e:  # a failing pass is a finding of the report, not its end
        report.passes[name] = PassResult(time.perf_counter() - start, f"{type(e).__name__}: {e}")
        return None
    finally:
        if name not in report.passes:
            report.passes[name] = PassResult(time.perf_counter() - start)


def read_text(path: Path) -> tuple[str, str]:
    """The document's markdown and where it came from, writing nothing: a fresh cache is read,
    anything else is extracted in memory with the document's own spacing choices (ADR-056)."""
    from doc_assistant.extractors import extract_to_markdown
    from doc_assistant.ingest.cache import get_cache_path, is_cache_fresh, rejoin_overrides

    cached = get_cache_path(path)
    if is_cache_fresh(path, cached):
        return cached.read_text(encoding="utf-8"), "cache"
    keep, spacing = rejoin_overrides(path)
    return extract_to_markdown(path, keep_spaced=keep, spacing=spacing), "extracted"


def _multi_column_pages(pdf_path: str) -> int:
    """Pages where text sits in two columns: a fifth or more of the text on each side of the
    page's middle, in blocks that do not cross it. Presence only; whether the reading order then
    broke is not measured here."""
    import pymupdf

    doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    try:
        count = 0
        for i in range(len(doc)):
            page = doc[i]
            mid = (page.rect.x0 + page.rect.x1) / 2
            left = right = total = 0
            for b in page.get_text("blocks"):  # type: ignore[no-untyped-call]
                if b[6] != 0:  # image block
                    continue
                n = len(str(b[4]).strip())
                total += n
                if b[2] < mid:
                    left += n
                elif b[0] > mid:
                    right += n
            if total and left >= 0.2 * total and right >= 0.2 * total:
                count += 1
        return count
    finally:
        doc.close()  # type: ignore[no-untyped-call]


def report_document(
    path: Path,
    *,
    stored_title: str | None = None,
    stored_authors: str | None = None,
    with_figures: bool = True,
    with_tables: bool = True,
) -> DocumentReport:
    """Run every fast-track pass on one document, read-only, and flag what looks wrong."""
    from doc_assistant.ingest import citations, figures, regions, tables
    from doc_assistant.metadata_extractor import _reads_as_prose, extract_metadata

    suffix = path.suffix.lower()
    rep = DocumentReport(path=str(path), name=path.name, format=suffix.lstrip("."))
    rep.stored_title, rep.stored_authors = stored_title, stored_authors

    got = _timed(rep, "text", lambda: read_text(path))
    if got is None:
        return rep
    markdown, rep.text_source = got
    segments = page_segments(markdown)
    rep.pages = len(segments) if segments[0][0] else 0
    rep.chars = len(markdown)
    rep.empty_pages = [p for p, t in segments if p and not t.strip()]

    meta = _timed(rep, "metadata", lambda: extract_metadata(markdown, filename=path.name))
    if meta is not None:
        rep.title, rep.authors, rep.year, rep.doi = meta.title, meta.authors, meta.year, meta.doi

    refs = _timed(rep, "references", lambda: citations.extract_from_markdown(path.name, markdown))
    if refs is not None:
        rep.references_found = refs.references_section_found
        rep.reference_entries = refs.count
        rep.reference_entries_without_year = sum(1 for c in refs.citations if c.year is None)
        rep.reference_notes = list(refs.notes)

    rep.running_lines = dict(sorted(running_lines(segments).items(), key=lambda kv: -kv[1]))
    rep.title_is_running_line = title_is_running_line(rep.title, rep.running_lines)
    rep.stored_title_is_running_line = title_is_running_line(stored_title, rep.running_lines)
    rep.authors_read_as_prose = bool(rep.authors) and _reads_as_prose(rep.authors or "")
    rep.stored_authors_read_as_prose = bool(stored_authors) and _reads_as_prose(
        stored_authors or ""
    )
    rep.repeated_pages = repeated_pages(segments)
    splits = hyphen_splits(markdown)
    rep.hyphen_splits, rep.hyphen_examples = len(splits), splits[:EXAMPLES_KEPT]

    if suffix != ".pdf":
        return rep  # figures, tables and columns have no path for other formats yet
    pdf = str(path)
    if with_figures:
        found = _timed(rep, "figures", lambda: figures.detect_figure_regions(pdf))
        if found is not None:
            rep.figures = len(found)
            rep.figures_caption_only = sum(
                1 for f in found if f.extraction_method == "caption_only"
            )
            rep.figures_without_caption = sum(1 for f in found if not f.caption)
    if with_tables:
        cand = _timed(rep, "table_pages", lambda: regions.table_candidate_pages(pdf))
        if cand is not None:
            rep.table_candidate_pages = len(cand)
            found_tables = _timed(
                rep, "tables", lambda: tables.extract_tables_from_pages(pdf, cand)
            )
            if found_tables is not None:
                rep.tables = len(found_tables)
                with_table = {t.page for t in found_tables}
                rep.tables_as_text_pages = sorted(
                    p for p in table_mentions(segments) if p not in with_table
                )
    columns = _timed(rep, "columns", lambda: _multi_column_pages(pdf))
    rep.multi_column_pages = columns or 0
    return rep
