"""Table extraction (Phase 6 / Feature 4a) — pdfplumber pass, spliced into markdown.

A post-ingest enrichment layer (see "Enrichment-Layer Pattern" in
``decisions.md``). The primary ingest extractor (PyMuPDF4LLM) flattens
tables into ambiguous text; this module re-reads the *source PDF*
with pdfplumber, renders each detected table as a GitHub-flavoured
markdown table, and splices the result back into the document's cached
``.md`` file.

Why splice rather than sidecar (the one allowed exception to "sidecar by
default"): tables are *text-shaped* (rows x cells). Splicing markdown back
into the cache preserves the "open the .md and see everything" property,
and on the next re-ingest each table becomes retrievable chunk content
with its row/column structure intact. Binary artifacts (figures) stay
sidecar — that is Feature 4b.

Detection lives elsewhere: ``regions.py`` classifies each page (table /
chart / photo / figure / text) from caption + curve-density + image-area
signals, and this module extracts structure only on the pages it routes to
tables. That split keeps figures and charts — which geometric detectors
happily mistake for tables — out of the extraction path.

Design choices
--------------
* **Pure & testable.** Extraction (``extract_tables`` / ``extract_tables_
  from_pages``) opens the PDF; rendering and splicing are plain string/
  dataclass transforms, mirroring ``citations.extract_from_markdown``.
* **Idempotent splice.** All extracted tables live inside one demarcated
  block (``_BLOCK_BEGIN`` … ``_BLOCK_END``) appended to the markdown.
  Re-splicing *replaces* the block — never appends a second one — so the
  CLI's ``--force`` path is safe and ``splice == splice∘splice``.
* **Per-table marker.** Each table carries
  ``<!-- table-extracted-by: pdfplumber page=N table=M -->`` for
  traceability and re-runnability, as specified in ``decisions.md``.
* **The enrichment never mutates the chunk store.** It writes the cache
  only; retrieval reflects the tables after the next ``ingest`` run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Minimum shape for a region to count as a real table rather than a
# layout artifact. Tunable here; deliberately not a config knob yet.
MIN_ROWS = 2
MIN_COLS = 2

# pdfplumber's default detector readily mis-classifies a column of prose
# (e.g. a Methods section in a single-column paper) as a tall, narrow
# "table" with one enormous run-together cell. Two guards reject that:
#   * MAX_CELL_CHARS — a genuine data cell is short; prose blocks are huge.
#   * a real table has data in at least MIN_COLS distinct columns, so we
#     require that many columns to carry non-empty content.
MAX_CELL_CHARS = 500

# Splice block markers. The whole block is bounded so it can be replaced
# atomically on re-run; the per-table marker is the decisions.md-specified
# traceability tag.
_BLOCK_BEGIN = "<!-- tables:pdfplumber:begin -->"
_BLOCK_END = "<!-- tables:pdfplumber:end -->"
_BLOCK_HEADING = "## Tables extracted by pdfplumber"
_TABLE_MARKER = "<!-- table-extracted-by: pdfplumber page={page} table={index} -->"

# Matches a whole spliced block (including surrounding blank lines) so it
# can be stripped/replaced cleanly. DOTALL so it spans the table rows.
_BLOCK_RE = re.compile(
    r"\n*" + re.escape(_BLOCK_BEGIN) + r".*?" + re.escape(_BLOCK_END) + r"\n*",
    re.DOTALL,
)


# ============================================================
# Dataclass
# ============================================================


@dataclass
class ExtractedTable:
    """One table detected on one PDF page.

    ``rows`` is a list of rows, each a list of cell strings. Cells are
    normalised: ``None`` → ``""`` and internal newlines collapsed to
    spaces (markdown table cells cannot contain raw newlines).
    """

    page: int  # 1-based page number
    index: int  # 1-based table number within the document
    rows: list[list[str]]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)


# ============================================================
# Cell / table rendering
# ============================================================


def _clean_cell(value: Any) -> str:
    """Normalise a pdfplumber cell to a single-line markdown-safe string."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("|", r"\|")  # escape the column separator
    return re.sub(r"\s+", " ", text).strip()


def _normalise_rows(raw_rows: list[list[Any]]) -> list[list[str]]:
    """Clean every cell and pad ragged rows to a uniform column count."""
    cleaned = [[_clean_cell(c) for c in row] for row in raw_rows if row is not None]
    width = max((len(r) for r in cleaned), default=0)
    return [r + [""] * (width - len(r)) for r in cleaned]


def _is_meaningful(rows: list[list[str]]) -> bool:
    """Keep only genuine data tables; reject layout artifacts and prose.

    Filters, in order: minimum shape, not all-empty, no prose-sized cell,
    and data spread across at least ``MIN_COLS`` non-empty columns (a
    single column of text dressed up as a table fails this).
    """
    width = max((len(r) for r in rows), default=0)
    if len(rows) < MIN_ROWS or width < MIN_COLS:
        return False
    if not any(cell for row in rows for cell in row):
        return False
    if any(len(cell) > MAX_CELL_CHARS for row in rows for cell in row):
        return False
    non_empty_cols = sum(
        1 for col in range(width) if any(col < len(row) and row[col] for row in rows)
    )
    return non_empty_cols >= MIN_COLS


def render_table_markdown(table: ExtractedTable) -> str:
    """Render one ``ExtractedTable`` as a marked GitHub-flavoured markdown table.

    The first row is treated as the header. If the source had a single
    row it would have been filtered out before reaching here, so a
    header + separator is always well-formed.
    """
    marker = _TABLE_MARKER.format(page=table.page, index=table.index)
    width = table.n_cols
    rows = [r + [""] * (width - len(r)) for r in table.rows]

    header = rows[0]
    body = rows[1:]
    lines = [
        marker,
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


# ============================================================
# Splice / strip (idempotent)
# ============================================================


def has_spliced_tables(markdown: str) -> bool:
    """True if a pdfplumber table block has already been spliced in."""
    return _BLOCK_BEGIN in markdown


def strip_spliced_tables(markdown: str) -> str:
    """Remove any existing spliced table block. No-op if none present."""
    return (
        _BLOCK_RE.sub("\n", markdown).rstrip() + "\n" if has_spliced_tables(markdown) else markdown
    )


def splice_tables(markdown: str, tables: list[ExtractedTable]) -> str:
    """Return ``markdown`` with the table block replaced by ``tables``.

    Idempotent: any pre-existing block is stripped first, so calling this
    twice with the same tables yields the same output. An empty ``tables``
    list strips the block and adds nothing.
    """
    base = strip_spliced_tables(markdown).rstrip()
    if not tables:
        return base + "\n" if base else ""

    parts = [_BLOCK_BEGIN, "", _BLOCK_HEADING, ""]
    for table in tables:
        parts.append(render_table_markdown(table))
        parts.append("")
    parts.append(_BLOCK_END)
    block = "\n".join(parts)
    return f"{base}\n\n{block}\n" if base else block + "\n"


# ============================================================
# Extraction (impure — opens the PDF)
# ============================================================


def extract_tables_from_pages(pdf_path: str, pages: list[int]) -> list[ExtractedTable]:
    """Extract meaningful tables, but only from the given (1-based) pages.

    Lazy-imports pdfplumber. Confining extraction to caption-gated pages is
    what keeps figures and prose out; the per-table content guards in
    ``_is_meaningful`` are the second line of defence.
    """
    import pdfplumber

    wanted = set(pages)
    out: list[ExtractedTable] = []
    index = 0
    if not wanted:
        return out
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if page_number not in wanted:
                continue
            try:
                raw_tables = page.extract_tables()
            except Exception as e:  # pragma: no cover - pdfplumber page-level failure
                log.warning("pdfplumber failed on page %d of %s: %s", page_number, pdf_path, e)
                continue
            for raw in raw_tables or []:
                rows = _normalise_rows(raw)
                if not _is_meaningful(rows):
                    continue
                index += 1
                out.append(ExtractedTable(page=page_number, index=index, rows=rows))
    return out


def extract_tables(pdf_path: str) -> list[ExtractedTable]:
    """Extract data tables from a PDF: classify pages, then extract.

    Two passes: ``regions.analyze_pages`` classifies each page (caption +
    curve-density + image-area) and returns the table-candidate pages;
    pdfplumber extracts structure on just those. Pages classified as chart
    or photo — even with a stray table caption — are excluded. A document
    with no table page yields ``[]``.
    """
    from doc_assistant.regions import table_candidate_pages

    return extract_tables_from_pages(pdf_path, table_candidate_pages(pdf_path))
