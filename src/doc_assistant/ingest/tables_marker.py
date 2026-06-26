"""Phase 6 / Feature 4a — parse + splice high-fidelity Marker tables.

Marker (the chosen table engine, run isolated out-of-process — see
``scripts/eval_marker_tables.py`` and ``docs/specs/feature-4a-marker-table-ingest.md``)
emits paginated markdown for the caption-gated candidate pages. This module is the
**pure** half of the ingest path: it parses Marker's markdown into per-page table
blocks and splices them **inline at each table's page region** in the cached ``.md``,
de-duping pymupdf4llm's lossy inline table in the same move. Running Marker (the
subprocess) is the CLI's job; this module never shells out.

Fidelity: Marker's tables carry ``<br>`` multi-row cells and bold — we keep the raw
markdown block verbatim rather than round-tripping through rows (which would drop it).

Idempotent: a re-run strips the prior ``<!-- table:marker:* -->`` blocks and re-splices
(``splice == splice∘splice``). Page-anchored via the cache's ``<!-- page:N -->`` markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tables import MAX_CELL_CHARS, MIN_COLS, strip_spliced_tables

# Page markers the extractor wrote into the cache (mirrors ingest.PAGE_MARKER).
_PAGE_MARKER_RE = re.compile(r"<!--\s*page:(\d+)\s*-->")

# Marker's --paginate_output page separator. NOTE: confirm the exact form against
# the pinned marker-pdf version at build time (see the spec's build-time
# confirmations). We only use it to *split* pages — page attribution comes from the
# known requested page order — so a near-miss degrades gracefully (one section).
_PAGE_DELIM_RE = re.compile(r"\n*\{?\d+\}?-{6,}\n*")

# A GFM separator row: |---|:--:|---| etc.
_SEP_ROW_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")

# A table caption begins a line with "Table N" (optionally headed/bolded). The splice
# anchors the grid right after it so the caption — the natural query magnet — and the
# numeric grid stay in one retrievable parent (see ``_place_block_in_span``).
_CAPTION_RE = re.compile(r"(?im)^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:\*+[ \t]*)?Table[ \t]+\d+\b")

# Idempotent splice-block wrapper (per page).
_BLOCK_BEGIN = "<!-- table:{engine}:page={page}:begin -->"
_BLOCK_END = "<!-- table:{engine}:page={page}:end -->"
_MARKER_BLOCK_RE = re.compile(
    r"\n*<!--\s*table:marker:page=\d+:begin\s*-->.*?<!--\s*table:marker:page=\d+:end\s*-->\n*",
    re.DOTALL,
)

# The atomic span of one spliced per-page table block (begin..end), without the
# surrounding-newline padding ``_MARKER_BLOCK_RE`` uses for stripping. The chunker
# (``ingest.build_parent_child_chunks``) keeps this whole so a wide table's header
# row and numeric data rows never split across parents. Engine-generic: matches any
# ``table:<engine>:page=N`` block following the ``_BLOCK_BEGIN`` convention.
TABLE_BLOCK_RE = re.compile(
    r"<!--\s*table:\w+:page=\d+:begin\s*-->.*?<!--\s*table:\w+:page=\d+:end\s*-->",
    re.DOTALL,
)


@dataclass
class MarkerTable:
    """One Marker-extracted table, kept as raw markdown to preserve fidelity."""

    page: int  # 1-based page number
    index: int  # 1-based order within the document
    markdown: str  # the raw GFM block (with <br>/bold intact)


# ============================================================
# GFM table detection (line-based; preserves raw blocks)
# ============================================================


def _gfm_table_line_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Return [start, end) line ranges of GFM tables (a pipe run with a separator row)."""
    ranges: list[tuple[int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        if "|" in lines[i]:
            j = i
            while j < n and "|" in lines[j]:
                j += 1
            if any(_SEP_ROW_RE.match(lines[k]) for k in range(i, j)):
                ranges.append((i, j))
            i = j
        else:
            i += 1
    return ranges


def _is_meaningful_md(block: str) -> bool:
    """Keep genuine data tables: a separator row, >=2 columns, >=1 data row, no prose cell."""
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) < 3:  # header + separator + >=1 data row
        return False
    if not any(_SEP_ROW_RE.match(ln) for ln in lines):
        return False
    header_cols = len([c for c in lines[0].split("|") if c.strip()])
    if header_cols < MIN_COLS:
        return False
    cells = [c.strip() for ln in lines for c in ln.split("|")]
    return not any(len(c) > MAX_CELL_CHARS for c in cells)


def _extract_gfm_tables(text: str) -> list[str]:
    """Return the raw markdown of each meaningful GFM table block in ``text``."""
    lines = text.split("\n")
    blocks = ["\n".join(lines[i:j]) for i, j in _gfm_table_line_ranges(lines)]
    return [b for b in blocks if _is_meaningful_md(b)]


def _strip_gfm_tables_text(text: str) -> str:
    """Remove *all* GFM table blocks from ``text`` (the pymupdf4llm lossy twin)."""
    lines = text.split("\n")
    drop: set[int] = set()
    for i, j in _gfm_table_line_ranges(lines):
        drop.update(range(i, j))
    return "\n".join(ln for k, ln in enumerate(lines) if k not in drop)


# ============================================================
# Parse Marker's paginated markdown
# ============================================================


def parse_marker_tables(marker_markdown: str, page_numbers: list[int]) -> list[MarkerTable]:
    """Parse paginated Marker markdown into per-page table blocks.

    ``page_numbers`` is the ordered 1-based candidate pages passed to Marker via
    ``--page_range``. Page attribution comes from this order (not from parsing the
    delimiter's page id), so it is robust to Marker's page-numbering semantics.
    """
    sections = _PAGE_DELIM_RE.split(marker_markdown)
    sections = [s for s in sections if s.strip()]
    if not sections:
        return []
    # Map section i -> the i-th requested page (clamp if counts disagree).
    tables: list[MarkerTable] = []
    order = 1
    for i, section in enumerate(sections):
        page = page_numbers[i] if i < len(page_numbers) else page_numbers[-1]
        for block in _extract_gfm_tables(section):
            tables.append(MarkerTable(page=page, index=order, markdown=block.strip()))
            order += 1
    return tables


# ============================================================
# Splice — page-anchored inline replacement (de-dup + placement)
# ============================================================


def has_marker_tables(markdown: str) -> bool:
    return bool(_MARKER_BLOCK_RE.search(markdown))


def strip_marker_tables(markdown: str) -> str:
    """Remove all spliced Marker blocks (idempotency helper)."""
    return _MARKER_BLOCK_RE.sub("\n\n", markdown).strip() + "\n"


def strip_pdfplumber_block(markdown: str) -> str:
    """Remove the pdfplumber append-block so the Marker pass supersedes it."""
    return strip_spliced_tables(markdown)


def _render_block(tables: list[MarkerTable], engine: str, page: int) -> str:
    begin = _BLOCK_BEGIN.format(engine=engine, page=page)
    end = _BLOCK_END.format(engine=engine, page=page)
    body = "\n\n".join(t.markdown for t in tables)
    return f"{begin}\n{body}\n{end}"


def _page_span(markdown: str, page: int) -> tuple[int, int] | None:
    """Char span of page ``page`` in the cache: [<!-- page:N -->, next page marker)."""
    start_m = re.search(rf"<!--\s*page:{page}\s*-->", markdown)
    if start_m is None:
        return None
    start = start_m.end()
    next_m = _PAGE_MARKER_RE.search(markdown, start)
    end = next_m.start() if next_m else len(markdown)
    return start, end


def _place_block_in_span(span_text: str, block: str) -> str:
    """Return ``span_text`` with ``block`` placed at its table's caption.

    The block is attached to the caption with a **single** newline (no blank line in
    between) so the chunker keeps caption + grid in one parent — the caption is the
    natural query magnet, the grid holds the values, and they must retrieve together.
    If no ``Table N`` caption is found in the span the block is appended at the end
    after a blank line (its own atomic parent), preserving the prior behaviour.
    """
    m = _CAPTION_RE.search(span_text)
    if m is None:
        return span_text.rstrip() + "\n\n" + block + "\n"
    para_end = span_text.find("\n\n", m.end())
    if para_end == -1:  # caption is the last paragraph in the span
        return span_text.rstrip() + "\n" + block + "\n"
    before = span_text[:para_end].rstrip()  # up to and including the caption paragraph
    after = span_text[para_end:].strip()  # remaining page prose
    return f"{before}\n{block}\n\n{after}\n" if after else f"{before}\n{block}\n"


def splice_tables_inline(
    markdown: str, tables: list[MarkerTable], *, engine: str = "marker"
) -> str:
    """Splice Marker tables inline at their page region; de-dup the lossy twin.

    For each page with tables: strip pymupdf4llm's GFM table(s) within that page's
    span only, then place the wrapped Marker block **right after the table caption**
    (``Table N: ...``), falling back to the span end when no caption is present. The
    caption is the natural query magnet, so anchoring the grid to it keeps the two in
    one retrievable parent (the chunker keeps the block whole and absorbs the attached
    caption). Pages Marker did not process are untouched. Idempotent (prior Marker
    blocks stripped first). If a page marker is missing, the block is appended at the
    document end.
    """
    markdown = strip_marker_tables(markdown)
    by_page: dict[int, list[MarkerTable]] = {}
    for t in tables:
        by_page.setdefault(t.page, []).append(t)

    # Splice highest page first so earlier-page edits don't shift later spans.
    for page in sorted(by_page, reverse=True):
        block = _render_block(by_page[page], engine, page)
        span = _page_span(markdown, page)
        if span is None:
            markdown = markdown.rstrip() + "\n\n" + block + "\n"
            continue
        start, end = span
        span_text = _strip_gfm_tables_text(markdown[start:end])
        new_span = _place_block_in_span(span_text, block)
        markdown = markdown[:start] + new_span + markdown[end:]
    return markdown
