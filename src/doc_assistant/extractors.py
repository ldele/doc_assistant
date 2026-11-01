"""Format-specific extractors. Each returns markdown text."""

import itertools
import re
import statistics
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)

# PyMuPDF4LLM emits a textual placeholder line for every inline image / vector graphic it
# declines to render, e.g. ``**==> picture [29 x 29] intentionally omitted <==**``. These
# carry no value in the chunk text (figures are handled by the Feature-4b sidecar) and
# pollute both retrieval and text-derived enrichment (KI-14). Anchor to the ``==> … <==``
# frame — NOT the word "picture" — so any object type PyMuPDF4LLM frames the same way is
# stripped. Whole line, tolerant of ``*``/``**`` emphasis and leading/trailing horizontal
# whitespace; ``[^\S\n]`` matches spaces/tabs but not newlines, so ``.`` stays on one line.
_IMAGE_PLACEHOLDER_LINE = re.compile(
    r"^[^\S\n]*\*{0,2}[^\S\n]*==>.*?<==[^\S\n]*\*{0,2}[^\S\n]*$",
    re.MULTILINE,
)
# Collapse the blank-line run a removed placeholder leaves behind so paragraph spacing
# stays single. Only applied when a placeholder was actually removed (see below).
_COLLAPSE_BLANK_LINES = re.compile(r"\n{3,}")


def count_image_placeholders(md: str) -> int:
    """Number of PyMuPDF4LLM image-placeholder lines in ``md`` (KI-14)."""
    return len(_IMAGE_PLACEHOLDER_LINE.findall(md))


def strip_image_placeholders(md: str) -> str:
    """Remove PyMuPDF4LLM image / vector-graphic placeholder lines from markdown.

    Deletes any whole line framed by ``==> … <==`` (tolerating ``*``/``**`` emphasis and
    surrounding whitespace), then collapses the blank-line run left behind so paragraph
    spacing stays single.

    A **no-op when no placeholder is present**: text with none is returned byte-for-byte
    unchanged, so unaffected documents keep their content hash and are never needlessly
    re-ingested. The blank-line collapse fires only when a placeholder was removed, so its
    (benign) side effect is confined to documents that are already changing. The transform
    is idempotent — a second pass finds nothing to strip. See KI-14.
    """
    if not _IMAGE_PLACEHOLDER_LINE.search(md):
        return md
    stripped = _IMAGE_PLACEHOLDER_LINE.sub("", md)
    return _COLLAPSE_BLANK_LINES.sub("\n\n", stripped)


# Everything that is markup or scaffolding rather than content, for the "did the markdown keep the
# page?" comparison below: page markers, image placeholders, heading hashes, table pipes, rules.
_PAGE_MARKER = re.compile(r"<!--\s*page:\d+\s*-->")
_NON_CONTENT = re.compile(r"[#*|\-\s]+")

# A page whose markdown retains less than this share of its text layer has been LOST, not merely
# reformatted, and the text layer is used instead.
#
# **Structural, not corpus-tuned.** The two populations do not overlap or even come close
# (measured 2026-08-07, this corpus): 14 healthy PDFs over 28 pages kept **97.3%-108.9%** (ratios
# slightly exceed 1.0 because markdown adds structure the raw text lacks), while the three
# scan-with-text-layer documents kept **0.0%-3.2%**. Any value in that ~94-point gap selects the
# same pages; 0.5 is the middle of it. If a future corpus lands *between* these populations, that
# is a real signal to re-measure, not to nudge the constant.
_TEXT_LAYER_KEPT_MIN = 0.5


def _substantive_len(text: str) -> int:
    """Length of ``text`` ignoring whitespace and pure-markup characters."""
    return len(_NON_CONTENT.sub("", _PAGE_MARKER.sub("", text)))


def extract_pdf_pymupdf(
    pdf_path: Path,
    keep_spaced: frozenset[str] = frozenset(),
    spacing: Mapping[str, str] | None = None,
    record: list[dict[str, Any]] | None = None,
) -> str:
    """PDF extraction with page markers preserved as HTML comments.

    **Closes letter-spacing the page does not show** (ADR-056): each page's markdown goes through
    :func:`apply_letter_spacing` with the runs :func:`letter_spacing_page_runs` reads from that
    page's glyphs. ``keep_spaced`` and ``spacing`` are the document's ignore-list entries: runs
    left as extracted, and runs spaced the user's way. Every run found is appended to ``record``
    (ADR-056 point 7) with its page and what was written, so the app can show what was done to
    this document's text; the counts are logged per document.

    **Falls back to the page's text layer when the markdown conversion loses it.** A scanned page
    that carries an invisible OCR text layer behind a full-page image is rendered by PyMuPDF4LLM as
    a *picture placeholder* — the text never reaches the markdown — and `strip_image_placeholders`
    (KI-14) then removes even that, leaving a heading marker or nothing at all. The document
    survives ingest with a handful of empty chunks and becomes silently unretrievable.

    Measured on this corpus (2026-08-07): three documents lost **97-100%** of their text this way —
    `hubel_wiesel_1959` kept 86 characters of 45,995, `hodgkin_huxley_1952` 3,219 of 88,754,
    `hebb_1949` 5,117 of 776,162 — and together they accounted for **6 of the 7** retrieval misses
    on the private eval set. Their text layers are good (88-94% word-like across sampled pages);
    nothing needed OCR, the extractor was simply discarding what was already there.

    The fallback stays **inside this one extraction path** — the same principle that decided
    ADR-039 — so a recovered page flows onward through page markers, chunking, table splicing and
    the health scorer exactly like any other. It cannot fire on a healthy page: see
    ``_TEXT_LAYER_KEPT_MIN``.
    """
    import pymupdf
    import pymupdf4llm

    _keep_ligature_letters_in_table_cells()
    doc = pymupdf.open(str(pdf_path))  # type: ignore[no-untyped-call]
    parts: list[str] = []
    found: list[dict[str, Any]] = []
    for page_num in range(len(doc)):
        # Mark the start of each page so chunks can be tagged
        parts.append(f"\n<!-- page:{page_num + 1} -->\n")
        page_md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
        page_md = _recover_lost_page(doc, page_num, page_md)
        try:
            runs = letter_spacing_page_runs(doc[page_num])
        except Exception as e:
            # A repair that fails the document would be worse than the spacing it repairs.
            log.warning(
                "letter_spacing_page_skipped",
                file=pdf_path.name,
                page=page_num + 1,
                error=str(e),
            )
            runs = []
        page_md, outcomes = apply_letter_spacing(page_md, runs, keep_spaced, spacing or {})
        found.extend({**run, "page": page_num + 1} for run in outcomes)
        parts.append(page_md)
    doc.close()  # type: ignore[no-untyped-call]
    if record is not None:
        record.extend(found)
    if found or keep_spaced or spacing:
        states = [run["state"] for run in found if run["in_text"]]
        log.info(
            "letter_spacing_closed",
            file=pdf_path.name,
            closed=states.count("closed"),
            kept=states.count("kept"),
            yours=states.count("yours"),
            maths_left=states.count("maths"),
            ignore_entries=len(keep_spaced) + len(spacing or {}),
        )
    return "\n".join(parts)


# KI-64. Importing pymupdf4llm turns PyMuPDF's quad corrections off for the whole process, and with
# them off the second letter of a ligature PyMuPDF splits back into letters ("fi" -> "f", "i") gets
# a box of zero width. Both table-cell functions keep a character only if more than half of its
# box lies inside the cell, and zero is never more than half of zero, so "fine-tuning" became
# "fne-tuning" in every table cell of 19 of 104 documents. Prose is not filtered by box and kept
# the letter. Width in points of the box a zero-width character is given at its own origin: enough
# to exist, far too little to reach a neighbouring cell. Structural, not tuned.
_HAIR = 0.01


def widen_zero_width_chars(blocks: Iterable[dict[str, Any]]) -> None:
    """Give every character of zero width a box a hair wide at its origin, in place (KI-64).

    ``blocks`` are PyMuPDF ``rawdict`` text blocks. A character whose box already has width is left
    as it is, so a second pass changes nothing; a block without characters (an OCR page's ``dict``
    blocks) is passed over."""
    for block in blocks:
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                for char in span.get("chars", ()):
                    x0, y0, x1, y1 = char["bbox"]
                    if x1 <= x0:
                        x = char["origin"][0]
                        char["bbox"] = (x, y0, x + _HAIR, y1)


class _WidenedTextPage:
    """A PyMuPDF text page whose ``rawdict`` gives zero-width characters a box (KI-64)."""

    def __init__(self, textpage: Any) -> None:
        self._textpage = textpage

    def extractRAWDICT(self, *args: Any, **kwargs: Any) -> Any:  # PyMuPDF's name
        raw = self._textpage.extractRAWDICT(*args, **kwargs)
        widen_zero_width_chars(raw["blocks"])
        return raw


class _CellTextOfTextPage:
    """``pymupdf.table.extract_cells`` reading a widened text page: the cell text pymupdf4llm takes
    when its layout module is not installed (KI-64)."""

    def __init__(self, original: Callable[..., str]) -> None:
        self.original = original

    def __call__(self, textpage: Any, cell: Any, markdown: bool = False) -> str:
        return self.original(_WidenedTextPage(textpage), cell, markdown=markdown)


class _CellTextOfBlocks:
    """``pymupdf4llm.helpers.utils.extract_cells`` on widened blocks: the cell text of
    pymupdf4llm's layout path, the one this app runs (KI-64)."""

    def __init__(self, original: Callable[..., str]) -> None:
        self.original = original

    def __call__(
        self, table_blocks: Any, cell: Any, markdown: bool = False, ocrpage: bool = False
    ) -> str:
        widen_zero_width_chars(table_blocks)
        return self.original(table_blocks, cell, markdown=markdown, ocrpage=ocrpage)


def _keep_ligature_letters_in_table_cells() -> None:
    """Wrap both table-cell functions so a ligature's second letter is kept (KI-64). Idempotent.

    Neither library's code is copied or edited: each function is handed characters whose boxes it
    can measure. Called from :func:`extract_pdf_pymupdf`, so it is part of the PDF extraction
    fingerprint, and changing it re-extracts every cached PDF (KI-40). If an upgrade moves either
    function, nothing is wrapped and the miss is logged rather than raised;
    ``test_the_table_cell_functions_keep_their_shape`` fails first."""
    import pymupdf.table
    from pymupdf4llm.helpers import utils

    for module, wrapper in ((pymupdf.table, _CellTextOfTextPage), (utils, _CellTextOfBlocks)):
        original = getattr(module, "extract_cells", None)
        if original is None:
            log.warning("ki64_cell_text_not_wrapped", module=module.__name__)
        elif not isinstance(original, wrapper):
            setattr(module, "extract_cells", wrapper(original))  # noqa: B010 - an untyped module


class _TextPage(Protocol):
    def get_text(self) -> str: ...


class _PagedDoc(Protocol):
    """Just enough of a PyMuPDF ``Document`` to read a page's text layer.

    Structural, so the real ``pymupdf.Document`` satisfies it without an import and a stub can too
    — which is what lets the recovery be tested without a PDF fixture.
    """

    def __getitem__(self, index: int, /) -> _TextPage: ...


def _recover_lost_page(doc: _PagedDoc, page_num: int, page_md: str) -> str:
    """Return ``page_md``, or the page's raw text layer when the markdown lost it.

    Kept separate from :func:`extract_pdf_pymupdf` so it is testable without a real PDF, and so the
    "is this page lost?" judgement has one home. Any failure to read the text layer returns the
    markdown unchanged — a recovery that raises would be worse than the gap it fixes.
    """
    try:
        raw = doc[page_num].get_text().strip()  # type: ignore[index]
    except Exception:
        return page_md
    raw_len = _substantive_len(raw)
    if raw_len == 0:
        return page_md  # genuinely no text layer (a true scan) — OCR territory, not this
    if _substantive_len(page_md) >= _TEXT_LAYER_KEPT_MIN * raw_len:
        return page_md
    return raw


# ADR-056. PyMuPDF writes a space wherever two glyphs are drawn apart and marks each one it
# writes (`char["synthetic"]`), so a title set with letter-spacing comes out as "N E U R O N S".
# These close the spaces the page does not show, from evidence in the PDF only. Each constant is
# a property of type, not of this library (all measured in em of the smaller glyph):
#   _HALF_SPACE_EM — half a quarter-em word space: a word break in tracked text is the tracking
#                    plus a space, a letter gap the tracking alone, and this sits between them;
#   _TOUCH_EM      — glyphs closer than this touch; no face sets a word space that narrow;
#   _BASELINE_EM   — a sub- or superscript sits well off its line; one line of letters does not.
_HALF_SPACE_EM = 0.125
_TOUCH_EM = 0.05
_BASELINE_EM = 0.1
#: Marks that may close a letter-spaced word ("NEURONS:"). A string, not a set: the extraction
#: fingerprint hashes this value's repr, and a set's repr order changes between processes.
_CLOSING_MARKS = ":;,.!?\u2019"
#: An ignore-list entry that keeps every run of a document spaced (ADR-056).
KEEP_ALL_SPACED = "*"
#: A page one image covers at least this much of is a scan, and its text an OCR layer whose glyph
#: positions are estimates, so the rule leaves it alone. Structural: measured 2026-10-24 on 3,341
#: pages, scans sit at 0.9 and above, every other page at 0.7 or below, and no page between; this
#: is the middle of that gap, as `_TEXT_LAYER_KEPT_MIN` sits in its own.
_SCANNED_PAGE_MIN = 0.8
#: Layout boxes (pymupdf4llm's own analysis) whose content is not running text: the rule leaves
#: them alone, however their glyphs are spaced.
_NOT_TEXT_BOXES = ("formula", "picture", "table")
#: Glyphs that make a run of touching glyphs maths rather than a word: it stays as extracted.
#: Not "/", "-", "." or ":", which URLs are made of.
_MATH_GLYPHS = "=+\u2212\u00d7\u00b7\u2219<>\u2264\u2265\u00b1\u2211\u222b\u221a\u2217^"
#: A URL or a piece of one cut at a line end: a scheme, ``www.``, or a domain before a path.
_URL_SHAPE = re.compile(r"://|\bwww\.|[a-z0-9]\.[a-z]{2,}/", re.IGNORECASE)
#: What may sit directly before a run in the markdown: a space, or the emphasis pymupdf4llm wraps
#: around bold, italic, monospaced or struck-out text. Anything else is another word's tail.
_RUN_START = r"(?<![^\s*_`~])"

#: A join, as ``(spaced, joined, kind)``: the run as PyMuPDF reads it with one space between
#: glyphs, the run as the page shows it, and its kind: ``"tracked"`` (letters drawn apart),
#: ``"invisible"`` (space glyphs between touching glyphs), ``"url"`` (an invisible run in a URL),
#: or ``"maths"`` (an invisible run of maths, whose joined form is its spaced form: left alone).
SpacingJoin = tuple[str, str, str]
#: One token of a line: its glyphs, and the spaces that follow them (``rawdict`` characters).
_GlyphToken = tuple[list[dict[str, Any]], list[dict[str, Any]]]


def _glyph_tokens(chars: list[dict[str, Any]]) -> list[_GlyphToken]:
    """A line's glyphs split at spaces: ``[(glyphs, the spaces that follow them)]``."""
    tokens: list[_GlyphToken] = []
    for ch in chars:
        if ch["c"].isspace():
            if tokens:
                tokens[-1][1].append(ch)
        elif not tokens or tokens[-1][1]:
            tokens.append(([ch], []))
        else:
            tokens[-1][0].append(ch)
    return tokens


def letter_spacing_joins(line: Mapping[str, Any]) -> list[SpacingJoin]:
    """The letter-spaced runs in one PyMuPDF ``rawdict`` text line, and how each closes (ADR-056).

    A run is three or more one-glyph tokens in a row with at least three letters. It closes as
    **tracked** text when its letter gaps are spaces PyMuPDF added, it is all letters (a closing
    mark may follow a letter), its glyphs share a baseline, its spacing is narrower than its
    letters, and its neighbours on the line sit at a word break: further than its spacing plus
    half a word space, or behind a space the PDF itself holds. Gaps up to its spacing plus half a
    word space close; wider ones stay one space. Failing that, any gap in the run under
    ``_TOUCH_EM`` closes as **invisible**: a space glyph between glyphs that touch.

    An invisible run in a URL (one that holds a URL shape, or follows one on its line) is a
    **url**; one holding a maths glyph and no URL is **maths**, and stays as extracted.

    What stays, on purpose: a space glyph the reader can see (``a r t i c l e``), an OCR layer's
    word fragments (their neighbours sit as close as their letters), maths, sub- and superscripts.
    A spaced word in a PDF that holds no space glyphs at all also stays: geometry cannot tell it
    from one-letter words."""
    return [join for join, _ in _line_runs(line)]


def _line_runs(line: Mapping[str, Any]) -> list[tuple[SpacingJoin, tuple[float, ...]]]:
    """:func:`letter_spacing_joins`, each with the box around its glyphs."""
    chars = [{**c, "size": s["size"]} for s in line.get("spans", ()) for c in s.get("chars", ())]
    dx, dy = line.get("dir", (1.0, 0.0))

    def extent(c: Mapping[str, Any]) -> tuple[float, float]:
        x0, y0, x1, y1 = c["bbox"]
        a, b = x0 * dx + y0 * dy, x1 * dx + y1 * dy
        return min(a, b), max(a, b)

    def gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        return (extent(b)[0] - extent(a)[1]) / (min(a["size"], b["size"]) or 1.0)

    tokens = _glyph_tokens(chars)
    runs: list[tuple[SpacingJoin, tuple[float, ...]]] = []
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and len(tokens[j][0]) == 1:
            j += 1
        if j - i >= 3:
            join = _close_run(tokens, i, j, gap, extent, (dx, dy))
            if join is not None:
                boxes = [tok[0][0]["bbox"] for tok in tokens[i:j]]
                box = (
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                )
                runs.append((join, box))
        i = max(j, i + 1)
    return runs


def _close_run(
    tokens: list[_GlyphToken],
    i: int,
    j: int,
    gap: Callable[[Mapping[str, Any], Mapping[str, Any]], float],
    extent: Callable[[Mapping[str, Any]], tuple[float, float]],
    direction: tuple[float, float],
) -> SpacingJoin | None:
    """How the run ``tokens[i:j]`` of one-glyph tokens closes, or ``None`` (rules: the caller)."""
    run = tokens[i:j]
    glyphs = [t[0][0] for t in run]
    if sum(c["c"].isalpha() for c in glyphs) < 3:
        return None
    gaps = [gap(a, b) for a, b in itertools.pairwise(glyphs)]
    letter_gaps = [
        gaps[k]
        for k in range(len(gaps))
        if run[k][1]
        and all(s["synthetic"] for s in run[k][1])
        and glyphs[k]["c"].isalpha()
        and glyphs[k + 1]["c"].isalpha()
    ]
    wordlike = all(
        c["c"].isalpha() or (k > 0 and c["c"] in _CLOSING_MARKS and glyphs[k - 1]["c"].isalpha())
        for k, c in enumerate(glyphs)
    )
    dx, dy = direction
    across = [c["origin"][0] * -dy + c["origin"][1] * dx for c in glyphs]
    one_baseline = max(across) - min(across) <= _BASELINE_EM * min(c["size"] for c in glyphs)

    close: list[bool] = []
    kind = "tracked"
    if len(letter_gaps) >= 2 and wordlike and one_baseline:
        t = statistics.median(letter_gaps)
        widths = sorted(
            (extent(c)[1] - extent(c)[0]) / (c["size"] or 1.0) for c in glyphs if c["c"].isalpha()
        )

        def word_break(spaces: list[dict[str, Any]], g: float) -> bool:
            return g > t + _HALF_SPACE_EM or any(not s["synthetic"] for s in spaces)

        edges = t < widths[len(widths) // 2]
        if i > 0:
            edges = edges and word_break(tokens[i - 1][1], gap(tokens[i - 1][0][-1], glyphs[0]))
        if j < len(tokens):
            edges = edges and word_break(run[-1][1], gap(glyphs[-1], tokens[j][0][0]))
        if edges:
            close = [g <= t + _HALF_SPACE_EM for g in gaps]
    if not any(close):
        kind = "invisible"
        close = [g < _TOUCH_EM for g in gaps]
    if not any(close):
        return None
    spaced = " ".join(c["c"] for c in glyphs)
    joined = glyphs[0]["c"] + "".join(
        ("" if shut else " ") + c["c"] for shut, c in zip(close, glyphs[1:], strict=True)
    )
    if kind == "invisible":
        before = "".join(c["c"] for c in tokens[i - 1][0]) if i > 0 else ""
        if _URL_SHAPE.search(joined) or _URL_SHAPE.search(before):
            kind = "url"
        elif any(c["c"] in _MATH_GLYPHS for c in glyphs):
            return spaced, spaced, "maths"
    return spaced, joined, kind


def letter_spacing_page_joins(page: Any) -> list[SpacingJoin]:
    """Every join on one PyMuPDF page, as ``(spaced, joined, kind)``: the runs of
    :func:`letter_spacing_page_runs` without their boxes."""
    return [(r["spaced"], r["joined"], r["kind"]) for r in letter_spacing_page_runs(page)]


def letter_spacing_page_runs(page: Any) -> list[dict[str, Any]]:
    """Every letter-spaced run on one PyMuPDF page, read with pymupdf4llm's layout-path flags.

    Each run is ``spaced``, ``joined`` and ``kind`` (see ``SpacingJoin``), its ``box`` on the page
    (x0, y0, x1, y1 in points), and the ``label`` of the layout box it sits in ("page-header",
    "text", ...; "" when it sits in none). None on a scanned page (``_SCANNED_PAGE_MIN``), and none
    inside a formula, picture or table box. The layout is only analysed on a page that has a run
    to check: it is the costly part of extraction, and few pages need it."""
    import pymupdf
    import pymupdf4llm  # noqa: F401  # activates the layout analysis extraction itself runs

    if _scanned_page(page):
        return []
    flags = (
        pymupdf.TEXT_COLLECT_STYLES
        | pymupdf.TEXT_COLLECT_VECTORS
        | pymupdf.TEXT_PRESERVE_IMAGES
        | pymupdf.TEXT_ACCURATE_BBOXES
        | pymupdf.TEXT_MEDIABOX_CLIP
        | pymupdf.TEXT_IGNORE_ACTUALTEXT
    )
    blocks = page.get_textpage(flags=flags).extractRAWDICT()["blocks"]
    runs = [run for b in blocks for line in b.get("lines", ()) for run in _line_runs(line)]
    if not runs:
        return []
    page.get_layout()
    rect = type(page.rect)  # PyMuPDF's Rect, reached through the page: it is untyped
    layout = [(rect(b[:4]), str(b[4])) for b in (page.layout_information or ())]
    skip = [box for box, label in layout if label in _NOT_TEXT_BOXES]
    found: list[dict[str, Any]] = []
    for (spaced, joined, kind), box in runs:
        where = rect(box)
        if _mostly_inside(where, skip):
            continue
        label = next((lab for b, lab in layout if _mostly_inside(where, [b])), "")
        found.append(
            {"spaced": spaced, "joined": joined, "kind": kind, "box": list(box), "label": label}
        )
    return found


def _scanned_page(page: Any) -> bool:
    """Whether one image covers at least ``_SCANNED_PAGE_MIN`` of the page."""
    area = page.rect.get_area() or 1.0
    return any(
        (type(page.rect)(info["bbox"]) & page.rect).get_area() / area >= _SCANNED_PAGE_MIN
        for info in page.get_image_info()
    )


def _mostly_inside(box: Any, boxes: Iterable[Any]) -> bool:
    """Whether more than half of ``box`` lies inside one of ``boxes`` (PyMuPDF rects)."""
    area = box.get_area()
    return area > 0 and any((box & other).get_area() > area / 2 for other in boxes)


def close_letter_spacing(
    page_md: str, joins: Iterable[SpacingJoin], keep_spaced: frozenset[str] = frozenset()
) -> tuple[str, list[SpacingJoin]]:
    """``page_md`` with each join's spaced form replaced by its joined form, and the joins applied.

    :func:`apply_letter_spacing` for bare joins, reporting only those it closed. A join whose
    spaced form is in ``keep_spaced`` (the ignore list, ADR-056) is left as it is, and so is every
    join when ``keep_spaced`` holds ``KEEP_ALL_SPACED``, and every ``"maths"`` run."""
    runs = [{"spaced": s, "joined": j, "kind": k} for s, j, k in joins]
    page_md, outcomes = apply_letter_spacing(page_md, runs, keep_spaced, {})
    return page_md, [
        (r["spaced"], r["joined"], r["kind"])
        for r in outcomes
        if r["state"] == "closed" and r["in_text"]
    ]


def user_spacing_is_valid(spaced: str, chosen: str) -> bool:
    """Whether ``chosen`` respaces ``spaced`` and changes nothing else (ADR-056 point 10).

    The same glyphs in the same order, single spaces only, none at either end: an entry that
    would change a letter is not a spacing, and the extractor never applies it."""
    return (
        bool(chosen)
        and chosen == chosen.strip(" ")
        and "  " not in chosen
        and chosen.replace(" ", "") == spaced.replace(" ", "")
    )


def apply_letter_spacing(
    page_md: str,
    runs: Iterable[Mapping[str, Any]],
    keep_spaced: frozenset[str] = frozenset(),
    spacing: Mapping[str, str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """``page_md`` with each run written as decided, and every run with what became of it.

    Text only, so it can be tested without a PDF. A run's spaced form is matched with one to
    three spaces between glyphs, as markdown writes it, once per run, inside any emphasis the
    markdown wraps around it (``**N E U**``; the emphasis stays). Each run comes back with a
    ``state`` (``closed`` by the rule, ``kept`` by the ignore list or ``KEEP_ALL_SPACED``,
    ``yours`` when the user's spacing was applied, ``maths`` left alone by the rule), the text
    now ``written`` there, and ``in_text``: False when the run never reached this markdown (a
    margin label the layout path drops). User spacing that fails :func:`user_spacing_is_valid`
    is logged and the rule decides instead. Only spaces are ever added or removed."""
    spacing = spacing or {}
    outcomes: list[dict[str, Any]] = []
    for run in runs:
        spaced, joined, kind = run["spaced"], run["joined"], run["kind"]
        chosen = spacing.get(spaced)
        if chosen is not None and not user_spacing_is_valid(spaced, chosen):
            log.warning("letter_spacing_entry_rejected", run=spaced, spacing=chosen)
            chosen = None
        if KEEP_ALL_SPACED in keep_spaced or spaced in keep_spaced:
            state, write = "kept", spaced
        elif chosen is not None:
            state, write = "yours", chosen
        elif kind == "maths":
            state, write = "maths", spaced
        else:
            state, write = "closed", joined
        pattern = re.compile(
            _RUN_START
            + r"[ \t]{1,3}".join(re.escape(g) for g in spaced.split(" "))
            + r"(?![^\W_])"
        )
        match = pattern.search(page_md)
        if match is not None and state in ("closed", "yours"):
            # Sliced in rather than `sub`bed: a replacement string reads a backslash as an escape,
            # and OCR noise can hold one.
            page_md = page_md[: match.start()] + write + page_md[match.end() :]
        elif match is not None:
            write = match.group(0)  # a word break in a spaced run can be wider than one space
        outcomes.append({**run, "state": state, "written": write, "in_text": match is not None})
    return page_md, outcomes


# Page furniture rather than content. `head` matters as much as the rest: `get_text()` reaches it,
# so a page `<title>` otherwise lands above the real `<h1>` as a bare line that nothing downstream
# can distinguish from an opening sentence.
_NON_CONTENT_TAGS = ("head", "script", "style", "nav", "footer")

# Inline elements — their boundaries sit *inside* a sentence. `get_text(separator="\n")` emits a
# newline at every tag boundary, so `must not <strong>split</strong> it` arrives as three lines;
# unwrapping them first removes the boundary rather than trying to repair the text afterwards.
# The separator itself has to stay `\n`: it is what keeps `<p>`, `<li>` and headings apart.
_INLINE_TAGS = (
    "a", "abbr", "b", "cite", "code", "em", "i", "mark",
    "q", "s", "small", "span", "strong", "sub", "sup", "u",
)  # fmt: skip


def _soup_to_markdown(soup: BeautifulSoup) -> str:
    """Reduce parsed HTML to markdown-ish text. Shared by the EPUB and HTML extractors.

    One function on purpose. These two extractors held the same rule twice and had already drifted
    apart — HTML dropped page chrome and EPUB did not — which is the shape of the ADR-013 display
    bug (2026-08-19): a rule written twice is a rule that disagrees with itself.

    Order matters. Chrome goes first so its text can never reach the output; headings are converted
    while they are still elements; inline tags are unwrapped and the tree smoothed **last**, so the
    adjacent text nodes an unwrap leaves behind are merged into one before ``get_text`` sees them.
    Without ``smooth()`` the newline simply moves from the tag boundary to the string boundary.
    """
    for tag in soup(list(_NON_CONTENT_TAGS)):
        tag.decompose()
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            tag.replace_with(f"\n{'#' * level} {tag.get_text()}\n")
    for name in _INLINE_TAGS:
        for tag in soup.find_all(name):
            tag.unwrap()
    soup.smooth()
    return str(soup.get_text(separator="\n").strip())


def extract_epub(epub_path: Path) -> str:
    """Extract EPUB to markdown by parsing inner HTML."""
    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(epub_path))
    parts: list[str] = []

    # Add title if available
    title = book.get_metadata("DC", "title")
    if title:
        parts.append(f"# {title[0][0]}\n")

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        # The generated navigation document is a manifest item like any other, so this loop would
        # otherwise emit the book's own table of contents as body prose.
        if isinstance(item, epub.EpubNav):
            continue
        text = _soup_to_markdown(BeautifulSoup(item.get_content(), "lxml"))
        if text:
            parts.append(text)

    return "\n\n".join(parts)


def extract_html(html_path: Path) -> str:
    """Extract HTML to markdown-ish text."""
    return _soup_to_markdown(
        BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    )


def extract_docx(docx_path: Path) -> str:
    """Extract DOCX to markdown."""
    from docx import Document as DocxDocument

    doc = DocxDocument(str(docx_path))
    parts: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Heuristic: detect headings from style names
        style = para.style.name.lower() if para.style else ""
        if "heading 1" in style:
            parts.append(f"# {text}")
        elif "heading 2" in style:
            parts.append(f"## {text}")
        elif "heading 3" in style:
            parts.append(f"### {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def extract_rtf(rtf_path: Path) -> str:
    """Extract rtf to markdown."""
    from striprtf.striprtf import rtf_to_text

    text = rtf_path.read_text(encoding="utf-8", errors="ignore")
    return str(rtf_to_text(text))  # type: ignore[no-untyped-call]


def extract_odt(odt_path: Path) -> str:
    """Extract odt to markdown."""
    from odf.opendocument import load
    from odf.text import P

    doc = load(str(odt_path))
    paragraphs: list[str] = []
    for p in doc.getElementsByType(P):
        para_text = "".join(node.data for node in p.childNodes if hasattr(node, "data"))
        if para_text.strip():
            paragraphs.append(para_text)
    return "\n\n".join(paragraphs)


def extract_text(path: Path) -> str:
    """Plain text or markdown -- read as-is."""
    return path.read_text(encoding="utf-8", errors="ignore")


# Dispatch table (excludes PDF which is special-cased)
_EXTRACTORS: dict[str, Callable[[Path], str]] = {
    ".epub": extract_epub,
    ".html": extract_html,
    ".htm": extract_html,
    ".docx": extract_docx,
    ".odt": extract_odt,
    ".rtf": extract_rtf,
    ".md": extract_text,
    ".txt": extract_text,
}

# All supported extensions (including PDF)
SUPPORTED_EXTENSIONS: set[str] = {".pdf", *_EXTRACTORS}

#: What actually read the bytes, per format — recorded as provenance on every document (KI-53).
#: Names the library that does the work, not the wrapper function, because that is the thing whose
#: behaviour a reader would go and check.
#:
#: Deliberately a *parallel* mapping rather than a second field on ``_EXTRACTORS``: that dict's
#: shape is walked by ``ingest.cache.extraction_fingerprint``, and a walk that trips falls back to
#: the whole-module fingerprint — which is a re-extraction of every cached document (KI-48). The
#: drift this costs is bought back by ``test_every_supported_format_names_its_extractor``.
_EXTRACTOR_NAMES: dict[str, str] = {
    ".epub": "ebooklib+bs4",
    ".html": "bs4",
    ".htm": "bs4",
    ".docx": "python-docx",
    ".odt": "odfpy",
    ".rtf": "striprtf",
    ".md": "verbatim",
    ".txt": "verbatim",
}


def extractor_name(path: Path, pdf_extractor: str = "pymupdf") -> str:
    """Name the extractor that ``extract_to_markdown`` would use for ``path``.

    Provenance, not dispatch — it must follow the same branch the extraction does, which is why
    PDFs return the configured extractor rather than a constant: that value is what selects the
    code that runs. An unsupported suffix returns ``"unknown"`` instead of raising; recording the
    absence of provenance is better than failing an ingest over a label.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_extractor
    return _EXTRACTOR_NAMES.get(suffix, "unknown")


def extract_to_markdown(
    path: Path,
    pdf_extractor: str = "pymupdf",
    keep_spaced: frozenset[str] = frozenset(),
    spacing: Mapping[str, str] | None = None,
    record: list[dict[str, Any]] | None = None,
) -> str:
    """Main entry point: extract any supported file to markdown.

    The extracted markdown is passed through ``strip_image_placeholders`` at the single
    exit so PyMuPDF4LLM image placeholders never reach the cache / chunk store (KI-14);
    the strip is a no-op for formats and documents without placeholders.

    ``keep_spaced`` and ``spacing`` are a PDF's letter-spacing ignore-list entries, and
    ``record`` collects the runs it found (ADR-056); other formats have none, because their text
    carries no spacing to misread.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # Marker (ML-based) was removed from the production path pending the
        # RTX engine eval (scripts/eval_marker_tables.py). Surface a clear
        # error instead of silently using a different extractor.
        if pdf_extractor != "pymupdf":
            raise ValueError(
                f"Unsupported PDF extractor '{pdf_extractor}'. Only 'pymupdf' is "
                "available; Marker support was removed (see scripts/eval_marker_tables.py)."
            )
        md = extract_pdf_pymupdf(path, keep_spaced, spacing, record)
    else:
        extractor = _EXTRACTORS.get(suffix)
        if extractor is None:
            raise ValueError(f"Unsupported format: {suffix}")
        md = extractor(path)

    return strip_image_placeholders(md)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


class IngestRefusedError(ValueError):
    """The library will not open this file: too large, or an archive shaped like a zip bomb.

    The message is a sentence for the person who added the file (docs/security.md S4, step S-1).
    """


# The supported formats that are zip archives underneath; every other format is opened as itself.
_ARCHIVE_FORMATS: dict[str, str] = {".epub": "EPUB", ".docx": "DOCX", ".odt": "ODT"}


def _size_label(n: float) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{int(n)} bytes"


def ingest_refusal(
    path: Path,
    *,
    max_bytes: int | None = None,
    max_expanded: int | None = None,
    max_ratio: float | None = None,
) -> str | None:
    """Why the library will not open ``path``, as a sentence — or ``None`` when it may.

    Reads the file's size and, for EPUB/DOCX/ODT, the archive's central directory; decompresses
    nothing. The declared sizes can be trusted as a bound: CPython's ``zipfile`` truncates every
    entry at its declared size, and all three libraries read through it (pinned by
    ``tests/unit/test_ingest_size_caps.py``). Caps default to ``config``.

    **Deliberately unreachable from ``extract_to_markdown``.** The extraction fingerprint hashes
    that call graph, so calling this from inside it would mark every cached document stale and
    re-extract the whole corpus (KI-48). The callers are ``get_format_status`` (the add review
    sheet) and ``ingest.cache.load_or_extract`` (just before extraction).
    """
    from doc_assistant import config

    max_bytes = config.MAX_INGEST_FILE_BYTES if max_bytes is None else max_bytes
    max_expanded = config.MAX_ARCHIVE_EXPANDED_BYTES if max_expanded is None else max_expanded
    max_ratio = config.MAX_ARCHIVE_ENTRY_RATIO if max_ratio is None else max_ratio
    try:
        size = path.stat().st_size
    except OSError:
        return None  # unreadable is reported by whoever opens the file, in its own terms
    if size > max_bytes:
        return (
            f"This file is {_size_label(size)}, over the {_size_label(max_bytes)} limit for one "
            "document, so it was not opened."
        )
    kind = _ARCHIVE_FORMATS.get(path.suffix.lower())
    if kind is None:
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, OSError, ValueError):
        return (
            f"This {kind} file is damaged or is not a real {kind} file — its archive could not be "
            "read, so it was not opened."
        )
    expanded = sum(e.file_size for e in entries)
    if expanded > max_expanded:
        return (
            f"This {kind} would expand to {_size_label(expanded)} when opened, over the "
            f"{_size_label(max_expanded)} limit, so it was not opened."
        )
    for entry in entries:
        if entry.file_size == 0:
            continue
        if entry.compress_size == 0 or entry.file_size / entry.compress_size >= max_ratio:
            factor = (
                "without limit"
                if entry.compress_size == 0
                else f"{entry.file_size / entry.compress_size:,.0f} times over"
            )
            return (
                f"This {kind} holds a part that expands {factor} — real documents do not "
                "compress like that, zip bombs do — so it was not opened."
            )
    return None


def get_format_status(path: Path) -> tuple[bool, str | None]:
    """Returns (supported, advisory_message).

    A supported format can still be refused — too large, or an archive shaped like a zip bomb
    (``ingest_refusal``). The add review sheet shows the sentence verbatim.
    """
    ext = path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        refusal = ingest_refusal(path)
        return refusal is None, refusal
    advice = {
        ".doc": "DOC format is not supported. Convert to DOCX or PDF first.",
        ".tex": "LaTeX is not supported yet. Compile to PDF first.",
        ".mobi": "Kindle MOBI is not supported. Try EPUB instead.",
    }
    return False, advice.get(ext, f"Format {ext} is not supported.")
