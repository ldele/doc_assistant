"""Format-specific extractors. Each returns markdown text."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from bs4 import BeautifulSoup

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


def extract_pdf_pymupdf(pdf_path: Path) -> str:
    """PDF extraction with page markers preserved as HTML comments.

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

    doc = pymupdf.open(str(pdf_path))  # type: ignore[no-untyped-call]
    parts: list[str] = []
    for page_num in range(len(doc)):
        # Mark the start of each page so chunks can be tagged
        parts.append(f"\n<!-- page:{page_num + 1} -->\n")
        page_md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
        parts.append(_recover_lost_page(doc, page_num, page_md))
    doc.close()  # type: ignore[no-untyped-call]
    return "\n".join(parts)


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


def extract_to_markdown(path: Path, pdf_extractor: str = "pymupdf") -> str:
    """Main entry point: extract any supported file to markdown.

    The extracted markdown is passed through ``strip_image_placeholders`` at the single
    exit so PyMuPDF4LLM image placeholders never reach the cache / chunk store (KI-14);
    the strip is a no-op for formats and documents without placeholders.
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
        md = extract_pdf_pymupdf(path)
    else:
        extractor = _EXTRACTORS.get(suffix)
        if extractor is None:
            raise ValueError(f"Unsupported format: {suffix}")
        md = extractor(path)

    return strip_image_placeholders(md)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def get_format_status(path: Path) -> tuple[bool, str | None]:
    """Returns (supported, advisory_message)."""
    ext = path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return True, None
    advice = {
        ".doc": "DOC format is not supported. Convert to DOCX or PDF first.",
        ".tex": "LaTeX is not supported yet. Compile to PDF first.",
        ".mobi": "Kindle MOBI is not supported. Try EPUB instead.",
    }
    return False, advice.get(ext, f"Format {ext} is not supported.")
