"""Format-specific extractors. Each returns markdown text."""

import re
from collections.abc import Callable
from pathlib import Path

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


def extract_pdf_pymupdf(pdf_path: Path) -> str:
    """PDF extraction with page markers preserved as HTML comments."""
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(str(pdf_path))  # type: ignore[no-untyped-call]
    parts: list[str] = []
    for page_num in range(len(doc)):
        # Mark the start of each page so chunks can be tagged
        parts.append(f"\n<!-- page:{page_num + 1} -->\n")
        page_md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
        parts.append(page_md)
    doc.close()  # type: ignore[no-untyped-call]
    return "\n".join(parts)


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
        soup = BeautifulSoup(item.get_content(), "lxml")
        # Convert headings to markdown
        for level in range(1, 7):
            for tag in soup.find_all(f"h{level}"):
                tag.replace_with(f"\n{'#' * level} {tag.get_text()}\n")
        # Get clean text from remaining content
        text = soup.get_text(separator="\n").strip()
        if text:
            parts.append(text)

    return "\n\n".join(parts)


def extract_html(html_path: Path) -> str:
    """Extract HTML to markdown-ish text."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    # Remove scripts and styles
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    # Convert headings
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            tag.replace_with(f"\n{'#' * level} {tag.get_text()}\n")
    return soup.get_text(separator="\n").strip()


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
