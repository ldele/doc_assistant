"""Format-specific extractors. Each returns markdown text."""
from pathlib import Path
from bs4 import BeautifulSoup


def extract_pdf_marker(pdf_path: Path) -> str:
    """High-quality PDF extraction with Marker (slower, needs models)."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return text


def extract_pdf_pymupdf(pdf_path: Path) -> str:
    """PDF extraction with page markers preserved as HTML comments."""
    import pymupdf4llm
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    parts = []
    for page_num in range(len(doc)):
        # Mark the start of each page so chunks can be tagged
        parts.append(f"\n<!-- page:{page_num + 1} -->\n")
        page_md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
        parts.append(page_md)
    doc.close()
    return "\n".join(parts)


def extract_epub(epub_path: Path) -> str:
    """Extract EPUB to markdown by parsing inner HTML."""
    from ebooklib import epub, ITEM_DOCUMENT

    book = epub.read_epub(str(epub_path))
    parts = []

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
    parts = []
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


def extract_text(path: Path) -> str:
    """Plain text or markdown — read as-is."""
    return path.read_text(encoding="utf-8", errors="ignore")


# Dispatch table
EXTRACTORS = {
    ".pdf": "pdf",  # special-cased to choose marker vs pymupdf
    ".epub": extract_epub,
    ".html": extract_html,
    ".htm": extract_html,
    ".docx": extract_docx,
    ".md": extract_text,
    ".txt": extract_text,
}


def extract_to_markdown(path: Path, pdf_extractor: str = "pymupdf") -> str:
    """Main entry point: extract any supported file to markdown."""
    suffix = path.suffix.lower()
    if suffix not in EXTRACTORS:
        raise ValueError(f"Unsupported format: {suffix}")

    if suffix == ".pdf":
        return extract_pdf_marker(path) if pdf_extractor == "marker" else extract_pdf_pymupdf(path)

    return EXTRACTORS[suffix](path)


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in EXTRACTORS