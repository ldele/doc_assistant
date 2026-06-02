"""Visual debug for table detection (Feature 4a inspection aid).

Renders each PDF page that *either* detector flags as containing a table,
with pdfplumber's table-finder overlay drawn on it, and prints a per-page
comparison of pdfplumber vs PyMuPDF ``find_tables()`` counts. Use it to
SEE detection precision/recall per document and to tune the extraction
strategy with evidence instead of guessing.

This is a dev/inspection tool, not part of the ingest path. It writes
PNGs to ``data/tables_debug/{stem}/`` and never touches the cache or DB.

Usage:
    python -m scripts.debug_tables --doc <hash>     # one document
    python -m scripts.debug_tables --doc <hash> --max-pages 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

from doc_assistant.config import DATA_PATH
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from scripts.extract_tables import _resolve_pdf_path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DEBUG_DIR = DATA_PATH / "tables_debug"


def _pymupdf_table_count(pdf_path: str) -> dict[int, int]:
    """Per-page count of tables PyMuPDF's detector finds (1-based pages)."""
    import pymupdf

    counts: dict[int, int] = {}
    doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    try:
        for i in range(len(doc)):
            page = doc[i]
            try:
                tabs = page.find_tables()
                counts[i + 1] = len(tabs.tables)
            except Exception:
                counts[i + 1] = 0
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    return counts


def _render(
    pdf_path: str, out_dir: Path, *, max_pages: int, resolution: int
) -> list[dict[str, object]]:
    """Render overlay PNGs for table-bearing pages. Returns per-page stats."""
    import pdfplumber

    pymupdf_counts = _pymupdf_table_count(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rendered = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            pp = len(page.find_tables())
            mu = pymupdf_counts.get(page_number, 0)
            if pp == 0 and mu == 0:
                continue
            saved = ""
            if rendered < max_pages:
                im = page.to_image(resolution=resolution)
                im.debug_tablefinder()
                path = out_dir / f"page_{page_number:03d}.png"
                im.save(str(path))
                saved = path.name
                rendered += 1
            rows.append({"page": page_number, "pdfplumber": pp, "pymupdf": mu, "image": saved})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=str, required=True, help="doc_hash or id prefix")
    parser.add_argument("--max-pages", type=int, default=8, help="Cap rendered pages")
    parser.add_argument("--resolution", type=int, default=120, help="PNG render DPI")
    args = parser.parse_args()

    with session_scope() as session:
        row = session.execute(
            select(Document.filename, Document.source_original).where(
                Document.doc_hash.startswith(args.doc), Document.format == "pdf"
            )
        ).first()

    if row is None:
        print(f"No PDF document matching '{args.doc}'.")
        return 1

    filename, source_original = row
    pdf_path = _resolve_pdf_path(source_original, filename)
    if pdf_path is None:
        print(f"Source PDF not found for {filename}.")
        return 1

    out_dir = DEBUG_DIR / Path(filename).stem
    print(f"Rendering table-detection debug for {filename} -> {out_dir}")
    rows = _render(str(pdf_path), out_dir, max_pages=args.max_pages, resolution=args.resolution)

    if not rows:
        print("Neither detector found any tables in this document.")
        return 0

    print(f"\n{'page':>5} {'pdfplumber':>11} {'pymupdf':>8}  image")
    print("-" * 50)
    for r in rows:
        print(
            f"{int(r['page']):>5} {int(r['pdfplumber']):>11} {int(r['pymupdf']):>8}  {r['image']}"
        )
    agree = sum(1 for r in rows if r["pdfplumber"] == r["pymupdf"])
    print(f"\nPages where the two detectors agree on count: {agree}/{len(rows)}")
    print(f"PNGs written: {min(args.max_pages, len(rows))} (open them to inspect detection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
