"""Compare table engines on the caption-gated candidate pages (RTX-machine eval).

Marker is ML-based and CPU-slow / GPU-fast — run this on the RTX box. It
emits three artifacts per document so you can judge which engine to trust:

  1. The caption-gated candidate pages (cheap; works anywhere).
  2. pdfplumber's extracted tables, rendered as markdown.
  3. Marker's full-document markdown (if `marker-pdf` is installed),
     saved to disk so you can read off its rendering of the candidate pages.

Marker is NOT a project dependency by default (multi-GB models). Install
it only where you intend to run it:

    uv add marker-pdf      # heavy; pulls torch + surya models

Usage:
    python -m scripts.eval_marker_tables --doc <hash>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

from doc_assistant.config import DATA_PATH
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.regions import analyze_pages
from doc_assistant.tables import extract_tables_from_pages, render_table_markdown
from scripts.extract_tables import _resolve_pdf_path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = DATA_PATH / "tables_debug"


def _marker_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("marker") is not None


def _marker_to_markdown(pdf_path: Path) -> str:
    """Run Marker on a PDF → markdown. Self-contained (Marker is not a project dep).

    Lives here, not in ``extractors.py``: Marker was removed from the
    production extraction path; this eval tool is the only thing that uses
    it, and only when installed on the RTX/GPU machine.
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    return str(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=str, required=True, help="doc_hash or id prefix")
    parser.add_argument(
        "--skip-marker", action="store_true", help="Only emit candidate pages + pdfplumber tables"
    )
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

    out_dir = OUT_DIR / Path(filename).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Candidate pages (cheap) — via the page classifier.
    classifications = analyze_pages(str(pdf_path))
    pages = [c.page for c in classifications if c.is_table_candidate]
    print(f"{filename}: table-candidate pages = {pages}")
    for c in classifications:
        if c.is_table_candidate or c.kind in ("chart", "photo", "figure"):
            print(f"  p{c.page}: {c.kind} ({c.reason})")

    # 2. pdfplumber tables on those pages.
    pp_tables = extract_tables_from_pages(str(pdf_path), pages)
    pp_md = "\n\n".join(render_table_markdown(t) for t in pp_tables) or "(none)"
    pp_path = out_dir / "pdfplumber_tables.md"
    pp_path.write_text(pp_md, encoding="utf-8")
    print(f"\npdfplumber extracted {len(pp_tables)} table(s) -> {pp_path}")

    # 3. Marker's markdown (slow; GPU-friendly).
    if args.skip_marker:
        print("\n--skip-marker set; not running Marker.")
        return 0
    if not _marker_available():
        print(
            "\nMarker not installed — skipping engine comparison.\n"
            "  Install where you want to run it (ideally the RTX/GPU machine):\n"
            "    uv add marker-pdf\n"
            "  then re-run this script. Compare Marker's rendering of the\n"
            f"  candidate pages {pages} against {pp_path.name}."
        )
        return 0

    print("\nRunning Marker (slow on CPU; uses GPU if available)...")
    marker_md = _marker_to_markdown(pdf_path)
    marker_path = out_dir / "marker_full.md"
    marker_path.write_text(marker_md, encoding="utf-8")
    print(f"Marker markdown -> {marker_path}")
    print(
        f"\nCompare by hand: open {marker_path.name} and read its rendering of "
        f"pages {pages} against {pp_path.name}. Which engine reproduces the "
        f"table structure (rows/columns/headers) faithfully?"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
