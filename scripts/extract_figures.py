"""Detect figure regions in ingested PDFs and persist them as a sidecar.

Reads each PDF document's *source PDF* with PyMuPDF, finds the figure
region(s) on every figure page (gated by ``regions.py``), pairs each with its
caption, renders a cropped PNG under ``data/figures/{doc_hash}/``, and writes
one ``Figure`` row per region.

Enrichment-Layer Pattern: PDF-only, idempotent, writes a *sidecar* (rows +
PNGs) — never the chunk store, the markdown cache, or any ``Document`` column.
Unlike tables, figures are NOT spliced and NO re-ingest is needed for the
records themselves; re-ingest only matters once Feature 4c turns figures into
retrievable chunks.

Idempotent: by default, skips documents that already have ``Figure`` rows. Use
``--force`` to delete a document's rows + PNGs and re-extract.

Usage:
    python -m scripts.extract_figures                  # dry-run all PDFs
    python -m scripts.extract_figures --apply          # write rows + PNGs
    python -m scripts.extract_figures --doc <hash>     # one doc only
    python -m scripts.extract_figures --apply --force  # re-extract everything
    python -m scripts.extract_figures --apply --dpi 200  # higher-res crops
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from sqlalchemy import delete, func, or_, select

from doc_assistant.config import DOCS_PATH, FIGURE_RENDER_DPI
from doc_assistant.db.models import Document, Figure
from doc_assistant.db.session import session_scope
from doc_assistant.figures import (
    FigureRegion,
    detect_figure_regions,
    figure_dir,
    figure_image_path,
    render_region,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _resolve_pdf_path(source_original: str, filename: str) -> Path | None:
    """Locate the source PDF (the DB path, else DOCS_PATH/filename)."""
    p = Path(source_original)
    if p.exists():
        return p
    candidate = DOCS_PATH / filename
    if candidate.exists():
        return candidate
    return None


def _existing_figure_count(document_id: str) -> int:
    with session_scope() as session:
        return int(
            session.execute(
                select(func.count()).select_from(Figure).where(Figure.document_id == document_id)
            ).scalar_one()
        )


def _apply_figures(
    document_id: str,
    doc_hash: str,
    pdf_path: Path,
    regions: list[FigureRegion],
    *,
    dpi: int,
    force: bool,
) -> int:
    """Render PNGs + persist ``Figure`` rows. Returns the number of PNGs written.

    With ``force`` the document's existing PNG directory is cleared first and
    its rows are deleted inside the same transaction as the inserts.
    """
    if force:
        existing_dir = figure_dir(doc_hash)
        if existing_dir.exists():
            shutil.rmtree(existing_dir)

    rendered = 0
    rows: list[Figure] = []
    import pymupdf

    doc = pymupdf.open(str(pdf_path))  # type: ignore[no-untyped-call]
    try:
        page_index: dict[int, int] = {}
        for region in regions:
            idx = page_index.get(region.page, 0)
            page_index[region.page] = idx + 1

            image_path: str | None = None
            if region.bbox is not None:
                out_path = figure_image_path(doc_hash, region.page, idx)
                render_region(doc[region.page - 1], region.bbox, out_path, dpi=dpi)
                image_path = str(out_path)
                rendered += 1

            bbox = region.bbox
            rows.append(
                Figure(
                    document_id=document_id,
                    doc_hash=doc_hash,
                    page=region.page,
                    bbox_x0=bbox[0] if bbox else None,
                    bbox_y0=bbox[1] if bbox else None,
                    bbox_x1=bbox[2] if bbox else None,
                    bbox_y1=bbox[3] if bbox else None,
                    kind=region.kind,
                    caption=region.caption,
                    image_path=image_path,
                    extraction_method=region.extraction_method,
                )
            )
    finally:
        doc.close()  # type: ignore[no-untyped-call]

    with session_scope() as session:
        if force:
            session.execute(delete(Figure).where(Figure.document_id == document_id))
        for row in rows:
            session.add(row)
    return rendered


def _run_one(
    document_id: str,
    doc_hash: str,
    filename: str,
    source_original: str,
    fmt: str,
    *,
    apply: bool,
    force: bool,
    dpi: int,
) -> dict[str, object]:
    """Process one document. Returns a row of stats."""
    row: dict[str, object] = {
        "filename": filename,
        "status": "ok",
        "figures": 0,
        "captioned": 0,
        "rendered": 0,
        "note": "",
    }

    if fmt != "pdf":
        row["status"] = "skip-nonpdf"
        row["note"] = "non-PDF source (v1 is PDF-only)"
        return row

    pdf_path = _resolve_pdf_path(source_original, filename)
    if pdf_path is None:
        row["status"] = "no-pdf"
        row["note"] = "source PDF not found"
        return row

    if not force and _existing_figure_count(document_id):
        row["status"] = "skipped"
        row["note"] = "already extracted (use --force)"
        return row

    # Per-doc isolation (covers detection AND rendering): one bad PDF errors
    # its own row and the batch run continues. The apply transaction rolls back
    # on failure, so a partial render leaves no Figure rows for this doc.
    try:
        regions = detect_figure_regions(str(pdf_path))
        row["figures"] = len(regions)
        row["captioned"] = sum(1 for r in regions if r.caption)
        if not regions:
            row["status"] = "no-figures"
        if apply and regions:
            rendered = _apply_figures(
                document_id, doc_hash, pdf_path, regions, dpi=dpi, force=force
            )
            row["rendered"] = rendered
            row["note"] = "written"
    except Exception as e:
        row["status"] = "error"
        row["note"] = f"{type(e).__name__}: {e}"

    return row


def _format_report(rows: list[dict[str, object]], *, apply: bool) -> str:
    total_figures = sum(int(r["figures"]) for r in rows)
    total_rendered = sum(int(r["rendered"]) for r in rows)
    with_figures = sum(1 for r in rows if int(r["figures"]) > 0)
    skipped = sum(1 for r in rows if r["status"] == "skipped")
    non_pdf = sum(1 for r in rows if r["status"] == "skip-nonpdf")
    no_pdf = sum(1 for r in rows if r["status"] == "no-pdf")
    errors = sum(1 for r in rows if r["status"] == "error")

    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Documents processed:       {len(rows)}")
    out.append(f"  With figures:             {with_figures}")
    out.append(f"  Already extracted (skip): {skipped}")
    out.append(f"  Non-PDF (skipped):        {non_pdf}")
    out.append(f"  Source PDF missing:       {no_pdf}")
    out.append(f"  Detection errors:         {errors}")
    out.append(f"Total figure regions:      {total_figures}")
    if apply:
        out.append(f"PNGs rendered:             {total_rendered}")
    out.append("=" * 76)
    out.append("")
    out.append(f"{'filename':<48} {'status':<12} {'figs':>4} {'capt':>4} {'note'}")
    out.append("-" * 76)
    for r in sorted(rows, key=lambda x: (-int(x["figures"]), str(x["filename"]))):
        out.append(
            f"{str(r['filename'])[:47]:<48} "
            f"{r['status']!s:<12} "
            f"{int(r['figures']):>4} "
            f"{int(r['captioned']):>4} "
            f"{str(r['note'])[:24]}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write Figure rows + render PNGs")
    parser.add_argument(
        "--force", action="store_true", help="Re-extract docs that already have figure rows"
    )
    parser.add_argument("--doc", type=str, help="Limit to one doc_hash or id prefix")
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIGURE_RENDER_DPI,
        help="Crop resolution (default %(default)s)",
    )
    args = parser.parse_args()

    with session_scope() as session:
        stmt = select(
            Document.id,
            Document.doc_hash,
            Document.filename,
            Document.source_original,
            Document.format,
        ).where(Document.is_archived.is_(False))
        if args.doc:
            stmt = stmt.where(
                or_(Document.doc_hash.startswith(args.doc), Document.id.startswith(args.doc))
            )
        docs = [tuple(r) for r in session.execute(stmt).all()]

    if not docs:
        print("No documents matched.")
        return 1

    print(f"Processing {len(docs)} document(s)... (apply={args.apply}, force={args.force})")
    rows = [
        _run_one(
            str(doc_id),
            str(doc_hash),
            str(filename),
            str(source_original),
            str(fmt),
            apply=args.apply,
            force=args.force,
            dpi=args.dpi,
        )
        for doc_id, doc_hash, filename, source_original, fmt in docs
    ]
    print(_format_report(rows, apply=args.apply))
    if not args.apply:
        print("\nDry run. Pass --apply to write figure records + PNGs.")
    else:
        print("\nFigures are a sidecar — no re-ingest needed (Feature 4c turns them into chunks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
