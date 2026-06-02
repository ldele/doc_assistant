"""Extract tables from ingested PDFs and splice them into the markdown cache.

Reads each PDF document's *source PDF* with pdfplumber, renders every
detected table as a GitHub-flavoured markdown table, and splices the
result into the document's cached ``.md`` file (inside a demarcated
``<!-- tables:pdfplumber:begin … :end -->`` block).

Enrichment-Layer Pattern: PDF-only, idempotent, writes the cache only —
never the chunk store. Tables become retrievable on the next ``ingest``
run that re-reads the cache.

Idempotent: by default, skips documents that already have a spliced
table block. Use ``--force`` to re-extract and replace it.

Usage:
    python -m scripts.extract_tables                  # dry-run all PDFs
    python -m scripts.extract_tables --apply          # write spliced .md
    python -m scripts.extract_tables --doc <hash>     # one doc only
    python -m scripts.extract_tables --apply --force  # re-splice everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select

from doc_assistant.config import CACHE_PATH, DOCS_PATH
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.tables import (
    ExtractedTable,
    extract_tables,
    has_spliced_tables,
    splice_tables,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def _resolve_cache_path(source_cache: str | None, source_original: str) -> Path | None:
    """Locate the cached ``.md`` path for a document (cross-host tolerant)."""
    if source_cache:
        p = Path(source_cache)
        if p.exists():
            return p
    original = Path(source_original)
    if original.exists():
        try:
            relative = original.relative_to(DOCS_PATH)
            derived = CACHE_PATH / relative.with_suffix(".md")
            if derived.exists():
                return derived
        except ValueError:
            pass
    for candidate in (source_cache, source_original):
        if not candidate:
            continue
        stem = Path(candidate.replace("\\", "/")).stem
        derived = CACHE_PATH / f"{stem}.md"
        if derived.exists():
            return derived
    return None


def _resolve_pdf_path(source_original: str, filename: str) -> Path | None:
    """Locate the source PDF (the DB path, else DOCS_PATH/filename)."""
    p = Path(source_original)
    if p.exists():
        return p
    candidate = DOCS_PATH / filename
    if candidate.exists():
        return candidate
    return None


def _run_one(
    filename: str,
    source_cache: str | None,
    source_original: str,
    *,
    apply: bool,
    force: bool,
) -> dict[str, object]:
    """Process one PDF document. Returns a row of stats."""
    cache_path = _resolve_cache_path(source_cache, source_original)
    pdf_path = _resolve_pdf_path(source_original, filename)

    row: dict[str, object] = {
        "filename": filename,
        "status": "ok",
        "tables": 0,
        "note": "",
    }

    if cache_path is None:
        row["status"] = "no-cache"
        row["note"] = "cached markdown not found"
        return row
    if pdf_path is None:
        row["status"] = "no-pdf"
        row["note"] = "source PDF not found"
        return row

    markdown = cache_path.read_text(encoding="utf-8")
    if has_spliced_tables(markdown) and not force:
        row["status"] = "skipped"
        row["note"] = "already spliced (use --force)"
        return row

    try:
        tables: list[ExtractedTable] = extract_tables(str(pdf_path))
    except Exception as e:
        row["status"] = "error"
        row["note"] = f"{type(e).__name__}: {e}"
        return row

    row["tables"] = len(tables)
    if not tables:
        row["status"] = "no-tables"

    if apply:
        new_markdown = splice_tables(markdown, tables)
        if new_markdown != markdown:
            cache_path.write_text(new_markdown, encoding="utf-8")
            row["note"] = "spliced" if tables else "cleared block"

    return row


def _format_report(rows: list[dict[str, object]], *, apply: bool) -> str:
    out: list[str] = []
    total_tables = sum(int(r["tables"]) for r in rows)
    with_tables = sum(1 for r in rows if int(r["tables"]) > 0)
    skipped = sum(1 for r in rows if r["status"] == "skipped")
    no_cache = sum(1 for r in rows if r["status"] == "no-cache")
    no_pdf = sum(1 for r in rows if r["status"] == "no-pdf")
    errors = sum(1 for r in rows if r["status"] == "error")

    out.append("=" * 76)
    out.append(f"PDF documents processed:   {len(rows)}")
    out.append(f"  With tables:              {with_tables}")
    out.append(f"  Already spliced (skipped):{skipped}")
    out.append(f"  No cached markdown:       {no_cache}")
    out.append(f"  Source PDF missing:       {no_pdf}")
    out.append(f"  Extraction errors:        {errors}")
    out.append(f"Total tables extracted:    {total_tables}")
    out.append("=" * 76)
    out.append("")
    out.append(f"{'filename':<55} {'status':<11} {'tables':>6} {'note'}")
    out.append("-" * 76)
    for r in sorted(rows, key=lambda x: (-int(x["tables"]), str(x["filename"]))):
        out.append(
            f"{str(r['filename'])[:54]:<55} "
            f"{r['status']!s:<11} "
            f"{int(r['tables']):>6} "
            f"{str(r['note'])[:30]}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write spliced markdown to the cache")
    parser.add_argument(
        "--force", action="store_true", help="Re-splice docs that already have a table block"
    )
    parser.add_argument("--doc", type=str, help="Limit to one doc_hash or id prefix")
    args = parser.parse_args()

    with session_scope() as session:
        stmt = select(Document.filename, Document.source_cache, Document.source_original).where(
            Document.is_archived.is_(False), Document.format == "pdf"
        )
        if args.doc:
            stmt = stmt.where(Document.doc_hash.startswith(args.doc))
        docs = [tuple(r) for r in session.execute(stmt).all()]

    if not docs:
        print("No PDF documents matched.")
        return 1

    print(f"Processing {len(docs)} PDF document(s)... (apply={args.apply}, force={args.force})")
    rows = [
        _run_one(fn, src_cache, src_orig, apply=args.apply, force=args.force)
        for fn, src_cache, src_orig in docs
    ]
    print(_format_report(rows, apply=args.apply))
    if not args.apply:
        print("\nDry run. Pass --apply to write spliced markdown.")
    else:
        print("\nNote: re-run `ingest` to pull the spliced tables into retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
