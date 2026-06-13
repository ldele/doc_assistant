"""Extract high-fidelity tables with **Marker** and splice them into the markdown cache.

Feature 4a primary path (Marker is the chosen table engine — see
``docs/specs/feature-4a-marker-table-ingest.md``). For each PDF: gate on the
caption-detected table-candidate pages (``regions.table_candidate_pages``), run
isolated Marker on just those pages (out-of-process, ``uvx --from marker-pdf
marker_single``), parse its paginated markdown, and **splice inline** at each
table's page region — de-duping pymupdf4llm's lossy inline table and superseding
any pdfplumber block.

Enrichment-Layer Pattern: PDF-only, idempotent, writes the cache only — never the
chunk store. Tables enter retrieval on the next ``ingest`` — incremental is fine: the
orphan sweep re-hashes each source and drops the pre-splice old-hash chunks
(``ingest._find_orphan_hashes``). A content change drops that doc's sidecar enrichment,
so re-run the citation + doc-vector enrichment afterwards. Marker is slow and loads
multi-GB models, so docs run in a **bounded pool** (``MARKER_MAX_WORKERS``); failures
are isolated per document.

Pinned: confirm ``marker-pdf`` version + the ``--paginate_output`` delimiter on the
machine you run this on (the RTX/GPU box). pdfplumber stays as a no-dep fallback
(``scripts/extract_tables.py``); this Marker pass supersedes it.

Usage:
    python -m scripts.extract_tables_marker                  # dry-run all PDFs
    python -m scripts.extract_tables_marker --apply          # write spliced .md
    python -m scripts.extract_tables_marker --doc <hash>     # one doc only
    python -m scripts.extract_tables_marker --apply --force  # re-splice everything
    python -m scripts.extract_tables_marker --apply --workers 3
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sqlalchemy import select

from doc_assistant.config import DATA_PATH, MARKER_MAX_WORKERS
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.regions import table_candidate_pages
from doc_assistant.tables_marker import (
    has_marker_tables,
    parse_marker_tables,
    splice_tables_inline,
    strip_pdfplumber_block,
)
from scripts.eval_marker_tables import _marker_command, _marker_to_markdown
from scripts.extract_tables import _resolve_cache_path, _resolve_pdf_path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_OUT_ROOT = DATA_PATH / "tables_debug"


def _run_one(
    filename: str,
    source_cache: str | None,
    source_original: str,
    *,
    apply: bool,
    force: bool,
) -> dict[str, object]:
    """Process one PDF: gate → Marker → parse → splice. Returns a stats row."""
    row: dict[str, object] = {"filename": filename, "status": "ok", "tables": 0, "note": ""}
    cache_path = _resolve_cache_path(source_cache, source_original)
    pdf_path = _resolve_pdf_path(source_original, filename)
    if cache_path is None:
        return {**row, "status": "no-cache", "note": "cached markdown not found"}
    if pdf_path is None:
        return {**row, "status": "no-pdf", "note": "source PDF not found"}

    markdown = cache_path.read_text(encoding="utf-8")
    if has_marker_tables(markdown) and not force:
        return {**row, "status": "skipped", "note": "already spliced (use --force)"}

    pages = table_candidate_pages(str(pdf_path))
    if not pages:
        return {**row, "status": "no-pages", "note": "no table-candidate pages"}

    out_dir = _OUT_ROOT / Path(filename).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        marker_md = _marker_to_markdown(pdf_path, pages, out_dir, paginate=True)
    except Exception as e:
        return {**row, "status": "error", "note": f"{type(e).__name__}: {e}"}

    tables = parse_marker_tables(marker_md, pages)
    row["tables"] = len(tables)
    if not tables:
        row["status"] = "no-tables"

    if apply:
        new_markdown = splice_tables_inline(strip_pdfplumber_block(markdown), tables)
        if new_markdown != markdown:
            cache_path.write_text(new_markdown, encoding="utf-8")
            row["note"] = "spliced" if tables else "superseded pdfplumber"
    return row


def _format_report(rows: list[dict[str, object]]) -> str:
    total = sum(int(r["tables"]) for r in rows)
    with_tables = sum(1 for r in rows if int(r["tables"]) > 0)
    out = [
        "=" * 76,
        f"PDF documents processed:   {len(rows)}",
        f"  With tables:              {with_tables}",
        f"  Already spliced (skipped):{sum(1 for r in rows if r['status'] == 'skipped')}",
        f"  No candidate pages:       {sum(1 for r in rows if r['status'] == 'no-pages')}",
        f"  Errors:                   {sum(1 for r in rows if r['status'] == 'error')}",
        f"Total Marker tables:       {total}",
        "=" * 76,
        f"{'filename':<55} {'status':<11} {'tables':>6} note",
        "-" * 76,
    ]
    for r in sorted(rows, key=lambda x: (-int(x["tables"]), str(x["filename"]))):
        out.append(
            f"{str(r['filename'])[:54]:<55} {r['status']!s:<11} "
            f"{int(r['tables']):>6} {str(r['note'])[:30]}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write spliced markdown to the cache")
    parser.add_argument("--force", action="store_true", help="Re-splice docs already spliced")
    parser.add_argument("--doc", type=str, help="Limit to one doc_hash or id prefix")
    parser.add_argument(
        "--workers",
        type=int,
        default=MARKER_MAX_WORKERS,
        help=f"Concurrent Marker subprocesses (default {MARKER_MAX_WORKERS}; drop to 1 on OOM)",
    )
    args = parser.parse_args()

    if _marker_command() is None:
        print(
            "Marker unavailable — need 'marker_single' on PATH or 'uvx' (ships with uv) to "
            "fetch it.\n  Run this on the GPU/RTX box. pdfplumber fallback: "
            "`python -m scripts.extract_tables`."
        )
        return 1

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

    print(
        f"Processing {len(docs)} PDF document(s) with Marker "
        f"(apply={args.apply}, force={args.force}, workers={args.workers})..."
    )
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(_run_one, fn, sc, so, apply=args.apply, force=args.force)
            for fn, sc, so in docs
        ]
        for fut in as_completed(futures):
            rows.append(fut.result())

    print(_format_report(rows))
    print(
        "\nDry run. Pass --apply to write spliced markdown."
        if not args.apply
        else (
            "\nNote: re-run `ingest` to pull the spliced tables into retrieval. "
            "Incremental ingest now self-cleans the pre-splice (old-hash) chunks. "
            "A content change drops the doc's citations/doc_similarities — re-run those enrichments after."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
