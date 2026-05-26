"""Extract citations from already-ingested documents.

Reads each document's cached markdown, runs the tier-1 regex extractor,
attempts to match each parsed citation to an existing library Document
(DOI / author+year / fuzzy title), and persists rows to the `citations`
table.

Idempotent: by default, skips documents that already have citations.
Use `--force` to re-extract.

Usage:
    python -m scripts.extract_citations                  # dry-run all docs
    python -m scripts.extract_citations --apply          # write rows
    python -m scripts.extract_citations --doc <hash>     # one doc only
    python -m scripts.extract_citations --apply --force  # re-extract everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import delete, func, select

from doc_assistant.citations import (
    ExtractionResult,
    extract_from_markdown,
    match_to_library,
)
from doc_assistant.config import CACHE_PATH, DOCS_PATH
from doc_assistant.db.models import Citation, Document
from doc_assistant.db.session import session_scope

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def _find_cached_text(source_cache: str | None, source_original: str) -> str | None:
    """Locate the cached markdown for a document.

    Tries, in order:
      1. The exact `source_cache` path from the DB.
      2. The path derived from `source_original` relative to DOCS_PATH.
      3. Filename-only lookup against the current CACHE_PATH.

    The third fallback handles the case where the DB stores absolute paths
    from a different host (e.g. Windows paths read from a Linux sandbox).
    """
    if source_cache:
        p = Path(source_cache)
        if p.exists():
            return p.read_text(encoding="utf-8")
    original = Path(source_original)
    if original.exists():
        try:
            relative = original.relative_to(DOCS_PATH)
            derived = CACHE_PATH / relative.with_suffix(".md")
            if derived.exists():
                return derived.read_text(encoding="utf-8")
        except ValueError:
            pass

    # Filename-only fallback: handles cross-host path mismatches.
    for candidate_path in (source_cache, source_original):
        if not candidate_path:
            continue
        stem = Path(candidate_path.replace("\\", "/")).stem
        derived = CACHE_PATH / f"{stem}.md"
        if derived.exists():
            return derived.read_text(encoding="utf-8")
    return None


def _persist(doc_id: str, result: ExtractionResult, *, force: bool) -> int:
    """Write Citation rows for one doc. Returns rows inserted. Idempotent."""
    inserted = 0
    with session_scope() as session:
        existing = session.execute(
            select(func.count()).select_from(Citation).where(
                Citation.source_document_id == doc_id
            )
        ).scalar_one()
        if existing and not force:
            return 0
        if existing and force:
            session.execute(
                delete(Citation).where(Citation.source_document_id == doc_id)
            )

        for parsed in result.citations:
            target_id = match_to_library(parsed)
            session.add(
                Citation(
                    source_document_id=doc_id,
                    target_document_id=target_id,
                    raw_citation_text=parsed.raw_text,
                    target_doi=parsed.doi,
                    target_title=parsed.title,
                    target_authors=parsed.authors,
                    target_year=parsed.year,
                    extraction_method=parsed.extraction_method,
                    confidence=parsed.confidence,
                )
            )
            inserted += 1
    return inserted


def _run_one(
    doc_id: str,
    filename: str,
    source_cache: str | None,
    source_original: str,
    *,
    apply: bool,
    force: bool,
) -> dict[str, object]:
    """Process one document. Returns a row of stats."""
    text = _find_cached_text(source_cache, source_original)
    if text is None:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "status": "no-cache",
            "refs_parsed": 0,
            "matches": 0,
            "needs_tier2": False,
            "notes": "cached markdown not found",
        }

    result = extract_from_markdown(doc_id, text)

    matches = 0
    for parsed in result.citations:
        if match_to_library(parsed) is not None:
            matches += 1

    row: dict[str, object] = {
        "doc_id": doc_id,
        "filename": filename,
        "status": "ok" if result.references_section_found else "no-refs",
        "refs_parsed": result.count,
        "matches": matches,
        "needs_tier2": result.needs_tier2,
        "notes": "; ".join(result.notes) if result.notes else "",
    }

    if apply:
        inserted = _persist(doc_id, result, force=force)
        row["inserted"] = inserted

    return row


def _format_report(rows: list[dict[str, object]], *, apply: bool) -> str:
    out: list[str] = []
    total_refs = sum(int(r["refs_parsed"]) for r in rows)
    total_matches = sum(int(r["matches"]) for r in rows)
    needs_tier2 = sum(1 for r in rows if r["needs_tier2"])
    no_section = sum(1 for r in rows if r["status"] == "no-refs")
    no_cache = sum(1 for r in rows if r["status"] == "no-cache")
    inserted_total = sum(int(r.get("inserted", 0)) for r in rows) if apply else 0

    out.append("=" * 76)
    out.append(f"Documents processed:       {len(rows)}")
    out.append(f"  References section found: {len(rows) - no_section - no_cache}")
    out.append(f"  No references heading:    {no_section}")
    out.append(f"  No cached markdown:       {no_cache}")
    out.append(f"  Tier-2 candidates (<5 refs): {needs_tier2}")
    out.append(f"Total citations parsed:    {total_refs}")
    out.append(f"Internal library matches:  {total_matches}")
    if apply:
        out.append(f"Citation rows inserted:    {inserted_total}")
    out.append("=" * 76)
    out.append("")
    out.append(
        f"{'filename':<55} {'status':<10} {'refs':>5} {'match':>6} {'note'}"
    )
    out.append("-" * 76)
    for r in sorted(rows, key=lambda x: (-int(x["refs_parsed"]), str(x["filename"]))):
        out.append(
            f"{str(r['filename'])[:54]:<55} "
            f"{r['status']!s:<10} "
            f"{int(r['refs_parsed']):>5} "
            f"{int(r['matches']):>6} "
            f"{str(r['notes'])[:25]}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist rows to DB")
    parser.add_argument(
        "--force", action="store_true", help="Re-extract docs that already have citations"
    )
    parser.add_argument("--doc", type=str, help="Limit to one doc_hash or id prefix")
    args = parser.parse_args()

    with session_scope() as session:
        stmt = select(
            Document.id, Document.filename, Document.source_cache, Document.source_original
        ).where(Document.is_archived.is_(False))
        if args.doc:
            stmt = stmt.where(Document.doc_hash.startswith(args.doc))
        docs = [tuple(r) for r in session.execute(stmt).all()]

    if not docs:
        print("No documents matched.")
        return 1

    print(f"Processing {len(docs)} document(s)... (apply={args.apply}, force={args.force})")
    rows = [
        _run_one(doc_id, fn, src_cache, src_orig, apply=args.apply, force=args.force)
        for doc_id, fn, src_cache, src_orig in docs
    ]
    print(_format_report(rows, apply=args.apply))
    if not args.apply:
        print("\nDry run. Pass --apply to write rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
