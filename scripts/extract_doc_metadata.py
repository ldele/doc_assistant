"""Backfill document-level metadata (title/authors/year/DOI) for the library.

For each Document in SQLite, reads the cached markdown and runs the regex
metadata extractor. Updates only NULL fields (won't overwrite manual edits)
unless --force is given.

Usage:
    python -m scripts.extract_doc_metadata                  # dry-run all
    python -m scripts.extract_doc_metadata --apply          # write
    python -m scripts.extract_doc_metadata --doc <hash>     # one doc
    python -m scripts.extract_doc_metadata --apply --force  # overwrite non-null
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select, update

from doc_assistant.config import CACHE_PATH, DOCS_PATH
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.metadata_extractor import DocMetadata, extract_metadata

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def _find_cached_text(source_cache: str | None, source_original: str) -> str | None:
    """Locate the cached markdown for a document (mirrors extract_citations.py)."""
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

    for candidate_path in (source_cache, source_original):
        if not candidate_path:
            continue
        stem = Path(candidate_path.replace("\\", "/")).stem
        derived = CACHE_PATH / f"{stem}.md"
        if derived.exists():
            return derived.read_text(encoding="utf-8")
    return None


def _persist(
    doc_id: str,
    metadata: DocMetadata,
    *,
    current: dict[str, object],
    force: bool,
) -> dict[str, object]:
    """Update Document fields. Returns dict of fields that changed."""
    changes: dict[str, object] = {}
    if metadata.title and (force or not current.get("title")):
        changes["title"] = metadata.title
    if metadata.authors and (force or not current.get("authors")):
        changes["authors"] = metadata.authors
    if metadata.year is not None and (force or current.get("year") is None):
        changes["year"] = metadata.year
    if metadata.doi and (force or not current.get("doi")):
        changes["doi"] = metadata.doi

    if not changes:
        return {}

    with session_scope() as session:
        session.execute(update(Document).where(Document.id == doc_id).values(**changes))
    return changes


def _run_one(
    doc_id: str,
    filename: str,
    source_cache: str | None,
    source_original: str,
    current: dict[str, object],
    *,
    apply: bool,
    force: bool,
) -> dict[str, object]:
    text = _find_cached_text(source_cache, source_original)
    if text is None:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "status": "no-cache",
            "confidence": 0.0,
            "fields_filled": 0,
            "changes": {},
        }

    metadata = extract_metadata(text, filename=filename)
    fields_filled = sum(
        1 for v in (metadata.title, metadata.authors, metadata.year, metadata.doi) if v
    )

    changes: dict[str, object] = {}
    if apply:
        changes = _persist(doc_id, metadata, current=current, force=force)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "status": "ok",
        "confidence": metadata.confidence,
        "fields_filled": fields_filled,
        "metadata": metadata,
        "changes": changes,
    }


def _format_report(rows: list[dict[str, object]], *, apply: bool) -> str:
    out: list[str] = []
    total = len(rows)
    no_cache = sum(1 for r in rows if r["status"] == "no-cache")
    total_changes = sum(len(r["changes"]) for r in rows) if apply else 0
    avg_conf = sum(float(r.get("confidence", 0)) for r in rows) / max(1, total)

    out.append("=" * 78)
    out.append(f"Documents processed: {total}")
    out.append(f"  No cached markdown: {no_cache}")
    out.append(f"Average confidence:  {avg_conf:.2f}")
    if apply:
        out.append(f"Total field updates: {total_changes}")
    out.append("=" * 78)
    out.append("")
    out.append(f"{'filename':<44} {'conf':>5} {'fill':>5} {'changes'}")
    out.append("-" * 78)
    for r in sorted(rows, key=lambda x: (-float(x.get("confidence", 0)), str(x["filename"]))):
        ch = r.get("changes", {})
        ch_str = ",".join(sorted(ch.keys())) if ch else ""
        out.append(
            f"{str(r['filename'])[:43]:<44} "
            f"{float(r['confidence']):>5.2f} "
            f"{int(r['fields_filled']):>5} "
            f"{ch_str}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes to DB")
    parser.add_argument("--force", action="store_true", help="Overwrite non-null fields")
    parser.add_argument("--doc", type=str, help="Limit to one doc_hash or id prefix")
    args = parser.parse_args()

    with session_scope() as session:
        stmt = select(
            Document.id,
            Document.filename,
            Document.source_cache,
            Document.source_original,
            Document.title,
            Document.authors,
            Document.year,
            Document.doi,
        ).where(Document.is_archived.is_(False))
        if args.doc:
            stmt = stmt.where(Document.doc_hash.startswith(args.doc))
        docs = [tuple(r) for r in session.execute(stmt).all()]

    if not docs:
        print("No documents matched.")
        return 1

    print(f"Processing {len(docs)} document(s)... (apply={args.apply}, force={args.force})")
    rows: list[dict[str, object]] = []
    for doc_id, fn, src_cache, src_orig, t, a, y, d in docs:
        current = {"title": t, "authors": a, "year": y, "doi": d}
        rows.append(
            _run_one(
                doc_id,
                fn,
                src_cache,
                src_orig,
                current,
                apply=args.apply,
                force=args.force,
            )
        )

    print(_format_report(rows, apply=args.apply))
    if not args.apply:
        print("\nDry run. Pass --apply to write fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
