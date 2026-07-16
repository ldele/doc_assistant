"""Fill Document title/authors/year/DOI from cached markdown (deterministic, $0).

Wires the existing ``metadata_extractor`` onto the ``Document`` registry so the library
grid shows real titles instead of filenames. Reads each document's cached markdown, runs
the academic-paper heuristics, and writes only the four metadata columns — never the
chunk store. Idempotent: a column is written only when currently NULL unless ``--force``.

Usage::

    python -m scripts.enrich_metadata                 # dry-run: report what would be filled
    python -m scripts.enrich_metadata --apply         # persist (only-if-NULL columns)
    python -m scripts.enrich_metadata --apply --force  # overwrite existing metadata too
    python -m scripts.enrich_metadata --doc <id|hash>  # one document
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.metadata_enrich import MetadataEnrichmentResult, enrich_metadata

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _resolve_doc_filter(doc_arg: str) -> str | None:
    """Map ``--doc`` (id-prefix or doc_hash-prefix) to a single Document.id."""
    with session_scope() as session:
        by_id = (
            session.execute(select(Document.id).where(Document.id.like(f"{doc_arg}%")))
            .scalars()
            .all()
        )
        if len(by_id) == 1:
            return str(by_id[0])
        by_hash = (
            session.execute(select(Document.id).where(Document.doc_hash.like(f"{doc_arg}%")))
            .scalars()
            .all()
        )
        if len(by_hash) == 1:
            return str(by_hash[0])
        return None


def _fmt(value: object, width: int) -> str:
    """Truncate for the table; em-dash for a missing field."""
    text = "—" if value is None or value == "" else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _format_report(result: MetadataEnrichmentResult, *, apply: bool) -> str:
    n = result.n_documents or 1

    def pct(found: int) -> str:
        return f"{found}/{result.n_documents} ({100 * found / n:.0f}%)"

    out: list[str] = []
    out.append("=" * 78)
    out.append(f"Documents scanned:  {result.n_documents}")
    out.append(f"  title found:      {pct(result.n_title)}")
    out.append(f"  authors found:    {pct(result.n_authors)}")
    out.append(f"  year found:       {pct(result.n_year)}")
    out.append(f"  DOI found:        {pct(result.n_doi)}")
    if apply:
        out.append(f"  columns written:  {result.total_fields_written}")
    out.append("=" * 78)
    out.append("")
    out.append(f"  {'file':32s} {'year':4s}  title")
    out.append("  " + "-" * 74)
    for d in sorted(result.docs, key=lambda x: x.filename):
        m = d.metadata
        mark = "*" if apply and d.written_fields else " "
        out.append(f"{mark} {_fmt(d.filename, 32):32s} {_fmt(m.year, 4):4s}  {_fmt(m.title, 34)}")
        if m.authors:
            out.append(f"  {'':32s} {'':4s}  by: {_fmt(m.authors, 50)}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist metadata to the DB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite columns that already hold a value (default: only fill NULLs)",
    )
    parser.add_argument(
        "--doc",
        type=str,
        help="Enrich only this document (id or doc_hash prefix). Default: whole corpus.",
    )
    args = parser.parse_args()

    filter_doc_id: str | None = None
    if args.doc:
        filter_doc_id = _resolve_doc_filter(args.doc)
        if filter_doc_id is None:
            print(f"--doc '{args.doc}' did not uniquely resolve to one document.")
            return 1

    result = enrich_metadata(apply=args.apply, force=args.force, document_id=filter_doc_id)
    if result.n_documents == 0:
        print("No documents with cached markdown found. Run ingestion first.")
        return 1

    print(_format_report(result, apply=args.apply))
    if not args.apply:
        print(
            "\nDry run. Pass --apply to persist (only-if-NULL columns; add --force to overwrite)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
