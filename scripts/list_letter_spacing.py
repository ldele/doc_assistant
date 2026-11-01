"""List every letter-spacing join extraction makes in the library, for review (ADR-056).

Read-only: it opens each library PDF and reads its glyphs, and writes nothing to the library, the
cache or the chunk store. The output is the review list ADR-056's gate 5 asks for: per document,
its sha256 and size (the ignore list's key) and every join — page, the run as PyMuPDF reads it,
the run as the page shows it, its kind, and whether the ignore list already keeps it spaced. A
join the user rejects becomes an entry in ``config.REJOIN_IGNORE_PATH``.

Usage:
    python -m scripts.list_letter_spacing                 # summary per document
    python -m scripts.list_letter_spacing --out joins.json  # and the full list as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from doc_assistant.extractors import KEEP_ALL_SPACED, letter_spacing_page_joins
from doc_assistant.ingest.cache import rejoin_ignore_entries
from doc_assistant.sources_manifest import sha256_file

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class Join:
    page: int
    spaced: str
    joined: str
    kind: str
    kept: bool


@dataclass
class DocumentJoins:
    file: str
    sha256: str
    size: int
    joins: list[Join] = field(default_factory=list)


def joins_for_pdf(path: Path) -> DocumentJoins:
    """Every join the extractor would make in ``path``, marked kept where the ignore list says."""
    import pymupdf

    keep = rejoin_ignore_entries(path)
    found = DocumentJoins(file=path.name, sha256=sha256_file(path), size=path.stat().st_size)
    with pymupdf.open(path) as doc:
        for page in doc:
            for spaced, joined, kind in letter_spacing_page_joins(page):
                kept = KEEP_ALL_SPACED in keep or spaced in keep
                found.joins.append(Join(page.number + 1, spaced, joined, kind, kept))
    return found


def library_pdfs() -> list[Path]:
    """The library's non-archived PDFs whose source file is on disk."""
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library.documents import resolve_source_path

    with session_scope() as session:
        rows = session.execute(
            select(Document.source_original, Document.filename).where(
                Document.is_archived.is_(False)
            )
        ).all()
    paths = [resolve_source_path(source, name) for source, name in rows if source]
    return sorted(p for p in paths if p is not None and p.suffix.lower() == ".pdf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List every letter-spacing join in the library, for review (ADR-056)."
    )
    parser.add_argument("--out", type=Path, help="write the full list as JSON to this file")
    args = parser.parse_args(argv)

    documents = [d for d in (joins_for_pdf(p) for p in library_pdfs()) if d.joins]
    total = sum(len(d.joins) for d in documents)
    kept = sum(j.kept for d in documents for j in d.joins)
    print(f"{total} joins in {len(documents)} documents ({kept} kept spaced by the ignore list)")
    for d in documents:
        kinds = {k: sum(j.kind == k for j in d.joins) for k in ("tracked", "invisible", "url")}
        maths = sum(j.kind == "maths" for j in d.joins)  # left as extracted, listed as not text
        print(
            f"  {d.file}: {len(d.joins)} ({kinds['tracked']} tracked, "
            f"{kinds['invisible']} invisible, {kinds['url']} url, {maths} maths left alone)"
        )
    if args.out:
        payload = {"version": 1, "documents": [asdict(d) for d in documents]}
        args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"full list: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
