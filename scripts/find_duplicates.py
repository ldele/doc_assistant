"""Find duplicate document files under ``DOCS_PATH``.

Walks the sources tree, hashes each supported file by raw bytes
(SHA-256, truncated to 16 hex chars to mirror ``doc_hash``), groups
files by hash, reports any group with two or more members. When a
cached extraction exists, also hashes the cached markdown so that two
files producing the same extracted content (different scans / OCR
artifacts of the same paper) surface as a second class of duplicate.

Pure read-only. Never deletes anything. Suggested deletes are
printed; the user decides.

Usage::

    python -m scripts.find_duplicates          # report all duplicates
    python -m scripts.find_duplicates --json   # machine-readable output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select

from doc_assistant.config import DOCS_PATH
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.extractors import is_supported
from doc_assistant.ingest import get_cache_path, is_cache_fresh

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming read


def _hash_file_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()[:16]


def _hash_cached_markdown(path: Path) -> str | None:
    """Return the cached markdown hash if a fresh cache exists, else None."""
    cached = get_cache_path(path) if path.is_relative_to(DOCS_PATH) else None
    if cached is None or not is_cache_fresh(path, cached):
        return None
    h = hashlib.sha256()
    h.update(cached.read_text(encoding="utf-8").encode("utf-8"))
    return h.hexdigest()[:16]


def _path_to_document() -> dict[str, dict[str, str]]:
    """Map source_original -> {id, filename, doc_hash} for all non-archived docs."""
    out: dict[str, dict[str, str]] = {}
    with session_scope() as session:
        rows = session.execute(
            select(
                Document.id,
                Document.filename,
                Document.source_original,
                Document.doc_hash,
            ).where(Document.is_archived.is_(False))
        ).all()
        for doc_id, filename, source_original, doc_hash in rows:
            out[str(source_original)] = {
                "id": str(doc_id),
                "filename": str(filename),
                "doc_hash": str(doc_hash),
            }
    return out


def _format_group(
    label: str,
    groups: dict[str, list[Path]],
    path_to_doc: dict[str, dict[str, str]],
) -> str:
    if not groups:
        return f"  {label}: none.\n"
    out: list[str] = [f"  {label}: {len(groups)} group(s)\n"]
    for digest, paths in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out.append(f"    [{digest}] {len(paths)} files")
        canonical: Path | None = None
        for p in paths:
            doc = path_to_doc.get(str(p))
            mark = ""
            if doc is not None:
                mark = f"  <- DB id {doc['id'][:8]}"
                if canonical is None:
                    canonical = p
            out.append(f"      {p}{mark}")
        if canonical is not None and len(paths) > 1:
            others = [p for p in paths if p != canonical]
            out.append(
                f"      -> keep `{canonical.name}`, "
                f"consider deleting: {', '.join(p.name for p in others)}"
            )
        out.append("")
    return "\n".join(out)


def find_duplicates() -> dict[str, Any]:
    """Scan ``DOCS_PATH`` for duplicate files. Returns a structured report."""
    if not DOCS_PATH.exists():
        return {"error": f"DOCS_PATH does not exist: {DOCS_PATH}"}

    files = [p for p in DOCS_PATH.rglob("*") if p.is_file() and is_supported(p)]
    by_bytes: dict[str, list[Path]] = defaultdict(list)
    by_markdown: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_bytes[_hash_file_bytes(path)].append(path)
        md_hash = _hash_cached_markdown(path)
        if md_hash is not None:
            by_markdown[md_hash].append(path)

    byte_dupes = {h: ps for h, ps in by_bytes.items() if len(ps) > 1}
    md_dupes_only = {
        h: ps
        for h, ps in by_markdown.items()
        if len(ps) > 1
        and not any(set(ps) == set(by_bytes[bh]) for bh in by_bytes if len(by_bytes[bh]) > 1)
    }
    return {
        "scanned": len(files),
        "byte_duplicates": {h: [str(p) for p in ps] for h, ps in byte_dupes.items()},
        "markdown_duplicates": {h: [str(p) for p in ps] for h, ps in md_dupes_only.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of the human-readable report"
    )
    args = parser.parse_args()

    report = find_duplicates()
    if "error" in report:
        print(report["error"])
        return 1

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    path_to_doc = _path_to_document()
    byte_dupes = {h: [Path(p) for p in ps] for h, ps in report["byte_duplicates"].items()}
    md_dupes = {h: [Path(p) for p in ps] for h, ps in report["markdown_duplicates"].items()}

    print("=" * 76)
    print(f"Scanned {report['scanned']} supported files under {DOCS_PATH}")
    print(f"Byte-identical duplicate groups:     {len(byte_dupes)}")
    print(f"Content-identical (markdown) extras: {len(md_dupes)}")
    print("=" * 76)
    print()
    print(_format_group("Byte-identical (safe to delete)", byte_dupes, path_to_doc))
    print(
        _format_group(
            "Content-identical via extracted markdown (different file, same content)",
            md_dupes,
            path_to_doc,
        )
    )
    print("Nothing was deleted. Review the suggestions and act manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
