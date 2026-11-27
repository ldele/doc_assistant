"""What the fast track's extraction finds in each document, read-only (0.9.0 plan, session 2).

Runs `doc_assistant.ingest.extraction_report.report_document` over the library and over any
folders given, and writes one JSON line per document plus a summary per set. It writes nothing to
the library, the cache, the chunk store or the figures folder: the library is opened as a
read-only SQLite file, a fresh cache is read, and a stale or missing one is extracted in memory.
Its only output is ``--out``. The output names documents, so for the owner's library it belongs
in a local-only place (``docs/reviews/`` is local-only, ADR-051).

Sets: a library document in the *Demo corpus* folder is ``public``, any other library document
``library``; a file under a ``--files`` folder is ``other`` (or the name given with
``--files NAME=DIR``).

Usage:
    python -m scripts.extraction_report --out report.jsonl                  # the library
    python -m scripts.extraction_report --no-library --files other=C:/samples --out o.jsonl
    python -m scripts.extraction_report --limit 5 --no-tables              # a quick look
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from doc_assistant import config
from doc_assistant.extractors import is_supported
from doc_assistant.ingest.extraction_report import report_document

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PUBLIC_FOLDER = "Demo corpus"


def library_documents(db_path: str) -> list[dict[str, Any]]:
    """Every library document that is not archived, with its stored title/authors and its set."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        public = {
            r[0]
            for r in con.execute(
                "select df.document_id from document_folders df join folders f "
                "on f.id = df.folder_id where f.name = ?",
                (PUBLIC_FOLDER,),
            )
        }
        rows = con.execute(
            "select id, source_original, title, authors from documents "
            "where is_archived = 0 order by filename"
        ).fetchall()
    finally:
        con.close()
    return [
        {
            "set": "public" if doc_id in public else "library",
            "path": Path(src),
            "title": title,
            "authors": authors,
        }
        for doc_id, src, title, authors in rows
        if src
    ]


def folder_documents(spec: str) -> list[dict[str, Any]]:
    name, _, folder = spec.rpartition("=")
    root = Path(folder)
    return [
        {"set": name or "other", "path": p, "title": None, "authors": None}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per set: how many documents show each fault, and the time each pass took in all."""
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for name, docs in sorted(by_set.items()):
        pdfs = [d for d in docs if d["format"] == "pdf"]
        seconds: dict[str, float] = defaultdict(float)
        for d in docs:
            for p, res in d["passes"].items():
                seconds[p] += res["seconds"]
        out[name] = {
            "documents": len(docs),
            "pdfs": len(pdfs),
            "unsupported": sum(1 for d in docs if d.get("unsupported")),
            "pass_errors": sum(1 for d in docs if any(r["error"] for r in d["passes"].values())),
            "no_text": sum(1 for d in docs if not d.get("unsupported") and d["chars"] < 200),
            "empty_pages_docs": sum(1 for d in docs if d["empty_pages"]),
            "no_title": sum(1 for d in docs if not d["title"]),
            "no_authors": sum(1 for d in docs if not d["authors"]),
            "no_year": sum(1 for d in docs if d["year"] is None),
            "title_is_running_line": sum(1 for d in docs if d["title_is_running_line"]),
            "stored_title_is_running_line": sum(
                1 for d in docs if d["stored_title_is_running_line"]
            ),
            "authors_read_as_prose": sum(1 for d in docs if d["authors_read_as_prose"]),
            "stored_authors_read_as_prose": sum(
                1 for d in docs if d["stored_authors_read_as_prose"]
            ),
            "running_lines_docs": sum(1 for d in docs if d["running_lines"]),
            "repeated_pages_docs": sum(1 for d in docs if d["repeated_pages"]),
            "hyphen_split_docs": sum(1 for d in docs if d["hyphen_splits"]),
            "hyphen_splits": sum(d["hyphen_splits"] for d in docs),
            "no_references_heading": sum(1 for d in docs if not d["references_found"]),
            "references_heading_no_entries": sum(
                1 for d in docs if d["references_found"] and d["reference_entries"] == 0
            ),
            "references_under_5": sum(
                1 for d in docs if d["references_found"] and 0 < d["reference_entries"] < 5
            ),
            "figures_docs": sum(1 for d in pdfs if d["figures"]),
            "figures": sum(d["figures"] for d in pdfs),
            "figures_caption_only": sum(d["figures_caption_only"] for d in pdfs),
            "figures_without_caption": sum(d["figures_without_caption"] for d in pdfs),
            "tables_docs": sum(1 for d in pdfs if d["tables"]),
            "tables": sum(d["tables"] for d in pdfs),
            "tables_as_text_docs": sum(1 for d in pdfs if d["tables_as_text_pages"]),
            "multi_column_docs": sum(1 for d in pdfs if d["multi_column_pages"]),
            "seconds_by_pass": {k: round(v, 1) for k, v in sorted(seconds.items())},
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="JSON lines, one per document")
    ap.add_argument("--summary", type=Path, help="the per-set summary as JSON (default: printed)")
    ap.add_argument("--no-library", action="store_true", help="skip the library's documents")
    ap.add_argument("--files", action="append", default=[], help="[NAME=]DIR of other documents")
    ap.add_argument("--limit", type=int, default=0, help="first N documents only")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    args = ap.parse_args(argv)

    docs: list[dict[str, Any]] = [] if args.no_library else library_documents(config.SQLITE_PATH)
    for spec in args.files:
        docs.extend(folder_documents(spec))
    if args.limit:
        docs = docs[: args.limit]

    rows: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.out.open("w", encoding="utf-8") as fh:
        for i, d in enumerate(docs, start=1):
            path: Path = d["path"]
            if not is_supported(path):
                row: dict[str, Any] = {
                    "set": d["set"],
                    "path": str(path),
                    "name": path.name,
                    "format": path.suffix.lstrip(".").lower(),
                    "unsupported": True,
                }
                print(f"[{i}/{len(docs)}] {d['set']:8} {path.name}: not a supported type")
            else:
                rep = report_document(
                    path,
                    stored_title=d["title"],
                    stored_authors=d["authors"],
                    with_figures=not args.no_figures,
                    with_tables=not args.no_tables,
                )
                row = {"set": d["set"], **rep.to_json()}
                rows.append(row)
                took = sum(p["seconds"] for p in row["passes"].values())
                print(f"[{i}/{len(docs)}] {d['set']:8} {path.name}: {took:.1f} s")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarise(rows)
    summary["_run"] = {"documents": len(docs), "seconds": round(time.perf_counter() - started, 1)}
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.summary:
        args.summary.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
