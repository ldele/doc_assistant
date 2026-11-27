"""The extraction record: what one run found in one document (0.9.0, `ingestion-record.md`).

Indexing used to read text and metadata only, so a document with no reference rows might have none
or might never have been looked at. Indexing now runs the figure and reference passes (the
functions *Re-run…* calls, never a copy) and the table splice (in `cache.load_or_extract`), and
this module writes, for each document, one `ExtractionRecord` row saying what each pass found, with
the parser's own reasons.

**A warning is one predicate over the current record** (:data:`WARNINGS`), defined here once and
read by the document page, the queue row and the Library summary, so their counts cannot disagree.

**Each row is complete.** A run that did not run a pass (a *Re-run… > References* alone) carries
that pass's section and state forward from the newest row that did, marked ``carried_from``, so
the newest row always says everything and the summary counts its columns directly. Old rows are
kept, at most :data:`RECORDS_KEPT` per document, for a fix's before and after.

**It never fails an ingest.** Every pass is caught and recorded as ``error`` with its reason.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from doc_assistant.db.models import Document, DocumentMeta, ExtractionRecord
from doc_assistant.db.session import session_scope

log = structlog.get_logger(__name__)

RECORD_VERSION = 1

#: The newest rows kept per document. Enough for a fix's before and after with room for a few
#: re-runs between; a structural bound so the table grows with the library, not with its history.
RECORDS_KEPT = 5

#: Below this many entries a found reference list is *partial*. Structural, not tuned: a real
#: paper's list is longer, and a parse that stops this early has usually stopped at a layout
#: break (the session-2 review's fault 10: one, two and three entries on three papers).
REFERENCES_FEW = 5

#: A share of the document's own pages, so a 400-page book and a 4-page note are judged alike.
EMPTY_PAGES_SHARE = 0.2

#: ``found`` ran and found some · ``partial`` found some, with a named gap · ``none_found`` ran and
#: found nothing · ``skipped`` not applicable (a reason says why) · ``error`` raised · ``not_run``
#: never run on this document by any recorded run.
STATES: tuple[str, ...] = ("found", "partial", "none_found", "skipped", "error", "not_run")

_PASSES = ("references", "figures", "tables")
_TABLE_MARKER = re.compile(r"<!-- table-extracted-by: pdfplumber page=(\d+) table=\d+ -->")

# --- facts read from the text ----------------------------------------------------------------- #


def text_facts(markdown: str) -> dict[str, Any]:
    """Pages, characters and empty pages of the extracted text (the spliced tables left out)."""
    from doc_assistant.ingest.extraction_report import page_segments
    from doc_assistant.ingest.tables import strip_spliced_tables

    segments = page_segments(strip_spliced_tables(markdown))
    marked = [(p, t) for p, t in segments if p > 0]
    return {
        "pages": len(marked),
        "chars": len(markdown),
        "empty_pages": [p for p, t in marked if not t.strip()],
    }


def table_facts(markdown: str, splice: dict[str, Any] | None, *, is_pdf: bool) -> dict[str, Any]:
    """What the table splice did, and the pages that name a table none was extracted from."""
    from doc_assistant.ingest.extraction_report import page_segments, table_mentions
    from doc_assistant.ingest.tables import strip_spliced_tables

    naming = sorted(table_mentions(page_segments(strip_spliced_tables(markdown))))
    with_table = {int(m) for m in _TABLE_MARKER.findall(markdown)}
    facts: dict[str, Any] = {
        "extracted": len(_TABLE_MARKER.findall(markdown)),
        "pages_naming_a_table": naming,
        "pages_without_one": [p for p in naming if p not in with_table],
        "seconds": None,
        "reason": None,
    }
    if not is_pdf:
        facts["reason"] = "tables are read from PDFs only"
    elif splice is None:
        facts["reason"] = "the text was read before indexing read tables"
    else:
        facts["seconds"] = splice.get("seconds")
        facts["reason"] = splice.get("error")
    return facts


# --- states ----------------------------------------------------------------------------------- #


def references_state(status: str, facts: dict[str, Any]) -> str:
    if status in ("error", "skipped"):
        return status
    entries = int(facts.get("entries") or 0)
    if entries == 0:
        return "none_found"
    return "partial" if entries < REFERENCES_FEW else "found"


def figures_state(status: str, facts: dict[str, Any]) -> str:
    if status in ("error", "skipped"):
        return status
    return "found" if int(facts.get("found") or 0) > 0 else "none_found"


def tables_state(facts: dict[str, Any], *, is_pdf: bool, ran: bool) -> str:
    if not is_pdf:
        return "skipped"
    if not ran:
        return "not_run"
    if facts.get("reason"):
        return "error"
    if not facts["extracted"]:
        return "none_found"
    return "partial" if facts["pages_without_one"] else "found"


# --- warnings --------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecordWarning:
    id: str
    text: str


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _pages(pages: Sequence[int]) -> str:
    shown = ", ".join(str(p) for p in pages[:5])
    more = f" and {len(pages) - 5} more" if len(pages) > 5 else ""
    return f"{'Page' if len(pages) == 1 else 'Pages'} {shown}{more}"


def _w_references_none(r: dict[str, Any]) -> str | None:
    refs = r["references"]
    if r["states"]["references"] != "none_found" or refs.get("heading"):
        return None
    note = (refs.get("notes") or [None])[0]
    return "No reference list found" + (f" ({note})" if note else "")


def _w_references_heading_only(r: dict[str, Any]) -> str | None:
    refs = r["references"]
    if r["states"]["references"] == "none_found" and refs.get("heading"):
        return "A references heading, but no entries"
    return None


def _w_references_few(r: dict[str, Any]) -> str | None:
    if r["states"]["references"] != "partial":
        return None
    return f"Only {_plural(int(r['references']['entries']), 'reference', 'references')} found"


def _w_figures_uncaptioned(r: dict[str, Any]) -> str | None:
    fig = r["figures"]
    if r["states"]["figures"] != "found":
        return None
    missing = int(fig.get("found") or 0) - int(fig.get("with_caption") or 0)
    return f"{_plural(missing, 'figure', 'figures')} without a caption" if missing > 0 else None


def _w_tables_missed(r: dict[str, Any]) -> str | None:
    if r["states"]["tables"] not in ("none_found", "partial"):
        return None
    missed = r["tables"].get("pages_without_one") or []
    if not missed:
        return None
    verb = "names" if len(missed) == 1 else "name"
    return f"{_pages(missed)} {verb} a table that wasn't extracted"


def _w_metadata_missing(r: dict[str, Any]) -> str | None:
    missing = [k for k in ("title", "authors", "year") if not r["metadata"].get(k)]
    return "No " + ", no ".join(missing) if missing else None


def _w_text_empty_pages(r: dict[str, Any]) -> str | None:
    text = r["text"]
    pages, empty = int(text.get("pages") or 0), len(text.get("empty_pages") or [])
    if pages and empty / pages >= EMPTY_PAGES_SHARE:
        return f"{_plural(empty, 'page has', 'pages have')} no text"
    return None


#: Every warning, in the order a reader should meet them. Ids are API values and summary issues.
WARNINGS: tuple[tuple[str, Callable[[dict[str, Any]], str | None]], ...] = (
    ("references_none", _w_references_none),
    ("references_heading_only", _w_references_heading_only),
    ("references_few", _w_references_few),
    ("figures_uncaptioned", _w_figures_uncaptioned),
    ("tables_missed", _w_tables_missed),
    ("metadata_missing", _w_metadata_missing),
    ("text_empty_pages", _w_text_empty_pages),
)
WARNING_IDS: frozenset[str] = frozenset(w for w, _ in WARNINGS)


def warnings_for(record: dict[str, Any]) -> list[RecordWarning]:
    out = [RecordWarning(wid, text) for wid, fn in WARNINGS if (text := fn(record)) is not None]
    if record.get("stale"):
        out.append(RecordWarning("stale", "Recorded for an earlier reading of this file"))
    return out


def summary_line(record: dict[str, Any]) -> str:
    """One line for the queue's finished row: what was found, with nothing invented."""
    parts: list[str] = []
    refs = record["references"]
    if record["states"]["references"] in ("found", "partial"):
        parts.append(_plural(int(refs["entries"]), "reference", "references"))
    elif record["states"]["references"] == "none_found":
        parts.append("no references")
    if record["states"]["figures"] in ("found", "none_found"):
        parts.append(_plural(int(record["figures"].get("found") or 0), "figure", "figures"))
    if record["states"]["tables"] in ("found", "partial", "none_found"):
        parts.append(_plural(int(record["tables"]["extracted"]), "table", "tables"))
    return " · ".join(parts)


# --- writing ---------------------------------------------------------------------------------- #


def _cached_text(doc: Document) -> tuple[str | None, Path | None]:
    from doc_assistant.ingest.cache import get_cache_path

    for path in (
        Path(doc.source_cache) if doc.source_cache else None,
        get_cache_path(Path(doc.source_original)),
    ):
        if path is None:
            continue
        try:
            if path.exists():
                return path.read_text(encoding="utf-8"), path
        except OSError as e:
            log.warning("record_cache_unreadable", file=str(path), error=str(e))
    return None, None


def _latest(session: Session, document_id: str) -> ExtractionRecord | None:
    return session.execute(
        select(ExtractionRecord)
        .where(ExtractionRecord.document_id == document_id)
        .order_by(ExtractionRecord.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def write_record(
    document_id: str,
    run_kind: str,
    passes: dict[str, tuple[str, dict[str, Any]]],
) -> ExtractionRecord | None:
    """Write one record for ``document_id``. ``passes`` maps ``references``/``figures`` to the
    ``(status, facts)`` this run produced; a pass absent from it is carried from the newest row.
    Tables are read from the cache as it stands: the splice runs with extraction, so they
    describe the text the document has now, whichever run wrote it."""
    from doc_assistant.ingest.cache import read_tables_record
    from doc_assistant.library.documents import effective_metadata

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        meta = session.get(DocumentMeta, document_id)
        title, authors, year = effective_metadata(doc, meta)
        is_pdf = (doc.format or "").lower() == "pdf"
        text, cache_path = _cached_text(doc)
        previous = _latest(session, document_id)
        prev_detail: dict[str, Any] = json.loads(previous.detail_json) if previous else {}

        detail: dict[str, Any] = {
            "text": text_facts(text)
            if text is not None
            else {"pages": 0, "chars": 0, "empty_pages": []},
            "metadata": {
                "title": bool(title and title.strip()),
                "authors": bool(authors and authors.strip()),
                "year": year is not None,
                "doi": bool(doc.doi),
            },
        }
        states: dict[str, str] = {}
        for name in ("references", "figures"):
            if name in passes:
                status, facts = passes[name]
                detail[name] = {**facts, "reason": facts.get("reason")}
                states[name] = (references_state if name == "references" else figures_state)(
                    status, facts
                )
            elif previous is not None:
                detail[name] = {
                    **prev_detail.get(name, {}),
                    "carried_from": previous.created_at.isoformat(),
                }
                states[name] = str(getattr(previous, f"{name}_state"))
            else:
                detail[name] = {"reason": "not looked for yet"}
                states[name] = "not_run"

        splice = read_tables_record(cache_path) if cache_path is not None else None
        tables = table_facts(text or "", splice, is_pdf=is_pdf)
        detail["tables"] = tables
        states["tables"] = tables_state(tables, is_pdf=is_pdf, ran=splice is not None)

        row = ExtractionRecord(
            document_id=document_id,
            doc_hash=doc.doc_hash,
            run_kind=run_kind,
            record_version=RECORD_VERSION,
            references_state=states["references"],
            figures_state=states["figures"],
            tables_state=states["tables"],
            detail_json=json.dumps(detail),
        )
        session.add(row)
        session.flush()
        _prune(session, document_id)
        session.expunge(row)
        return row


def _prune(session: Session, document_id: str) -> None:
    keep = [
        r[0]
        for r in session.execute(
            select(ExtractionRecord.id)
            .where(ExtractionRecord.document_id == document_id)
            .order_by(ExtractionRecord.created_at.desc())
            .limit(RECORDS_KEPT)
        ).all()
    ]
    session.execute(
        delete(ExtractionRecord).where(
            ExtractionRecord.document_id == document_id, ExtractionRecord.id.not_in(keep)
        )
    )


def run_index_passes(
    document_ids: Iterable[str],
    on_found: Callable[[str, str, str | None], None] | None = None,
) -> int:
    """Run the figure and reference passes on documents this index run produced, and record them.

    The passes are *Re-run…*'s own functions (`reingest`). Each is caught on its own, so one
    unreadable figure page still leaves the references and the record. ``on_found`` is told each
    document's source path, its one-line summary and its first warning, for the indexing screen's
    row. Returns the records written.
    """
    from doc_assistant import reingest

    written = 0
    for doc in reingest._load_docs(list(document_ids)):
        passes: dict[str, tuple[str, dict[str, Any]]] = {}
        for name, fn in (
            ("figures", reingest._rerun_figures),
            ("references", reingest._rerun_references),
        ):
            facts: dict[str, Any] = {}
            try:
                status, detail = fn(doc, facts)
            except Exception as e:  # a pass must not lose a completed ingest
                log.warning("index_pass_failed", document_id=doc.id, part=name, error=str(e))
                status, detail = "error", f"{type(e).__name__}: {e}"
            if status != "ok":
                facts["reason"] = detail
            passes[name] = (status, facts)
        try:
            row = write_record(doc.id, "index", passes)
            if row is not None:
                written += 1
                if on_found is not None:
                    rec = as_dict(row, row.doc_hash)
                    first = rec["warnings"][0]["text"] if rec["warnings"] else None
                    on_found(doc.source_original, rec["summary"], first)
        except Exception as e:
            log.warning("extraction_record_failed", document_id=doc.id, error=str(e))
    return written


# --- reading ---------------------------------------------------------------------------------- #


def as_dict(row: ExtractionRecord, current_hash: str | None) -> dict[str, Any]:
    detail = json.loads(row.detail_json)
    record: dict[str, Any] = {
        "created_at": row.created_at.isoformat(),
        "run_kind": row.run_kind,
        "states": {
            "references": row.references_state,
            "figures": row.figures_state,
            "tables": row.tables_state,
        },
        "stale": current_hash is not None and row.doc_hash != current_hash,
        **{k: detail.get(k, {}) for k in ("text", "metadata", *_PASSES)},
    }
    record["warnings"] = [{"id": w.id, "text": w.text} for w in warnings_for(record)]
    record["summary"] = summary_line(record)
    return record


def current_record(document_id: str) -> dict[str, Any] | None:
    """The newest record for a document, read for the screens, or None when never recorded."""
    with session_scope() as session:
        row = _latest(session, document_id)
        if row is None:
            return None
        doc = session.get(Document, document_id)
        return as_dict(row, doc.doc_hash if doc else None)


def latest_records(session: Session, document_ids: set[str] | None) -> dict[str, dict[str, Any]]:
    """The newest record per document, read in one grouped query (never one per document)."""
    newest = (
        select(
            ExtractionRecord.document_id.label("doc_id"),
            func.max(ExtractionRecord.created_at).label("at"),
        )
        .group_by(ExtractionRecord.document_id)
        .subquery()
    )
    query = (
        select(ExtractionRecord, Document.doc_hash)
        .join(
            newest,
            (ExtractionRecord.document_id == newest.c.doc_id)
            & (ExtractionRecord.created_at == newest.c.at),
        )
        .join(Document, Document.id == ExtractionRecord.document_id)
    )
    out: dict[str, dict[str, Any]] = {}
    for row, current_hash in session.execute(query).all():
        if document_ids is not None and row.document_id not in document_ids:
            continue
        out[row.document_id] = as_dict(row, current_hash)
    return out
