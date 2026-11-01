"""Letter-spacing per document, and the ignore list behind it (ADR-056 points 5, 7, 9 and 10).

What the app shows: for one document, every letter-spaced run its extraction found (the record
written beside the cache, point 7), what was written into the text for it, and what the user's
ignore-list entries now ask for. What the app changes: the ignore list itself, one run, one
document or all of it. A change takes effect when that document's text is re-run, because the
ignore list goes into the document's cache fingerprint (`ingest.cache.document_fingerprint`).

The list is the user's own file (`config.REJOIN_IGNORE_PATH`), keyed by each PDF's sha256 so an
entry follows the file through a rename. Nothing here writes to the cache, the chunk store or the
registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from doc_assistant import config
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.extractors import KEEP_ALL_SPACED, user_spacing_is_valid
from doc_assistant.fsutil import atomic_write_text

log = structlog.get_logger(__name__)

#: Padding around a run's box for its crop, in points: enough of the line either side to read
#: it in place, never the whole page. Structural (a crop, not a page).
CROP_PAD_X = 36.0
CROP_PAD_Y = 8.0


class SpacingListError(RuntimeError):
    """The ignore list cannot be read, so it is not overwritten: the message says where it is."""


@dataclass(frozen=True)
class SpacingRow:
    """One letter-spaced run of a document, however many pages repeat it.

    ``state`` is what extraction wrote (``closed``, ``kept``, ``yours``, ``maths``); ``pending``
    is what the ignore list now asks for when that differs, which happens until the document's
    text is re-run; ``chosen`` is the user's spacing when there is one."""

    spaced: str
    rule: str
    written: str
    kind: str
    label: str
    state: str
    pending: str | None
    chosen: str | None
    pages: list[int]
    crop: list[float] | None


@dataclass(frozen=True)
class DocumentSpacing:
    """A document's letter-spacing, as the document page's block shows it."""

    document_id: str
    filename: str
    pageable: bool
    keep_all: bool
    rows: list[SpacingRow] = field(default_factory=list)


@dataclass(frozen=True)
class IgnoreListEntry:
    """One document's entries in the ignore list, as Settings lists them."""

    file: str
    sha256: str
    size: int
    document_id: str | None
    keep_all: bool
    kept: list[str]
    spacing: dict[str, str]


def crop_box(box: list[float]) -> list[float]:
    """A run's box padded to show it in its line (``CROP_PAD_X`` / ``CROP_PAD_Y``)."""
    x0, y0, x1, y1 = box
    return [max(0.0, x0 - CROP_PAD_X), max(0.0, y0 - CROP_PAD_Y), x1 + CROP_PAD_X, y1 + CROP_PAD_Y]


def _document(document_id: str) -> tuple[str, str, str, str | None] | None:
    """``(filename, format, source_original, source_cache)``, or None for an unknown document."""
    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        return doc.filename, (doc.format or ""), doc.source_original, doc.source_cache


def document_spacing(document_id: str) -> DocumentSpacing | None:
    """The letter-spacing block of one document, or None when the document is unknown.

    Rows come from the record its last extraction wrote, one per spaced form (a running head
    repeated on 16 pages is one row, with 16 pages), and only for runs that reached the text.
    Entries in the ignore list that name no recorded run are shown too, without pages."""
    from doc_assistant.ingest.cache import get_cache_path, read_spacing_record, rejoin_overrides
    from doc_assistant.library.documents import resolve_source_path

    found = _document(document_id)
    if found is None:
        return None
    filename, fmt, source_original, source_cache = found
    if fmt.lower() != "pdf":
        return DocumentSpacing(document_id, filename, pageable=False, keep_all=False)
    path = resolve_source_path(source_original, filename)
    cached = Path(source_cache) if source_cache else get_cache_path(Path(source_original))
    keep, spacing = rejoin_overrides(path) if path is not None else (frozenset(), {})
    keep_all = KEEP_ALL_SPACED in keep

    grouped: dict[str, dict[str, Any]] = {}
    for run in read_spacing_record(cached):
        if not run.get("in_text"):
            continue
        row = grouped.setdefault(str(run["spaced"]), {**run, "pages": []})
        row["pages"].append(int(run.get("page", 0)))

    rows: list[SpacingRow] = []
    for spaced, run in grouped.items():
        chosen = spacing.get(spaced)
        if chosen is not None and not user_spacing_is_valid(spaced, chosen):
            chosen = None
        if keep_all or spaced in keep:
            wanted = "kept"
        elif chosen is not None:
            wanted = "yours"
        elif run.get("kind") == "maths":
            wanted = "maths"
        else:
            wanted = "closed"
        state = str(run.get("state", "closed"))
        changed = wanted != state or (wanted == "yours" and chosen != run.get("written"))
        box = run.get("box")
        rows.append(
            SpacingRow(
                spaced=spaced,
                rule=str(run.get("joined", spaced)),
                written=str(run.get("written", spaced)),
                kind=str(run.get("kind", "")),
                label=str(run.get("label", "")),
                state=state,
                pending=wanted if changed else None,
                chosen=chosen,
                pages=sorted(set(run["pages"])),
                crop=crop_box([float(v) for v in box])
                if isinstance(box, list) and len(box) == 4
                else None,
            )
        )
    for spaced in sorted((set(keep) - {KEEP_ALL_SPACED} | set(spacing)) - set(grouped)):
        rows.append(
            SpacingRow(
                spaced=spaced,
                rule=spaced,
                written=spaced,
                kind="",
                label="",
                state="unknown",
                pending="kept" if spaced in keep else "yours",
                chosen=spacing.get(spaced),
                pages=[],
                crop=None,
            )
        )
    return DocumentSpacing(document_id, filename, pageable=True, keep_all=keep_all, rows=rows)


# --- the ignore list ---------------------------------------------------------------------------


def _list_path() -> Path:
    return Path(config.REJOIN_IGNORE_PATH)


def _read_list() -> dict[str, Any]:
    """The ignore list as stored, or an empty one when there is no file. A file that exists but
    cannot be read raises: writing over it would lose entries the user may still want."""
    from doc_assistant.ingest.cache import ignore_entry_is_valid

    path = _list_path()
    if not path.exists():
        return {"version": 1, "documents": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SpacingListError(
            f"the letter-spacing ignore list at {path} cannot be read ({e}); fix or move it first"
        ) from e
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("documents", []), list)
        or not all(ignore_entry_is_valid(e) for e in data.get("documents", []))
    ):
        raise SpacingListError(
            f"the letter-spacing ignore list at {path} is not in the expected form"
        )
    data.setdefault("version", 1)
    data.setdefault("documents", [])
    return data


def _write_list(data: dict[str, Any]) -> None:
    from doc_assistant.ingest import cache

    data["documents"] = [
        e for e in data["documents"] if e.get("keep") or e.get("spacing")
    ]  # an entry that keeps nothing is no entry
    atomic_write_text(_list_path(), json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    cache._ignore_list_memo.clear()


def _identity(document_id: str) -> tuple[str, str, int]:
    """``(filename, sha256, size)`` of a document's PDF; raises LookupError when it has none."""
    from doc_assistant.library.documents import resolve_source_path
    from doc_assistant.sources_manifest import sha256_file

    found = _document(document_id)
    if found is None:
        raise LookupError("document not found")
    filename, fmt, source_original, _ = found
    if fmt.lower() != "pdf":
        raise LookupError("only a PDF has letter-spacing to keep")
    path = resolve_source_path(source_original, filename)
    if path is None:
        raise LookupError(f"the source file is not available: {source_original}")
    return filename, sha256_file(path), path.stat().st_size


def _entry(data: dict[str, Any], filename: str, sha: str, size: int) -> dict[str, Any]:
    for found in data["documents"]:
        if found.get("sha256") == sha:
            entry: dict[str, Any] = found
            entry["file"], entry["size"] = filename, size
            entry.setdefault("keep", [])
            entry.setdefault("spacing", {})
            return entry
    e = {"sha256": sha, "size": size, "file": filename, "keep": [], "spacing": {}}
    data["documents"].append(e)
    return e


def keep_run(document_id: str, spaced: str) -> None:
    """Keep one run of a document as extracted."""
    filename, sha, size = _identity(document_id)
    data = _read_list()
    e = _entry(data, filename, sha, size)
    if spaced not in e["keep"]:
        e["keep"].append(spaced)
    e["spacing"].pop(spaced, None)
    _write_list(data)


def set_run_spacing(document_id: str, spaced: str, chosen: str) -> None:
    """Space one run of a document the user's way; ValueError if ``chosen`` changes a letter."""
    if not user_spacing_is_valid(spaced, chosen):
        raise ValueError("a spacing may only add or remove spaces between the run's own letters")
    filename, sha, size = _identity(document_id)
    data = _read_list()
    e = _entry(data, filename, sha, size)
    e["spacing"][spaced] = chosen
    e["keep"] = [k for k in e["keep"] if k != spaced]
    _write_list(data)


def remove_run(document_id: str, spaced: str) -> None:
    """Give one run back to the rule: drop whatever the list held for it."""
    filename, sha, size = _identity(document_id)
    data = _read_list()
    e = _entry(data, filename, sha, size)
    e["keep"] = [k for k in e["keep"] if k != spaced]
    e["spacing"].pop(spaced, None)
    _write_list(data)


def set_keep_all(document_id: str, keep: bool) -> None:
    """Keep every run of a document as extracted, or stop doing so."""
    filename, sha, size = _identity(document_id)
    data = _read_list()
    e = _entry(data, filename, sha, size)
    e["keep"] = [k for k in e["keep"] if k != KEEP_ALL_SPACED] + (
        [KEEP_ALL_SPACED] if keep else []
    )
    _write_list(data)


def list_entries() -> list[IgnoreListEntry]:
    """Every document the ignore list names, matched to the library by file name and size."""
    from sqlalchemy import select

    from doc_assistant.library.documents import resolve_source_path

    data = _read_list()
    names = {str(e.get("file", "")) for e in data["documents"]}
    with session_scope() as session:
        docs = [
            (str(i), str(f), str(s))
            for i, f, s in session.execute(
                select(Document.id, Document.filename, Document.source_original).where(
                    Document.filename.in_(names)
                )
            ).all()
        ]
    by_name: dict[str, list[tuple[str, Path | None]]] = {}
    for doc_id, filename, source in docs:
        by_name.setdefault(filename, []).append((doc_id, resolve_source_path(source, filename)))
    out: list[IgnoreListEntry] = []
    for e in data["documents"]:
        keep = [str(k) for k in e.get("keep", [])]
        spacing = {str(k): str(v) for k, v in (e.get("spacing") or {}).items()}
        document_id = None
        for doc_id, path in by_name.get(str(e.get("file", "")), []):
            try:
                if path is not None and path.stat().st_size == e.get("size"):
                    document_id = doc_id
                    break
            except OSError:
                continue
        out.append(
            IgnoreListEntry(
                file=str(e.get("file", "")),
                sha256=str(e.get("sha256", "")),
                size=int(e.get("size", 0)),
                document_id=document_id,
                keep_all=KEEP_ALL_SPACED in keep,
                kept=[k for k in keep if k != KEEP_ALL_SPACED],
                spacing=spacing,
            )
        )
    return out


def remove_entries(sha256: str | None = None) -> dict[str, Any]:
    """Remove one document's entries (by its PDF's sha256), or every entry; returns the list as
    it was, for the undo that sends it back (:func:`restore_entries`)."""
    data = _read_list()
    before: dict[str, Any] = json.loads(json.dumps(data))
    data["documents"] = (
        [] if sha256 is None else [e for e in data["documents"] if e.get("sha256") != sha256]
    )
    _write_list(data)
    log.info("rejoin_ignore_removed", sha256=sha256 or "all", entries=len(before["documents"]))
    return before


def restore_entries(previous: dict[str, Any]) -> None:
    """Put back a list :func:`remove_entries` returned (the undo)."""
    from doc_assistant.ingest.cache import ignore_entry_is_valid

    documents = previous.get("documents") if isinstance(previous, dict) else None
    if not isinstance(documents, list) or not all(ignore_entry_is_valid(e) for e in documents):
        raise ValueError("not a letter-spacing ignore list")
    _write_list({"version": previous.get("version", 1), "documents": documents})
