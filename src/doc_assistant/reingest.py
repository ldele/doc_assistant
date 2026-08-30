"""Per-document, per-part re-ingest — the app's answer to "this one came out wrong" (ADR-048).

Settings offers one button: index the folder. That is the whole-corpus instrument, and it is the
wrong one for a single bad extraction. This module is the per-document form: pick a document, pick
the parts to re-derive, pay only for those.

**Only four parts exist, and the reason is measurement, not taste** (`docs/performance.md`):

* the four here have a genuine per-document form;
* `keywords` and `doc_vectors` (Connections) recompute corpus-wide by construction — scoping
  `extract_keywords` to one document was measured to save **4%** — and `epistemics` / `gaps` have
  no scope at all. Offering them per-document would spend a whole-corpus pass while implying
  otherwise, and would change results for documents the user did not select. They are declined by
  name in the UI rather than hidden, so a user who cannot find the button learns there is none.

**The parts differ in cost by four orders of magnitude**, which is why `ReingestPart` carries a
`cost` string and the caller is expected to show it *before* running: metadata is milliseconds,
text is tens of seconds and can be minutes on a scan.

**`text` is the only part that moves identity, and it cleans up after itself.** `ingest.main` runs
`cleanup_orphans_*` only when `files is None` — a per-document re-extract is precisely the
`files is not None` branch, so nothing sweeps the superseded chunks. Re-extraction changes the
text, which changes `doc_hash` (ADR-042); ADR-047's fallback keeps the row and its sidecars
attached, but the *old* hash's chunks would stay in both stores and stay retrievable.
`_rerun_text` therefore records the hash before it starts and purges it after, if it moved.

**Known duplication, recorded rather than discovered later (ADR-048's first consequence):**
`scripts/extract_doc_metadata.py`, `extract_citations.py` and `extract_figures.py` each still carry
their own copy of this per-document orchestration. Rewiring three working runners is its own
increment; until then a change to per-document metadata logic has two homes.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

#: Reported as `(done, total, current)` — the same shape `ingest.main` uses, so the API's status
#: object needs no second vocabulary.
ProgressFn = Callable[[int, int, str | None], None]


@dataclass(frozen=True)
class ReingestPart:
    """One re-runnable pass, with what it costs stated in the user's terms.

    `cost` is deliberately an order of magnitude in prose, not a prediction: the numbers behind it
    (`docs/performance.md`) are machine- and document-dependent, and a control that promised "14.7
    seconds" would be wrong more often than it was right. Re-read it whenever that record is
    re-measured.
    """

    id: str
    label: str
    #: What it re-derives, for the control's second line.
    blurb: str
    #: Typical cost for one document. Shown before anything runs.
    cost: str
    #: Whether re-running it changes `doc_hash` — true only for `text`, and the reason it needs a
    #: confirmation the cheap parts do not.
    moves_identity: bool


PARTS: tuple[ReingestPart, ...] = (
    ReingestPart(
        id="metadata",
        label="Metadata",
        blurb="Title, authors, year and DOI, re-read from the extracted text.",
        cost="instant",
        moves_identity=False,
    ),
    ReingestPart(
        id="figures",
        label="Figures",
        blurb="Find figure regions again and re-crop them. PDFs only.",
        cost="a few seconds",
        moves_identity=False,
    ),
    ReingestPart(
        id="references",
        label="References",
        blurb="Parse the reference list again and re-match it against your library.",
        cost="about 10 seconds",
        moves_identity=False,
    ),
    ReingestPart(
        id="text",
        label="Text & chunks",
        blurb="Extract the file again from scratch, then re-chunk and re-embed it.",
        cost="30 seconds to a few minutes — longer for a scanned document",
        moves_identity=True,
    ),
)

PART_IDS: frozenset[str] = frozenset(p.id for p in PARTS)

#: Passes that exist but have no honest per-document form (ADR-048). Named in the UI so their
#: absence reads as a decision rather than a missing feature.
CORPUS_WIDE_PASSES: tuple[str, ...] = ("Connections", "Keywords", "Epistemics", "Gaps")


class UnknownPart(ValueError):
    """A part id that is not in `PARTS` — the API maps it to a 400."""


@dataclass(frozen=True)
class PartOutcome:
    """What one part did to one document. `skipped` always carries its reason."""

    document_id: str
    filename: str
    part: str
    status: str  # "ok" | "skipped" | "error"
    detail: str


@dataclass(frozen=True)
class ReingestResult:
    outcomes: tuple[PartOutcome, ...]

    @property
    def ok(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "ok")

    @property
    def skipped(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "skipped")

    @property
    def errors(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "error")


@dataclass(frozen=True)
class _DocRow:
    """The document columns every part needs, read once per document."""

    id: str
    doc_hash: str
    filename: str
    source_original: str
    source_cache: str | None
    format: str


def _load_docs(document_ids: Sequence[str]) -> list[_DocRow]:
    """Read the target rows, preserving the caller's order and dropping unknown ids."""
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    wanted = list(dict.fromkeys(document_ids))
    if not wanted:
        return []
    with session_scope() as session:
        rows = session.execute(
            select(
                Document.id,
                Document.doc_hash,
                Document.filename,
                Document.source_original,
                Document.source_cache,
                Document.format,
            ).where(Document.id.in_(wanted))
        ).all()
    by_id = {
        str(r[0]): _DocRow(
            id=str(r[0]),
            doc_hash=str(r[1]),
            filename=str(r[2]),
            source_original=str(r[3]),
            source_cache=str(r[4]) if r[4] else None,
            format=str(r[5] or "").lower(),
        )
        for r in rows
    }
    return [by_id[i] for i in wanted if i in by_id]


def _cached_text(doc: _DocRow) -> str | None:
    """The extracted markdown for this document, or None if it was never cached.

    Reads what is already on disk and never extracts: three of the four parts are cheap precisely
    because they work from the cache, and silently paying for an extraction inside "re-run
    metadata" would defeat the whole point of splitting the parts up.
    """
    from doc_assistant.ingest.cache import get_cache_path

    candidates: list[Path] = []
    if doc.source_cache:
        candidates.append(Path(doc.source_cache))
    original = Path(doc.source_original)
    candidates.append(get_cache_path(original))
    for path in candidates:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except OSError as e:  # a locked or unreadable cache is a skip, not a crash
            log.warning("reingest_cache_unreadable", file=str(path), error=str(e))
    return None


def _source_path(doc: _DocRow) -> Path | None:
    """The document's source file, if it is reachable right now."""
    from doc_assistant import config

    p = Path(doc.source_original)
    if p.exists():
        return p
    fallback = config.DOCS_PATH / doc.filename
    return fallback if fallback.exists() else None


# --- the parts ------------------------------------------------------------------------------- #


def _rerun_metadata(doc: _DocRow) -> tuple[str, str]:
    """Re-read title/authors/year/DOI from the cached markdown.

    **Overwrites the extracted defaults, and only those.** ADR-013 keeps a user's own edits in the
    separate `DocumentMeta` override table and merges them at read time with the override winning,
    so `Document.title` and friends hold nothing but the extractor's previous answer. Replacing it
    is the whole point of asking for a re-run; the usual `--force` hazard does not apply here.
    """
    from sqlalchemy import update

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.metadata_extractor import extract_metadata

    text = _cached_text(doc)
    if text is None:
        return "skipped", "no extracted text cached — re-run Text & chunks first"

    meta = extract_metadata(text, filename=doc.filename)
    changes: dict[str, Any] = {}
    if meta.title:
        changes["title"] = meta.title
    if meta.authors:
        changes["authors"] = meta.authors
    if meta.year is not None:
        changes["year"] = meta.year
    if meta.doi:
        changes["doi"] = meta.doi
    if not changes:
        return "skipped", "the extractor found no metadata in this document"

    with session_scope() as session:
        session.execute(update(Document).where(Document.id == doc.id).values(**changes))
    return "ok", f"filled {', '.join(sorted(changes))}"


def _rerun_figures(doc: _DocRow) -> tuple[str, str]:
    """Detect figure regions again and re-crop them, replacing this document's rows and PNGs.

    Detection only — the VLM description pass costs money per figure and is never part of a
    checklist (ADR-048, and the KI-4 credit-leak lesson).

    The clear-then-write is unconditional so a document that legitimately drops to **zero** figures
    loses its stale rows: guarding it on `regions` is the bug that left `hebb_1949.pdf` holding 365
    rejected page-scan rows after the ceiling correctly rejected all of them.
    """
    from sqlalchemy import delete

    from doc_assistant.db.models import Figure
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.figures import (
        detect_figure_regions,
        figure_dir,
        figure_image_path,
        render_region,
    )

    if doc.format != "pdf":
        return "skipped", "figure detection is PDF-only"
    pdf = _source_path(doc)
    if pdf is None:
        return "skipped", "the source file is not reachable"

    regions = detect_figure_regions(str(pdf))

    existing = figure_dir(doc.doc_hash)
    if existing.exists():
        shutil.rmtree(existing, ignore_errors=True)

    import pymupdf

    rendered = 0
    rows: list[Figure] = []
    handle = pymupdf.open(str(pdf))  # type: ignore[no-untyped-call]
    try:
        per_page: dict[int, int] = {}
        for region in regions:
            index = per_page.get(region.page, 0)
            per_page[region.page] = index + 1
            image_path: str | None = None
            if region.bbox is not None:
                out = figure_image_path(doc.doc_hash, region.page, index)
                render_region(handle[region.page - 1], region.bbox, out, dpi=_figure_dpi())
                image_path = str(out)
                rendered += 1
            bbox = region.bbox
            rows.append(
                Figure(
                    document_id=doc.id,
                    doc_hash=doc.doc_hash,
                    page=region.page,
                    bbox_x0=bbox[0] if bbox else None,
                    bbox_y0=bbox[1] if bbox else None,
                    bbox_x1=bbox[2] if bbox else None,
                    bbox_y1=bbox[3] if bbox else None,
                    kind=region.kind,
                    caption=region.caption,
                    image_path=image_path,
                    extraction_method=region.extraction_method,
                )
            )
    finally:
        handle.close()  # type: ignore[no-untyped-call]

    with session_scope() as session:
        session.execute(delete(Figure).where(Figure.document_id == doc.id))
        for row in rows:
            session.add(row)

    if not rows:
        return "ok", "no figures found — previous figures cleared"
    return "ok", f"{len(rows)} figure(s), {rendered} cropped"


def _figure_dpi() -> int:
    from doc_assistant import config

    return int(config.FIGURE_RENDER_DPI)


def _rerun_references(doc: _DocRow) -> tuple[str, str]:
    """Parse the reference list again and re-match it against the library.

    Replaces this document's `Citation` rows wholesale — a re-run that merged would keep whatever
    the previous parse got wrong, which is the thing being re-run to fix.
    """
    from sqlalchemy import delete

    from doc_assistant.db.models import Citation
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.citations import (
        extract_from_markdown,
        load_library_candidates,
        match_to_library,
    )

    text = _cached_text(doc)
    if text is None:
        return "skipped", "no extracted text cached — re-run Text & chunks first"

    result = extract_from_markdown(doc.id, text)
    candidates = load_library_candidates()
    matched = 0
    with session_scope() as session:
        session.execute(delete(Citation).where(Citation.source_document_id == doc.id))
        for parsed in result.citations:
            target_id = match_to_library(parsed, candidates=candidates)
            if target_id:
                matched += 1
            session.add(
                Citation(
                    source_document_id=doc.id,
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
    if not result.citations:
        return "ok", "no reference list found — previous references cleared"
    return "ok", f"{len(result.citations)} reference(s), {matched} matched in your library"


def _rerun_text(doc: _DocRow) -> tuple[str, str]:
    """Extract the file again from scratch, re-chunk, re-embed — and sweep what it superseded.

    The sweep is not optional. `ingest.main` cleans orphans only when `files is None`, and this is
    the `files is not None` path, so the previous `doc_hash`'s chunks would otherwise stay in both
    stores and stay retrievable — the document silently indexed twice. ADR-047 keeps the row and
    its sidecars attached across the hash change; the chunks are the half that needs doing here.
    """
    from doc_assistant import ingest
    from doc_assistant.ingest.cache import _fingerprint_path, get_cache_path

    source = _source_path(doc)
    if source is None:
        return "skipped", "the source file is not reachable"

    before = doc.doc_hash
    # Invalidate this ONE document's cache. Deleting the fingerprint too, so a cache written by the
    # current extractor is not judged fresh on the next pass and quietly skipped.
    cached = Path(doc.source_cache) if doc.source_cache else get_cache_path(source)
    for path in (cached, _fingerprint_path(cached)):
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            return "error", f"could not clear the cached text ({type(e).__name__}: {e})"

    stats = ingest.main(files=[source])
    if stats.get("errors"):
        return "error", "extraction failed — the document is unchanged"

    after = _current_hash(doc.id)
    if after and after != before:
        removed = _purge_superseded_chunks(before)
        return "ok", f"re-extracted; {removed} superseded chunk(s) removed"
    return "ok", "re-extracted; the text was unchanged"


def _current_hash(document_id: str) -> str | None:
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        row = session.execute(
            select(Document.doc_hash).where(Document.id == document_id)
        ).scalar_one_or_none()
    return str(row) if row else None


def _purge_superseded_chunks(doc_hash: str) -> int:
    """Drop a superseded hash's chunks from **both** vector stores. Returns the count removed.

    Opened with ``embedding_function=None`` on purpose: deleting by a metadata filter embeds
    nothing, and constructing the real embedder here would load a model to run a ``DELETE``. Both
    stores, not just the live one — which of the two is live depends on ``USE_PARENT_CHILD``, and
    chunks left in the other would come back the moment that flag moved.
    """
    from langchain_chroma import Chroma

    from doc_assistant import config
    from doc_assistant.embeddings import get_active_model_name, get_collection_name

    collection = get_collection_name(get_active_model_name())
    removed = 0
    for path in (config.CHROMA_PATH, config.PC_CHROMA_PATH):
        try:
            store = Chroma(
                persist_directory=str(path),
                embedding_function=None,
                collection_name=collection,
            )
            found = store.get(where={"doc_hash": doc_hash}, include=[])
            ids = list(found.get("ids", []))
            if ids:
                store.delete(ids=ids)
                removed += len(ids)
        except Exception as e:  # a store that will not answer must not fail the whole re-run
            log.warning("reingest_purge_failed", doc_hash=doc_hash, path=str(path), error=str(e))
    return removed


_RUNNERS: dict[str, Callable[[_DocRow], tuple[str, str]]] = {
    "metadata": _rerun_metadata,
    "figures": _rerun_figures,
    "references": _rerun_references,
    "text": _rerun_text,
}


def rerun(
    document_ids: Sequence[str],
    parts: Sequence[str],
    on_progress: ProgressFn | None = None,
) -> ReingestResult:
    """Re-run `parts` for each of `document_ids`. One outcome per (document, part).

    Ordering is deliberate: parts run in `PARTS` order, so `text` runs **last** for a document.
    Re-extraction rewrites the cached markdown that metadata and references read, and running them
    first would derive from the text the user asked to replace.

    A part that raises is recorded as an error and the run continues — one unreadable PDF must not
    abandon a selection of forty.
    """
    unknown = [p for p in parts if p not in PART_IDS]
    if unknown:
        raise UnknownPart(f"unknown part(s): {', '.join(sorted(unknown))}")
    ordered = [p.id for p in PARTS if p.id in set(parts)]
    docs = _load_docs(document_ids)
    total = len(docs) * len(ordered)
    outcomes: list[PartOutcome] = []
    done = 0

    for doc in docs:
        for part in ordered:
            if on_progress is not None:
                on_progress(done, total, f"{doc.filename} · {part}")
            try:
                status, detail = _RUNNERS[part](doc)
            except Exception as e:
                log.warning("reingest_part_failed", document_id=doc.id, part=part, error=str(e))
                status, detail = "error", f"{type(e).__name__}: {e}"
            outcomes.append(
                PartOutcome(
                    document_id=doc.id,
                    filename=doc.filename,
                    part=part,
                    status=status,
                    detail=detail,
                )
            )
            done += 1
            if on_progress is not None:
                on_progress(done, total, f"{doc.filename} · {part}")

    return ReingestResult(outcomes=tuple(outcomes))
