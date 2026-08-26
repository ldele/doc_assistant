"""Citation queries (Phase 4) — resolved in-corpus citation edges, both directions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import structlog
from sqlalchemy import select

from doc_assistant.db.models import Citation, Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest.citations import (
    MIN_CONTAINED_TITLE_CHARS,
    resolution_is_credible,
)

log = structlog.get_logger(__name__)

__all__ = ["MIN_CONTAINED_TITLE_CHARS", "resolution_is_credible"]

# ============================================================
# Citation queries (Phase 4)
# ============================================================


@dataclass
class GraphNode:
    """One node in a citation subgraph."""

    id: str
    filename: str
    title: str | None
    is_center: bool


@dataclass
class GraphEdge:
    """One directed edge in a citation subgraph."""

    source: str
    target: str


@dataclass
class CitationGraph:
    """Result of `graph_subgraph` — typed alternative to dict[str, Any]."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass
class CitationEdge:
    """A single citation edge — source -> target, internal or external."""

    raw_text: str | None
    target_title: str | None
    target_authors: str | None
    target_year: int | None
    target_doi: str | None
    target_document_id: str | None  # None = external (not in library)
    target_filename: str | None  # convenience for UI
    extraction_method: str | None
    confidence: float | None


def _row_to_edge(row: Any) -> CitationEdge:
    return CitationEdge(
        raw_text=row.raw_citation_text,
        target_title=row.target_title,
        target_authors=row.target_authors,
        target_year=row.target_year,
        target_doi=row.target_doi,
        target_document_id=row.target_document_id,
        target_filename=row.target_filename,
        extraction_method=row.extraction_method,
        confidence=row.confidence,
    )


def cites_out(doc_id: str) -> list[CitationEdge]:
    """Return all citations this doc makes (papers it cites)."""
    from doc_assistant.db.models import Citation

    with session_scope() as session:
        stmt = (
            select(
                Citation.raw_citation_text,
                Citation.target_title,
                Citation.target_authors,
                Citation.target_year,
                Citation.target_doi,
                Citation.target_document_id,
                Document.filename.label("target_filename"),
                Citation.extraction_method,
                Citation.confidence,
            )
            .outerjoin(Document, Document.id == Citation.target_document_id)
            .where(Citation.source_document_id == doc_id)
            .order_by(Citation.target_year.desc().nulls_last(), Citation.target_authors)
        )
        return [_row_to_edge(r) for r in session.execute(stmt).all()]


# ============================================================
# The reference list (the Library document view's References block)
# ============================================================
# The bibliography as the *paper* carries it, not as the graph sees it: one list, every
# extracted reference in it, and the ones that are already in the library carrying the id
# that makes them a link. `document_connections` deliberately splits the same rows into
# resolved/unresolved for the exploration panel — a reader looking at a reference list
# wants neither split nor a semantic neighbour.


# A publication year outside this window is extraction noise, not a year: the regex sometimes
# lifts an identifier or a page number out of a reference line (5 of this corpus's 4,282 parsed
# years land in 2034-2089). Structural bounds — nothing below is corpus-tuned — but they matter
# for ordering as much as for display: sorting newest-first would otherwise put every corrupt
# value at the top of the list, where it is the first thing the reader sees.
MIN_PLAUSIBLE_YEAR = 1800


def plausible_year(year: int | None, *, today: date | None = None) -> int | None:
    """The year if it can be a publication year, else ``None`` (unknown, shown as nothing)."""
    if year is None:
        return None
    latest = (today or date.today()).year + 1  # a paper can carry next year's date in press
    return year if MIN_PLAUSIBLE_YEAR <= year <= latest else None


@dataclass
class DocumentReference:
    """One entry in a document's reference list.

    ``target_document_id`` is the whole point: set ⇒ this reference is a document the user
    already owns, and the UI renders it as a link. ``title``/``authors``/``year``/``doi``
    are *extraction output* parsed from ``raw_text`` — shown as such, never as metadata the
    library vouches for. ``library_title`` is the owned document's own title, which is the
    trustworthy label when the two disagree.
    """

    raw_text: str | None
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    target_document_id: str | None
    target_filename: str | None
    library_title: str | None


@dataclass
class DocumentReferences:
    """A document's reference list + the counts that keep a capped list honest.

    ``total`` is every extracted reference, ``in_library`` how many resolve to an owned
    document, ``shown`` the length of ``references`` — so the UI can say "showing N of M"
    rather than truncating in silence.
    """

    references: list[DocumentReference]
    total: int
    in_library: int
    shown: int


# Payload bound for the reference list. A wire-size cap, not a corpus-tuned threshold: the
# full count travels as `total`, and `document_references` spends the budget on the resolved
# rows first, so raising or lowering it can never drop a reference the user owns.
REFERENCES_CAP = 200


def document_references(doc_id: str, *, cap: int = REFERENCES_CAP) -> DocumentReferences | None:
    """One document's reference list, or ``None`` if the document is unknown (⇒ 404).

    A pure read over the ``citations`` sidecar — no model, no network, no writes. A document
    with no extracted references returns an empty list, not ``None`` (the 0-doc contract):
    "this paper's bibliography was never extracted" is an ordinary state, not an error.

    **Every stored resolution is re-checked here** (``resolution_is_credible``). One that no
    longer agrees with the document it names keeps its place in the list and loses only its
    link — see that function for why the write side cannot be trusted on its own.

    **Order is year-descending, then authors** — the paper's own numbering is *not recorded*
    (``Citation`` has no ordinal column and its id is a uuid4, so insertion order is not
    recoverable), and the UI says so rather than implying the list is the paper's. Years that
    cannot be years are dropped first, so they sink instead of heading the list.

    **The cap never drops a reference the user owns:** linked rows are taken first, the
    remainder fills with the rest, and the merged list is re-sorted into one order.
    """
    with session_scope() as session:
        if session.get(Document, doc_id) is None:
            return None

    edges = cites_out(doc_id)
    # `cites_out` drops Citation.id, so re-read the owned rows' own title + doi by id — both
    # are what a stored resolution is checked against.
    library_meta: dict[str, tuple[str | None, str | None]] = {}
    owned_ids = {e.target_document_id for e in edges if e.target_document_id is not None}
    if owned_ids:
        with session_scope() as session:
            library_meta = {
                str(i): (t, d)
                for i, t, d in session.execute(
                    select(Document.id, Document.title, Document.doi).where(
                        Document.id.in_(owned_ids)
                    )
                ).all()
            }

    refs: list[DocumentReference] = []
    for e in edges:
        lib_title, lib_doi = library_meta.get(e.target_document_id or "", (None, None))
        linked = e.target_document_id is not None and resolution_is_credible(
            parsed_title=e.target_title,
            parsed_doi=e.target_doi,
            library_title=lib_title,
            library_doi=lib_doi,
        )
        refs.append(
            DocumentReference(
                raw_text=e.raw_text,
                title=e.target_title,
                authors=e.target_authors,
                year=plausible_year(e.target_year),
                doi=e.target_doi,
                target_document_id=e.target_document_id if linked else None,
                target_filename=e.target_filename if linked else None,
                library_title=lib_title if linked else None,
            )
        )

    # Re-sort on the *cleaned* year. Python's sort is stable, so `cites_out`'s author ordering
    # survives inside each year.
    refs.sort(key=lambda r: (r.year is None, -(r.year or 0)))
    in_library = sum(1 for r in refs if r.target_document_id is not None)

    if len(refs) > cap:
        owned = [(i, r) for i, r in enumerate(refs) if r.target_document_id is not None]
        rest = [(i, r) for i, r in enumerate(refs) if r.target_document_id is None]
        kept = owned[:cap] + rest[: max(0, cap - len(owned))]
        # Restore the single reading order the caller was promised.
        refs = [r for _, r in sorted(kept, key=lambda pair: pair[0])]

    return DocumentReferences(
        references=refs,
        total=len(edges),
        in_library=in_library,
        shown=len(refs),
    )


def cited_by(doc_id: str) -> list[tuple[str, str, str | None]]:
    """Return (source_doc_id, source_filename, raw_citation_text) for incoming citations."""
    from doc_assistant.db.models import Citation

    with session_scope() as session:
        stmt = (
            select(
                Document.id,
                Document.filename,
                Citation.raw_citation_text,
            )
            .join(Citation, Citation.source_document_id == Document.id)
            .where(Citation.target_document_id == doc_id)
            .order_by(Document.filename)
        )
        return [(str(r[0]), str(r[1]), r[2]) for r in session.execute(stmt).all()]


def graph_subgraph(doc_id: str, depth: int = 1) -> CitationGraph:
    """Return a CitationGraph centered on doc_id (internal edges only)."""
    from doc_assistant.db.models import Citation

    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    frontier = {doc_id}
    visited: set[str] = set()

    with session_scope() as session:
        center = session.execute(
            select(Document.id, Document.filename, Document.title).where(Document.id == doc_id)
        ).first()
        if center is None:
            return CitationGraph()
        nodes[doc_id] = GraphNode(
            id=doc_id, filename=center.filename, title=center.title, is_center=True
        )

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                outs = session.execute(
                    select(
                        Citation.target_document_id,
                        Document.filename,
                        Document.title,
                    )
                    .join(Document, Document.id == Citation.target_document_id)
                    .where(Citation.source_document_id == nid)
                    .where(Citation.target_document_id.is_not(None))
                ).all()
                for tgt_id, tgt_fn, tgt_title in outs:
                    if tgt_id not in nodes:
                        nodes[tgt_id] = GraphNode(
                            id=tgt_id, filename=tgt_fn, title=tgt_title, is_center=False
                        )
                        next_frontier.add(tgt_id)
                    edges.append(GraphEdge(source=nid, target=tgt_id))
                ins = session.execute(
                    select(
                        Citation.source_document_id,
                        Document.filename,
                        Document.title,
                    )
                    .join(Document, Document.id == Citation.source_document_id)
                    .where(Citation.target_document_id == nid)
                ).all()
                for src_id, src_fn, src_title in ins:
                    if src_id not in nodes:
                        nodes[src_id] = GraphNode(
                            id=src_id, filename=src_fn, title=src_title, is_center=False
                        )
                        next_frontier.add(src_id)
                    edges.append(GraphEdge(source=src_id, target=nid))
            frontier = next_frontier
            if not frontier:
                break

    edge_keys: set[tuple[str, str]] = set()
    deduped: list[GraphEdge] = []
    for e in edges:
        key = (e.source, e.target)
        if key not in edge_keys:
            edge_keys.add(key)
            deduped.append(e)
    return CitationGraph(nodes=list(nodes.values()), edges=deduped)


def reresolve_stored_citations(*, apply: bool = False) -> dict[str, int]:
    """Recompute ``target_document_id`` for every stored citation. Returns a before/after count.

    **Resolution was frozen at extraction time** (KI-45 defect 2): `target_document_id` is
    computed once, when the citing document is parsed, so each row records what the library looked
    like that day. The corpus has since grown and gained titles and DOIs, and — more importantly —
    the matcher itself has been corrected. Neither improvement reaches a stored row on its own,
    because `extract_citations` returns early for a document that already has rows.

    This pass closes that gap without re-extraction: it re-runs the *current* matcher over the
    *stored* parsed fields. No PDF is opened, no text is re-parsed, nothing is deleted. A row
    whose reference no longer resolves has its link cleared, which is the honest outcome — the
    paper still cites it, we simply no longer claim to hold it.

    Dry by default. ``apply=False`` reports what would change and writes nothing.
    """
    from doc_assistant.ingest.citations import ParsedCitation, match_to_library

    stats = {"rows": 0, "before": 0, "after": 0, "gained": 0, "lost": 0, "changed": 0}
    with session_scope() as session:
        rows = session.execute(select(Citation)).scalars().all()
        for row in rows:
            stats["rows"] += 1
            old = row.target_document_id
            if old:
                stats["before"] += 1
            parsed = ParsedCitation(
                raw_text=row.raw_citation_text or "",
                doi=row.target_doi,
                title=row.target_title,
                authors=row.target_authors,
                year=row.target_year,
                extraction_method=row.extraction_method or "",
                confidence=row.confidence or 0.0,
            )
            new = match_to_library(parsed)
            # A self-citation is not a link worth storing: a document's own reference list
            # resolving to itself tells the reader nothing and pollutes the citation graph.
            if new == row.source_document_id:
                new = None
            if new:
                stats["after"] += 1
            if new != old:
                if old and not new:
                    stats["lost"] += 1
                elif new and not old:
                    stats["gained"] += 1
                else:
                    stats["changed"] += 1
                if apply:
                    row.target_document_id = new
        if not apply:
            session.rollback()
    log.info("citations_reresolved", **stats, applied=apply)
    return stats
