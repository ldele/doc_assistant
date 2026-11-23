"""The record tools: folders, documents, one document, concepts (ADR-066 choice 3; list_concepts
added by the user's decision of 2026-11-17).

Reads of the library's records only: no model, no write, no chunk store. Each is honest on an
empty library (an empty list and a total of zero) and bounded by its page, not by the library:
`list_documents` counts in SQL and reads one page of rows, `get_document` caps its lists and
says how many there are in all.
"""

from __future__ import annotations

import json

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from doc_assistant.db.models import (
    BREADTH_BROAD,
    Concept,
    ConceptAlias,
    ConceptDefinition,
    Document,
    DocumentMeta,
    Figure,
    Folder,
    document_folders,
)
from doc_assistant.db.session import session_scope
from doc_assistant.library.documents import effective_metadata
from doc_assistant.read_surface.shapes import (
    ConceptItem,
    ConceptList,
    ConceptRecord,
    DefinitionSource,
    DocumentItem,
    DocumentPage,
    DocumentRecord,
    FigureItem,
    FolderItem,
    FolderList,
    ReferenceItem,
)
from doc_assistant.source_checks import addressed_to_ai

#: The most documents one `list_documents` call returns. A wire-size bound, not tuned to a corpus.
MAX_PAGE = 100
#: The default page.
DEFAULT_PAGE = 50
#: How many references and figures `get_document` lists; the totals count every one.
LIST_CAP = 200


def _page_bounds(offset: int, limit: int) -> tuple[int, int]:
    return max(0, offset), max(1, min(limit, MAX_PAGE))


def list_folders() -> FolderList:
    """Every folder, by name, with how many documents it holds."""
    from doc_assistant.library.folders import list_folders as folders

    items = [FolderItem(folder_id=f.id, name=f.name, documents=f.doc_count) for f in folders()]
    return FolderList(folders=items, total=len(items))


def _items(session: Session, docs: list[Document]) -> list[DocumentItem]:
    """Document rows as list items, with the person's own metadata edits applied (ADR-013)."""
    if not docs:
        return []
    ids = [d.id for d in docs]
    overrides = {
        m.document_id: m
        for m in session.execute(
            select(DocumentMeta).where(DocumentMeta.document_id.in_(ids))
        ).scalars()
    }
    names: dict[str, list[str]] = {}
    for doc_id, name in session.execute(
        select(document_folders.c.document_id, Folder.name)
        .join(Folder, Folder.id == document_folders.c.folder_id)
        .where(document_folders.c.document_id.in_(ids))
        .order_by(Folder.name)
    ).all():
        names.setdefault(str(doc_id), []).append(str(name))
    out: list[DocumentItem] = []
    for d in docs:
        title, authors, year = effective_metadata(d, overrides.get(d.id))
        out.append(
            DocumentItem(
                document_id=str(d.id),
                title=(title or "").strip() or d.filename,
                filename=d.filename,
                authors=authors,
                year=year,
                format=d.format,
                pages=d.page_count,
                folders=names.get(str(d.id), []),
            )
        )
    return out


def list_documents(
    *, folder_id: str | None = None, offset: int = 0, limit: int = DEFAULT_PAGE
) -> DocumentPage:
    """One page of the library's documents, by file name, optionally within one folder.

    An unknown folder is an empty page, as an empty folder is. Archived documents are not
    listed, as in the Library."""
    offset, limit = _page_bounds(offset, limit)
    with session_scope() as session:
        query = select(Document).where(Document.is_archived.is_(False))
        counter = select(func.count(Document.id)).where(Document.is_archived.is_(False))
        if folder_id is not None:
            member = select(document_folders.c.document_id).where(
                document_folders.c.folder_id == folder_id
            )
            query = query.where(Document.id.in_(member))
            counter = counter.where(Document.id.in_(member))
        total = int(session.execute(counter).scalar() or 0)
        docs = list(
            # Name, then id: two documents may share a file name, and a page must not move
            # between calls (A8', case 5).
            session.execute(
                query.order_by(Document.filename, Document.id).offset(offset).limit(limit)
            )
            .scalars()
            .all()
        )
        items = _items(session, docs)
    return DocumentPage(
        documents=items, total=total, offset=offset, limit=limit, folder_id=folder_id
    )


def get_document(document_id: str) -> DocumentRecord | None:
    """One document's record, or ``None`` when no document has this id."""
    from doc_assistant.document_checks import document_check
    from doc_assistant.library.citations import document_references

    with session_scope() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            return None
        (item,) = _items(session, [doc])
        doi = doc.doi
        figures_total = int(
            session.execute(
                select(func.count(Figure.id)).where(Figure.document_id == document_id)
            ).scalar()
            or 0
        )
        figures = [
            FigureItem(
                page=int(page),
                kind=str(kind),
                caption=caption,
                addressed_to_ai=addressed_to_ai(caption) if caption else [],
            )
            for page, kind, caption in session.execute(
                select(Figure.page, Figure.kind, Figure.caption)
                .where(Figure.document_id == document_id)
                .order_by(Figure.page, Figure.id)  # one order on every call (A8', case 5)
                .limit(LIST_CAP)
            ).all()
        ]
    refs = document_references(document_id, cap=LIST_CAP)
    references = [
        ReferenceItem(
            text=r.raw_text,
            title=r.title,
            authors=r.authors,
            year=r.year,
            doi=r.doi,
            in_library_document_id=r.target_document_id,
            addressed_to_ai=addressed_to_ai(r.raw_text) if r.raw_text else [],
        )
        for r in (refs.references if refs is not None else [])
    ]
    check = document_check(document_id)
    return DocumentRecord(
        document=item,
        doi=doi,
        references=references,
        references_total=refs.total if refs is not None else 0,
        figures=figures,
        figures_total=figures_total,
        text_addressed_to_ai=(
            check.n_marked if check is not None and check.state == "current" else None
        ),
    )


def get_concept(concept: str) -> ConceptRecord | None:
    """A concept the person has taken on, by id, by name or by one of its forms (case does not
    matter). ``None`` when nothing matches, and for a *term* (ADR-054): a row the person has not
    taken on is not offered as a concept."""
    wanted = concept.strip()
    if not wanted:
        return None
    taken_on = (Concept.kind == "concept", Concept.graph_include.is_(True))
    with session_scope() as session:
        row = session.get(Concept, wanted)
        if row is None or not (row.kind == "concept" and row.graph_include):
            row = session.execute(
                select(Concept)
                .where(*taken_on)
                .where(
                    or_(
                        func.lower(Concept.label) == wanted.lower(),
                        Concept.id.in_(
                            select(ConceptAlias.concept_id).where(
                                func.lower(ConceptAlias.alias) == wanted.lower()
                            )
                        ),
                    )
                )
                .order_by(func.lower(Concept.label) != wanted.lower(), Concept.label)
                .limit(1)
            ).scalar_one_or_none()
        if row is None:
            return None
        forms = sorted(row.aliases, key=lambda a: a.alias.casefold())
        chosen = session.execute(
            select(ConceptDefinition).where(
                ConceptDefinition.concept_id == row.id, ConceptDefinition.status == "chosen"
            )
        ).scalar_one_or_none()
        source: DefinitionSource | None = None
        definition = row.definition
        if chosen is not None:
            definition = chosen.text
            if chosen.source == "passage":
                provenance = json.loads(chosen.provenance_json or "{}")
                doc_id = provenance.get("document_id")
                doc = session.get(Document, doc_id) if doc_id else None
                source = DefinitionSource(
                    kind="passage",
                    document_id=doc_id,
                    title=((doc.title or "").strip() or doc.filename) if doc else None,
                    passage_id=provenance.get("chunk_key"),
                )
            else:
                source = DefinitionSource(kind="user")
        return ConceptRecord(
            concept_id=str(row.id),
            name=row.label,
            exact_forms=[a.alias for a in forms if a.breadth != BREADTH_BROAD],
            broad_forms=[a.alias for a in forms if a.breadth == BREADTH_BROAD],
            definition=definition,
            definition_source=source,
        )


def list_concepts() -> ConceptList:
    """The concepts the person has taken on, by name: what `get_concept` can return.

    Terms are not listed (ADR-054): a row the person has not taken on is not offered as a
    concept. Bounded by the person's own vocabulary, not by the library."""
    with session_scope() as session:
        rows = session.execute(
            select(Concept.id, Concept.label, Concept.definition)
            .where(Concept.kind == "concept", Concept.graph_include.is_(True))
            .order_by(func.lower(Concept.label))
        ).all()
        chosen = {
            str(cid)
            for cid in session.execute(
                select(ConceptDefinition.concept_id).where(ConceptDefinition.status == "chosen")
            ).scalars()
        }
        forms: dict[str, int] = {
            str(cid): int(n)
            for cid, n in session.execute(
                select(ConceptAlias.concept_id, func.count(ConceptAlias.id)).group_by(
                    ConceptAlias.concept_id
                )
            ).all()
        }
    items = [
        ConceptItem(
            concept_id=str(cid),
            name=str(label),
            forms=forms.get(str(cid), 0),
            has_definition=str(cid) in chosen or bool((definition or "").strip()),
        )
        for cid, label, definition in rows
    ]
    return ConceptList(concepts=items, total=len(items))
