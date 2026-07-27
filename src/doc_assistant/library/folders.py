"""Folder organisation (docs/specs/feature-corpus-folders.md, ADR-025 F1).

Flat named folders with bulk membership. Deleting a folder never deletes documents (spec D6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import func, select

from doc_assistant.db.models import Document, Folder
from doc_assistant.db.session import session_scope

log = structlog.get_logger(__name__)


# ============================================================
# Folders (docs/specs/feature-corpus-folders.md — ADR-025 F1)
# ============================================================
# Manual organisation over the previously dormant Folder/document_folders schema
# (0 rows before F1). Flat in v1: every folder is created at the root
# (parent_folder_id NULL) and no caller sets a parent — the hierarchical column
# stays unused until nesting is decided (spec D1), because "does scoping a parent
# include its children?" is a question F2's retrieval scoping would otherwise have
# to invent an answer to.
#
# Since F2 a folder also scopes chat retrieval (`folder_doc_hashes` below is the resolver).
# The is_archived lesson still binds in the other direction: a scoped turn must SAY it was
# scoped, in the provenance record and on the answer.


@dataclass
class FolderSummary:
    """One folder plus its live member count.

    ``doc_count`` counts **non-archived** members only, so it agrees with what
    ``list_documents`` puts in the grid (spec D5). Archived members keep their
    ``document_folders`` row and reappear if the document is un-archived.
    """

    id: str
    name: str
    description: str | None = None
    parent_id: str | None = None
    doc_count: int = 0


def _folder_doc_count(session: Any, folder_id: str) -> int:
    """Number of non-archived documents in ``folder_id``."""
    from doc_assistant.db.models import document_folders

    stmt = (
        select(func.count(func.distinct(document_folders.c.document_id)))
        .select_from(document_folders)
        .join(Document, Document.id == document_folders.c.document_id)
        .where(document_folders.c.folder_id == folder_id, Document.is_archived.is_(False))
    )
    return int(session.execute(stmt).scalar() or 0)


def _build_folder(session: Any, folder: Any) -> FolderSummary:
    return FolderSummary(
        id=str(folder.id),
        name=folder.name,
        description=folder.description,
        parent_id=folder.parent_folder_id,
        doc_count=_folder_doc_count(session, str(folder.id)),
    )


def list_folders() -> list[FolderSummary]:
    """Every folder with its member count, sorted by name (case-insensitive)."""
    with session_scope() as session:
        folders = [_build_folder(session, f) for f in session.execute(select(Folder)).scalars()]
    folders.sort(key=lambda f: f.name.casefold())
    return folders


def get_folder(folder_id: str) -> FolderSummary | None:
    """One folder by id, or None if unknown."""
    with session_scope() as session:
        folder = session.get(Folder, folder_id)
        if folder is None:
            return None
        return _build_folder(session, folder)


def _find_by_name(session: Any, name: str, exclude_id: str | None = None) -> Any:
    """The root folder whose name matches ``name`` case-insensitively, or None."""
    query = select(Folder).where(
        func.lower(Folder.name) == name.casefold(), Folder.parent_folder_id.is_(None)
    )
    if exclude_id is not None:
        query = query.where(Folder.id != exclude_id)
    return session.execute(query).scalars().first()


def create_folder(name: str, description: str | None = None) -> FolderSummary:
    """Create a folder at the root. Idempotent on the case-folded name.

    Name uniqueness is enforced **here**, not by the database: ``uq_folder_name_parent``
    is ``(name, parent_folder_id)`` and SQLite treats NULL parents as distinct, so the
    constraint never fires for root folders (spec D2/D4). Mirrors
    ``create_keyword_family``'s get-or-create so the route behaves the same way twice.
    """
    name = name.strip()
    if not name:
        raise ValueError("name must not be blank")
    with session_scope() as session:
        existing = _find_by_name(session, name)
        if existing is not None:
            return _build_folder(session, existing)
        folder = Folder(name=name, description=description)
        session.add(folder)
        session.flush()
        log.info("folder_created", folder_id=folder.id, name=name)
        return _build_folder(session, folder)


def rename_folder(folder_id: str, new_name: str) -> FolderSummary | None:
    """Rename a folder. None if the folder is unknown; ValueError on blank or collision."""
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("new_name must not be blank")
    with session_scope() as session:
        folder = session.get(Folder, folder_id)
        if folder is None:
            return None
        if _find_by_name(session, new_name, exclude_id=folder_id) is not None:
            raise ValueError(f"a folder named {new_name!r} already exists")
        folder.name = new_name
        session.flush()
        return _build_folder(session, folder)


def delete_folder(folder_id: str) -> bool:
    """Delete a folder. Returns True if it existed.

    Deletes the folder only — never a document. The ``document_folders`` rows go with it
    via ``ON DELETE CASCADE`` (``PRAGMA foreign_keys=ON``, ``db/session.py``); the documents
    themselves are untouched, so this is not an ADR-014 delete path (spec D6).
    """
    with session_scope() as session:
        folder = session.get(Folder, folder_id)
        if folder is None:
            return False
        session.delete(folder)
        log.info("folder_deleted", folder_id=folder_id, name=folder.name)
        return True


def _edit_membership(
    folder_id: str, document_ids: Sequence[str], *, add: bool
) -> FolderSummary | None:
    """Add or remove documents on a folder. Idempotent; unknown document ids are skipped."""
    with session_scope() as session:
        folder = session.get(Folder, folder_id)
        if folder is None:
            return None
        current = {d.id for d in folder.documents}
        for document_id in document_ids:
            if add:
                if document_id in current:
                    continue
                doc = session.get(Document, document_id)
                if doc is None:
                    continue  # inform-don't-block: a stale id skips, the batch continues
                folder.documents.append(doc)
                current.add(document_id)
            else:
                doc = next((d for d in folder.documents if d.id == document_id), None)
                if doc is not None:
                    folder.documents.remove(doc)
                    current.discard(document_id)
        session.flush()
        return _build_folder(session, folder)


def add_documents_to_folder(folder_id: str, document_ids: Sequence[str]) -> FolderSummary | None:
    """Add documents to a folder. None if the folder is unknown. Idempotent."""
    return _edit_membership(folder_id, document_ids, add=True)


def remove_documents_from_folder(
    folder_id: str, document_ids: Sequence[str]
) -> FolderSummary | None:
    """Remove documents from a folder. None if the folder is unknown. Idempotent."""
    return _edit_membership(folder_id, document_ids, add=False)


def folder_document_ids(folder_id: str) -> list[str]:
    """Ids of the non-archived documents in a folder ([] for an unknown folder)."""
    with session_scope() as session:
        folder = session.get(Folder, folder_id)
        if folder is None:
            return []
        return [d.id for d in folder.documents if not d.is_archived]


def folder_doc_hashes(folder_id: str) -> list[str]:
    """``doc_hash`` of every non-archived document in a folder ([] for an unknown folder).

    The retrieval-scope resolver (ADR-025 F2): chunks carry only ``doc_hash``, so this is the
    key that scopes both retrieval arms. An unknown folder returning ``[]`` is deliberate and
    load-bearing — an empty scope must retrieve nothing, never fall back to the whole corpus
    (docs/specs/feature-corpus-folders-scope.md S3). Archived members are excluded, matching
    ``folder_document_ids`` and the Library grid.
    """
    with session_scope() as session:
        folder = session.get(Folder, folder_id)
        if folder is None:
            return []
        return [d.doc_hash for d in folder.documents if not d.is_archived and d.doc_hash]
