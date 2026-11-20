"""The retrieval scope one turn ran under: its value object, its record shape and its wording.

ADR-025 F2 scoped a turn to one folder; ADR-066 choice 7 (session A7b,
`docs/specs/chat-scoped-questions.md`) lets it run on **several folders**, searched as one set of
documents, or on **one concept's documents** (its ``concept_presence``, exact forms only).

This module is the one place that knows the scope's shapes, so that every reader agrees:

- :class:`ScopeView` is what a turn carries and the window renders;
- :func:`scope_dict` writes ``answer_records.retrieval_scope_json``;
- :func:`read_scope` reads it back, in this shape **or** the one-folder shape written before A7b
  (``{folder_id, folder_name, doc_count}``), so an old record reads as a one-folder scope;
- :func:`scope_summary` and :func:`scope_label` are the words a scope is stated in.

It imports nothing from the answer path, so ``provenance``, ``conversations`` and the read
surface can all use it without a cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: The sentence a concept scope adds when the concept graph is out of date (G2). Pinned by a test.
GRAPH_STALE_NOTE = "The concept graph is out of date; rebuild it to refresh this list."


@dataclass(frozen=True)
class ScopeFolder:
    """One folder a turn asked for, as it stood when the turn ran.

    ``doc_count`` is this folder's own count, before the union. ``folder_name is None`` means the
    folder was unknown or deleted; a count of 0 means it held no documents. Either way it added
    nothing to the search, and the note names it as left out (G1)."""

    folder_id: str
    folder_name: str | None
    doc_count: int


@dataclass(frozen=True)
class ScopeConcept:
    """The concept a turn was scoped to (G2). ``graph_stale`` is the concept graph's staleness
    verdict at the time of the turn (`knowledge.concept_graph_view`)."""

    concept_id: str
    name: str
    graph_stale: bool = False


@dataclass(frozen=True)
class ScopeView:
    """The retrieval scope one turn ran under, render-ready; ``None`` = the whole library.

    A **content filter** (which documents), not a quality knob, which is why it rides beside
    ``RagOverrides`` rather than inside it (feature-corpus-folders-scope.md, S1). A turn has
    folders or a concept, never both. ``doc_count`` is the size of the union: the number of
    documents retrieval actually saw, each counted once.
    """

    folders: list[ScopeFolder] = field(default_factory=list)
    concept: ScopeConcept | None = None
    doc_count: int = 0

    @property
    def searched(self) -> list[ScopeFolder]:
        """The folders that added documents to the search."""
        return [f for f in self.folders if f.folder_name is not None and f.doc_count > 0]

    @property
    def dropped(self) -> list[ScopeFolder]:
        """The folders that added nothing: unknown, deleted or empty (G1)."""
        return [f for f in self.folders if f.folder_name is None or f.doc_count == 0]


def scope_dict(scope: ScopeView | None) -> dict[str, Any] | None:
    """``ScopeView`` → the JSON shape persisted in ``answer_records.retrieval_scope_json``.
    ``None`` stays ``None`` so an unscoped turn writes NULL, exactly like every pre-F2 row."""
    if scope is None:
        return None
    return {
        "folders": [
            {"folder_id": f.folder_id, "folder_name": f.folder_name, "doc_count": f.doc_count}
            for f in scope.folders
        ],
        "concept": (
            {
                "concept_id": scope.concept.concept_id,
                "name": scope.concept.name,
                "graph_stale": scope.concept.graph_stale,
            }
            if scope.concept is not None
            else None
        ),
        "doc_count": scope.doc_count,
    }


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _folder(raw: dict[str, Any]) -> ScopeFolder | None:
    folder_id = raw.get("folder_id")
    if not isinstance(folder_id, str):
        return None
    return ScopeFolder(
        folder_id=folder_id,
        folder_name=_str_or_none(raw.get("folder_name")),
        doc_count=_int(raw.get("doc_count")),
    )


def read_scope(raw: str | dict[str, Any] | None) -> ScopeView | None:
    """A record's scope, from its JSON text or its parsed value, in either shape.

    ``None``, empty text or JSON ``null`` → ``None`` (the whole library). The shape written before
    A7b, ``{folder_id, folder_name, doc_count}``, reads as a one-folder scope. Text that is not
    JSON, or a value that is neither shape, reads as an empty scope rather than ``None``: a turn
    whose record says it was scoped must never be shown as a whole-library one."""
    if raw is None:
        return None
    value: object = raw
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            value = json.loads(raw)
        except ValueError:
            return ScopeView()
    if value is None:
        return None
    if not isinstance(value, dict):
        return ScopeView()
    if "folders" in value or "concept" in value:
        folders = [
            f
            for f in (
                _folder(item) for item in value.get("folders") or [] if isinstance(item, dict)
            )
            if f is not None
        ]
        concept_raw = value.get("concept")
        concept = None
        if isinstance(concept_raw, dict) and isinstance(concept_raw.get("concept_id"), str):
            concept = ScopeConcept(
                concept_id=concept_raw["concept_id"],
                name=_str_or_none(concept_raw.get("name")) or concept_raw["concept_id"],
                graph_stale=bool(concept_raw.get("graph_stale")),
            )
        return ScopeView(folders=folders, concept=concept, doc_count=_int(value.get("doc_count")))
    old = _folder(value)
    if old is None:
        return ScopeView()
    return ScopeView(folders=[old], doc_count=old.doc_count)


# ============================================================
# The words (the spec's open question 1: settled in the build, pinned by a test)
# ============================================================


def _documents(n: int) -> str:
    return f"{n} document{'' if n == 1 else 's'}"


def _names(names: list[str]) -> str:
    return ", ".join(f"**{n}**" for n in names)


def scope_summary(scope: ScopeView) -> str:
    """The scope in one or more markdown sentences, the same on every surface that states it.

    Folders: "Searched **A**, **B** only: 14 documents." A folder that added nothing: "**C** held
    no documents and was left out.", or "A folder that no longer exists was left out." A concept:
    "Searched the documents where **X** is present: 6 documents.", and :data:`GRAPH_STALE_NOTE`
    when the graph is out of date. The desktop app's ``lib/chat/scope.ts`` says the same."""
    if scope.concept is not None:
        out = (
            f"Searched the documents where **{scope.concept.name}** is present: "
            f"{_documents(scope.doc_count)}."
        )
        return out + (f" {GRAPH_STALE_NOTE}" if scope.concept.graph_stale else "")
    searched = scope.searched
    if searched:
        parts = [
            f"Searched {_names([f.folder_name or '' for f in searched])} only: "
            f"{_documents(scope.doc_count)}."
        ]
    else:
        parts = ["Nothing was searched."]
    for f in scope.dropped:
        if f.folder_name is None:
            parts.append("A folder that no longer exists was left out.")
        else:
            parts.append(f"**{f.folder_name}** held no documents and was left out.")
    return " ".join(parts)


def scope_label(scope: ScopeView | None) -> str | None:
    """One plain-text line for surfaces that show a constraint rather than a full note (the A/B
    compare card). ``None`` on an unscoped run."""
    if scope is None:
        return None
    if scope.concept is not None:
        stale = "; the concept graph is out of date" if scope.concept.graph_stale else ""
        return (
            f"documents where {scope.concept.name} is present "
            f"({_documents(scope.doc_count)}{stale})"
        )
    searched = scope.searched
    if not searched:
        if len(scope.folders) == 1 and scope.folders[0].folder_name is None:
            return "a folder that no longer exists (0 documents)"
        return "no documents (every chosen folder was empty or gone)"
    label = f"{', '.join(f.folder_name or '' for f in searched)} ({_documents(scope.doc_count)})"
    left_out = [f.folder_name or "a folder that no longer exists" for f in scope.dropped]
    return label + (f"; left out: {', '.join(left_out)}" if left_out else "")
