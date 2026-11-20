"""The read surface's shapes are a promise (ADR-066 choice 8): this snapshot fails when a field is
removed or renamed. Adding a field is allowed: append it to its entry here in the same change.
Removing or renaming one needs a new `SURFACE_VERSION`, and this snapshot rewritten under it."""

from __future__ import annotations

import dataclasses
import inspect

from doc_assistant.read_surface import shapes

SNAPSHOT_VERSION = 1

SNAPSHOT: dict[str, tuple[str, ...]] = {
    "CitedPassage": ("n", "document_id", "title", "page", "section", "excerpt"),
    "ConceptItem": ("concept_id", "name", "forms", "has_definition"),
    "ConceptList": ("concepts", "total", "version"),
    "ConceptRecord": (
        "concept_id",
        "name",
        "exact_forms",
        "broad_forms",
        "definition",
        "definition_source",
        "version",
    ),
    "ConversationItem": ("conversation_id", "title", "turns", "started_at", "last_at", "labels"),
    "ConversationLabels": ("documents", "folders"),
    "ConversationList": ("conversations", "total", "offset", "limit", "version"),
    "ConversationRecord": ("conversation_id", "title", "turns", "labels", "version"),
    "DefinitionSource": ("kind", "document_id", "title", "passage_id"),
    "DocumentItem": (
        "document_id",
        "title",
        "filename",
        "authors",
        "year",
        "format",
        "pages",
        "folders",
    ),
    "DocumentPage": ("documents", "total", "offset", "limit", "folder_id", "version"),
    "DocumentRecord": (
        "document",
        "doi",
        "references",
        "references_total",
        "figures",
        "figures_total",
        "text_addressed_to_ai",
        "version",
    ),
    "FigureItem": ("page", "kind", "caption"),
    "FolderItem": ("folder_id", "name", "documents"),
    "FolderList": ("folders", "total", "version"),
    "LabelDocument": ("document_id", "title"),
    "LabelFolder": ("folder_id", "name"),
    "Passage": (
        "passage_id",
        "document_id",
        "title",
        "authors",
        "year",
        "page",
        "section",
        "score",
        "fenced",
        "addressed_to_ai",
    ),
    "PassageContext": (
        "passage_id",
        "document_id",
        "title",
        "page",
        "before",
        "fenced",
        "after",
        "at_document_start",
        "at_document_end",
        "addressed_to_ai",
        "version",
    ),
    "ReferenceItem": ("text", "title", "authors", "year", "doi", "in_library_document_id"),
    "SearchResult": ("question", "passages", "count", "note", "folder_id", "version"),
    "Turn": ("question", "answer", "asked_at", "cited", "folder", "scope"),
    "TurnScope": ("folders", "concept", "documents"),
}


def _shapes() -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(f.name for f in dataclasses.fields(obj))
        for name, obj in inspect.getmembers(shapes, inspect.isclass)
        if dataclasses.is_dataclass(obj) and obj.__module__ == shapes.__name__
    }


def test_the_version_is_the_snapshots() -> None:
    assert shapes.SURFACE_VERSION == SNAPSHOT_VERSION


def test_no_shape_lost_or_renamed_a_field() -> None:
    now = _shapes()
    for name, fields in SNAPSHOT.items():
        assert name in now, f"shape {name} is gone"
        assert now[name][: len(fields)] == fields, f"{name}: fields removed, renamed or reordered"


def test_every_shape_is_in_the_snapshot() -> None:
    assert set(_shapes()) == set(SNAPSHOT), "a new shape needs its snapshot entry"


def test_every_top_level_result_carries_the_version() -> None:
    results = (
        "DocumentPage",
        "FolderList",
        "DocumentRecord",
        "SearchResult",
        "ConversationList",
        "ConversationRecord",
        "ConceptRecord",
        "PassageContext",
        "ConceptList",
    )
    for name in results:
        assert "version" in SNAPSHOT[name], name
