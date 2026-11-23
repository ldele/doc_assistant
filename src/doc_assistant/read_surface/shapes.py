"""The read surface's result shapes, under one version number (ADR-066 choice 8).

These are what an outside AI program receives once A4 serves them, so they are a promise: a later
version may add a field, and never removes one or changes what a field means without a new
``SURFACE_VERSION``. `tests/unit/test_read_surface_shapes.py` holds a snapshot of every field;
a removed or renamed field fails it.

Every item names its source: a document by ``document_id`` (and its title, so a reader need not
look it up), a passage by ``passage_id`` and page, a conversation by ``conversation_id``, a
concept by ``concept_id``. A list result carries its ``total``, so an empty library reads as zero
items of zero, never as an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

#: The version of every shape below. Raised only when a field is removed or changes meaning.
SURFACE_VERSION = 1


@dataclass(frozen=True)
class DocumentItem:
    """A document as a list shows it. ``title`` falls back to the file name, never empty."""

    document_id: str
    title: str
    filename: str
    authors: str | None
    year: int | None
    format: str
    pages: int | None
    folders: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentPage:
    """One page of `list_documents`: ``total`` counts every match, ``documents`` this page."""

    documents: list[DocumentItem]
    total: int
    offset: int
    limit: int
    folder_id: str | None = None
    version: int = SURFACE_VERSION


@dataclass(frozen=True)
class FolderItem:
    folder_id: str
    name: str
    documents: int


@dataclass(frozen=True)
class FolderList:
    folders: list[FolderItem]
    total: int
    version: int = SURFACE_VERSION


@dataclass(frozen=True)
class ReferenceItem:
    """One entry of a document's reference list, as extraction parsed it (not vouched for).

    ``in_library_document_id`` is set when the reference is a document in this library.
    ``addressed_to_ai`` is what the text check found in ``text`` (A8'), as on a passage."""

    text: str | None
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    in_library_document_id: str | None
    addressed_to_ai: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FigureItem:
    """A figure or table found in a document: its page, its kind and its caption, if any.
    ``addressed_to_ai`` is what the text check found in the caption (A8')."""

    page: int
    kind: str
    caption: str | None
    addressed_to_ai: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentRecord:
    """One document's record: its metadata, reference list and figures' captions.

    ``references_total`` and ``figures_total`` count everything on record; the lists may be
    capped. ``text_addressed_to_ai`` is how many paragraphs the current text check marked
    (A2b), ``None`` when the document has no current check."""

    document: DocumentItem
    doi: str | None
    references: list[ReferenceItem]
    references_total: int
    figures: list[FigureItem]
    figures_total: int
    text_addressed_to_ai: int | None
    version: int = SURFACE_VERSION


@dataclass(frozen=True)
class Passage:
    """A passage a search returned. ``fenced`` is the passage's text inside the fence the chat's
    prompt uses (S-9), sent once: the text between the tags is the passage as stored.
    ``addressed_to_ai`` names what the text check found in it (A2): a passage written to an AI,
    not to a reader. ``page`` is the page it starts on, derived from the document's page markers
    when the passage did not store one; ``None`` for a format without pages. ``score`` is the
    reranker's, comparable only within one result."""

    passage_id: str | None
    document_id: str | None
    title: str
    authors: str | None
    year: int | None
    page: int | None
    section: str | None
    score: float
    fenced: str
    addressed_to_ai: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PassageContext:
    """A passage with the text around it, read from the document (``get_passage``).

    ``before``, ``fenced`` and ``after`` are each fenced document text; ``fenced`` is the passage
    itself. ``at_document_start`` / ``at_document_end`` say the window reached an end of the
    document rather than its size."""

    passage_id: str
    document_id: str
    title: str
    page: int | None
    before: str
    fenced: str
    after: str
    at_document_start: bool
    at_document_end: bool
    addressed_to_ai: list[str] = field(default_factory=list)
    version: int = SURFACE_VERSION


@dataclass(frozen=True)
class ConceptItem:
    """A concept as `list_concepts` shows it: its id, its name and how many forms it has."""

    concept_id: str
    name: str
    forms: int
    has_definition: bool


@dataclass(frozen=True)
class ConceptList:
    concepts: list[ConceptItem]
    total: int
    version: int = SURFACE_VERSION


@dataclass(frozen=True)
class SearchResult:
    """What `search_library` found. ``note`` says why the list is empty when it is."""

    question: str
    passages: list[Passage]
    count: int
    note: str | None = None
    folder_id: str | None = None
    version: int = SURFACE_VERSION


@dataclass(frozen=True)
class LabelDocument:
    document_id: str
    title: str


@dataclass(frozen=True)
class LabelFolder:
    folder_id: str
    name: str


@dataclass(frozen=True)
class ConversationLabels:
    """The documents and folders a conversation's answers cited (ADR-066 choice 7), read from
    its answer records each time; nothing is stored for them."""

    documents: list[LabelDocument] = field(default_factory=list)
    folders: list[LabelFolder] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationItem:
    conversation_id: str
    title: str
    turns: int
    started_at: datetime
    last_at: datetime
    labels: ConversationLabels


@dataclass(frozen=True)
class ConversationList:
    """One page of `list_conversations`, newest first; ``total`` counts every match."""

    conversations: list[ConversationItem]
    total: int
    offset: int
    limit: int
    version: int = SURFACE_VERSION


@dataclass(frozen=True)
class CitedPassage:
    """A passage an answer was given, in the order its ``[n]`` markers number them.
    ``addressed_to_ai`` is what the text check finds in the excerpt today (A8')."""

    n: int
    document_id: str | None
    title: str
    page: int | None
    section: str | None
    excerpt: str
    addressed_to_ai: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TurnScope:
    """What a turn was limited to (A7b, G6). ``folders`` are the folders it asked for, by name
    (``None`` for one that no longer exists); ``concept`` the concept whose documents it searched;
    ``documents`` how many documents it searched, each counted once."""

    folders: list[str | None]
    concept: str | None
    documents: int


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str
    asked_at: datetime
    cited: list[CitedPassage]
    #: The folder the turn was limited to, by name, when it was limited to exactly one folder;
    #: ``None`` for the whole library, several folders or a concept (read ``scope`` for those).
    folder: str | None = None
    #: What the turn was limited to; ``None`` for the whole library (added in A7b, same version).
    scope: TurnScope | None = None


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    title: str
    turns: list[Turn]
    labels: ConversationLabels
    version: int = SURFACE_VERSION


@dataclass(frozen=True)
class DefinitionSource:
    """Where a chosen definition came from: ``kind`` is ``user`` (the person wrote it) or
    ``passage`` (a sentence of a document, named here)."""

    kind: str
    document_id: str | None = None
    title: str | None = None
    passage_id: str | None = None


@dataclass(frozen=True)
class ConceptRecord:
    """A concept the person has taken on: its name, its forms and the definition they chose
    (ADR-054). Exact forms always mean the concept; broad ones sometimes do."""

    concept_id: str
    name: str
    exact_forms: list[str]
    broad_forms: list[str]
    definition: str | None
    definition_source: DefinitionSource | None
    version: int = SURFACE_VERSION
