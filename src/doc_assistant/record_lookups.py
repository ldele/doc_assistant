"""Record questions answered from the library's records, by fixed rule or named command (A6).

ADR-066 choice 5, `docs/specs/chat-record-lookups.md`. A question about the library itself (a
count, a folder, one document's metadata, what is missing, a curated concept) maps to **one
lookup**: a call of a `read_surface.records` function, or of `library_summary` for counts and
health (`query_router.library_lookup`, today's answers). No model and no embedding is called, so
the answer is the same on every model, and it ends with `RECORDS_NOTE`.

A named document is found by its recorded title, else its filename stem: every content word of
the span the question gives must be a word of it (decision 2). One match answers, several are
listed, none returns ``None`` and the question goes to retrieval. There is no score and no
threshold. A folder is found by its name, case aside; an unknown folder says so and lists the
folders, since the question was about a folder.

Gaps are not a lookup (decision 1): their signals are of uneven trust (`docs/knowledge-layer.md`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from doc_assistant.query_router import _TOPIC_WORDS, is_library_query, library_lookup
from doc_assistant.read_surface import records
from doc_assistant.read_surface.shapes import DocumentItem, FolderItem, FolderList

#: The answer's last line (decision 3, open question 2). Pinned by a test.
RECORDS_NOTE = "_From the library's records: no passage was read and no model was asked._"

#: How many names a list answer shows before it says how many more there are. A reading bound
#: for a chat reply, not tuned to a corpus.
SHOWN = 50

#: The commands that call a lookup and write a record (decision 4).
LOOKUP_COMMANDS = frozenset({"folder", "concept", "document"})


@dataclass(frozen=True)
class RecordAnswer:
    """A lookup's answer and what it read: what `AnswerRecord.records_json` stores.

    ``ids`` are the records the answer names (documents, folders or concepts); ``rows`` is how
    many rows the lookup returned; ``asked`` is set when several documents matched and the
    answer asks which one."""

    text: str
    lookup: str
    args: dict[str, Any] = field(default_factory=dict)
    ids: list[str] = field(default_factory=list)
    rows: int = 0
    asked: bool = False

    def as_record(self, via: str) -> dict[str, Any]:
        """The ``records_json`` payload; ``via`` is ``rule`` or ``command``."""
        out: dict[str, Any] = {
            "lookup": self.lookup,
            "args": self.args,
            "ids": self.ids,
            "rows": self.rows,
            "via": via,
        }
        if self.asked:
            out["asked"] = True
        return out


def _answer(text: str, lookup: str, **kw: Any) -> RecordAnswer:
    return RecordAnswer(text=text.rstrip() + "\n\n" + RECORDS_NOTE, lookup=lookup, **kw)


def _more(total: int, shown: int) -> str:
    return f"\n\n…and {total - shown} more." if total > shown else ""


# ============================================================
# Reading the question
# ============================================================

_DOC_NOUN = r"(?:documents?|files?|papers?|pdfs?|articles?)"
_FIELD = r"(?P<field>titles?|years?|authors?)"

#: Words of a named span that do not name the document (case aside).
_SPAN_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "for",
        "in",
        "on",
        "and",
        "to",
        "by",
        "with",
        "from",
        "my",
        "our",
        "its",
        "it",
        "this",
        "that",
        "these",
        "those",
        "paper",
        "papers",
        "document",
        "documents",
        "article",
        "articles",
        "file",
        "files",
        "pdf",
        "pdfs",
        "book",
        "books",
        "study",
        "library",
        "collection",
        "et",
        "al",
    ]
)

#: Quote marks around a name: straight, and the curly pairs.
_QUOTES = " ,'\"" + "".join(map(chr, (0x201C, 0x201D, 0x2018, 0x2019)))

#: What a question adds after the document's name to say where to look.
_SPAN_TAILS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r",?\s+according to (?:my|the) (?:library|collection)$",
        r",?\s+as (?:my|the) (?:library|collection) (?:records|has|lists) it$",
        r",?\s+(?:as )?recorded in (?:my|the) (?:library|collection)$",
        r",?\s+(?:in|from|within) (?:my|the) (?:library|collection)$",
    )
]

#: A topical word after a document noun makes it a content question: "the paper about X".
_TOPICAL_SPAN = re.compile(rf"\b{_DOC_NOUN}\s+(?:{_TOPIC_WORDS})\b", re.IGNORECASE)


def _normal(text: str) -> str:
    """One line, single spaces, no closing punctuation."""
    return " ".join(text.split()).rstrip(" ?.!")


def _strip_tails(span: str) -> str:
    changed = True
    while changed:
        changed = False
        for tail in _SPAN_TAILS:
            cut = tail.sub("", span)
            if cut != span:
                span, changed = cut.strip(), True
    return span.strip(_QUOTES)


def _words(text: str) -> list[str]:
    """Words, case and punctuation aside (an underscore splits a filename stem)."""
    return re.findall(r"[^\W_]+", text.casefold())


def _content_words(span: str) -> list[str]:
    return [w for w in _words(span) if w not in _SPAN_STOP]


# ============================================================
# The lookups
# ============================================================


def _all_documents() -> Iterator[DocumentItem]:
    """Every listed document, a page at a time (bounded by the page, not the library)."""
    offset = 0
    while True:
        page = records.list_documents(offset=offset, limit=records.MAX_PAGE)
        yield from page.documents
        offset += len(page.documents)
        if not page.documents or offset >= page.total:
            return


def _has_title(item: DocumentItem) -> bool:
    """The read surface shows the file name where no title is recorded."""
    return item.title != item.filename


def find_documents(span: str) -> list[DocumentItem]:
    """The documents a named span matches (decision 2): every content word of the span is a word
    of the recorded title, or of the filename stem where no title is recorded. No score."""
    wanted = _content_words(span)
    if not wanted:
        return []
    found = []
    for item in _all_documents():
        if _has_title(item):
            have = set(_words(item.title))
        else:
            have = set(_words(item.filename.rsplit(".", 1)[0]))
        if all(w in have for w in wanted):
            found.append(item)
    return found


def _doc_line(item: DocumentItem) -> str:
    title = f"*{item.title}* (`{item.filename}`)" if _has_title(item) else f"`{item.filename}`"
    return f"{title}, `/document {item.document_id[:8]}`"


def _several(span: str, matches: list[DocumentItem], field_name: str | None) -> RecordAnswer:
    lines = [f"Several documents match “{span}”:"]
    lines += [f"- {_doc_line(d)}" for d in matches[:SHOWN]]
    text = "\n".join(lines) + _more(len(matches), min(len(matches), SHOWN))
    text += "\n\nWhich one do you mean? Ask again with more of its title, or use its `/document`."
    return _answer(
        text,
        "list_documents",
        args={"name": span, "field": field_name},
        ids=[d.document_id for d in matches[:SHOWN]],
        rows=len(matches),
        asked=True,
    )


def _not_recorded(value: object) -> str:
    return str(value) if value not in (None, "") else "not recorded"


def _document_answer(item_id: str, span: str, field_name: str | None) -> RecordAnswer:
    """One document's metadata, the asked field first (`get_document`)."""
    rec = records.get_document(item_id)
    args = {"name": span, "field": field_name}
    if rec is None:  # removed between the match and the read
        return _answer("No document is recorded under that name.", "get_document", args=args)
    d = rec.document
    name = f"`{d.filename}`"
    title = d.title if _has_title(d) else None
    first = ""
    if field_name == "title":
        first = (
            f"The library records the title of {name} as **{title}**."
            if title
            else f"No title is recorded for {name}."
        )
    elif field_name == "authors":
        first = (
            f"The library records the authors of {name} as **{d.authors}**."
            if d.authors
            else f"No authors are recorded for {name}."
        )
    elif field_name == "year":
        first = (
            f"The library records **{d.year}** as the year of {name}."
            if d.year
            else f"No year is recorded for {name}."
        )
    elif field_name == "type":
        pages = f", {d.pages} pages" if d.pages else ""
        first = f"{name} is a **{d.format.upper()}** file{pages}."
    lines = [first] if first else [f"## {title or d.filename}"]
    lines += [
        "",
        f"- **Title:** {_not_recorded(title)}",
        f"- **Authors:** {_not_recorded(d.authors)}",
        f"- **Year:** {_not_recorded(d.year)}",
        f"- **Type:** {d.format.upper()}" + (f", {d.pages} pages" if d.pages else ""),
        f"- **Folders:** {', '.join(d.folders) if d.folders else 'none'}",
        f"- **DOI:** {_not_recorded(rec.doi)}",
        f"- **File:** `{d.filename}` · `/document {d.document_id[:8]}`",
    ]
    return _answer("\n".join(lines), "get_document", args=args, ids=[d.document_id], rows=1)


def document_by_name(span: str, field_name: str | None = None) -> RecordAnswer | None:
    """A named document's metadata, or the documents that match, or ``None`` when none does."""
    matches = find_documents(span)
    if not matches:
        return None
    if len(matches) > 1:
        return _several(span, matches, field_name)
    return _document_answer(matches[0].document_id, span, field_name)


def folders() -> RecordAnswer:
    """Every folder, with how many documents each holds (`list_folders`)."""
    listed = records.list_folders()
    if not listed.folders:
        return _answer("The library has no folders yet.", "list_folders")
    lines = [f"**{listed.total} folder(s):**", ""]
    lines += [f"- {f.name} ({f.documents} documents)" for f in listed.folders]
    return _answer(
        "\n".join(lines),
        "list_folders",
        ids=[f.folder_id for f in listed.folders],
        rows=listed.total,
    )


def _folder_matches(name: str, listed: FolderList) -> list[FolderItem]:
    """The folders named ``name``, case and surrounding quotes aside: A6's folder matcher."""
    wanted = name.strip().strip(_QUOTES).casefold()
    return [f for f in listed.folders if f.name.casefold() == wanted]


def _unknown_folder(name: str, listed: FolderList, args: dict[str, Any]) -> RecordAnswer:
    if listed.folders:
        known = ", ".join(f.name for f in listed.folders)
        text = f"No folder is named “{name}”. The folders are: {known}."
    else:
        text = f"No folder is named “{name}”: the library has no folders yet."
    return _answer(
        text,
        "list_folders",
        args=args,
        ids=[f.folder_id for f in listed.folders],
        rows=listed.total,
    )


def folder(name: str, *, count_only: bool = False) -> RecordAnswer:
    """One folder's documents, or its count; an unknown folder says so and lists the folders."""
    listed = records.list_folders()
    args: dict[str, Any] = {"folder": name, "count_only": count_only}
    hit = next(iter(_folder_matches(name, listed)), None)
    if hit is None:
        return _unknown_folder(name, listed, args)
    args["folder_id"] = hit.folder_id
    if count_only:
        page = records.list_documents(folder_id=hit.folder_id, limit=1)
        return _answer(
            f"The folder **{hit.name}** holds **{page.total}** documents.",
            "list_documents",
            args=args,
            ids=[hit.folder_id],
            rows=page.total,
        )
    page = records.list_documents(folder_id=hit.folder_id, limit=SHOWN)
    if page.total == 0:
        return _answer(
            f"The folder **{hit.name}** holds no documents.", "list_documents", args=args
        )
    lines = [f"The folder **{hit.name}** holds **{page.total}** documents:", ""]
    lines += [f"- {_doc_line(d)}" for d in page.documents]
    return _answer(
        "\n".join(lines) + _more(page.total, len(page.documents)),
        "list_documents",
        args=args,
        ids=[d.document_id for d in page.documents],
        rows=page.total,
    )


def _missing(item: DocumentItem, field_name: str) -> bool:
    if field_name == "title":
        return not _has_title(item)
    if field_name == "year":
        return item.year is None
    return not (item.authors or "").strip()


def documents_missing(field_name: str) -> RecordAnswer:
    """The documents with no title, no year or no authors recorded (`list_documents`)."""
    found = [d for d in _all_documents() if _missing(d, field_name)]
    args = {"field": field_name}
    if not found:
        empty = records.list_documents(limit=1).total == 0
        return _answer(
            "The library holds no documents yet."
            if empty
            else f"Every document in the library has its {field_name} recorded.",
            "list_documents",
            args=args,
        )
    lines = [f"**{len(found)}** document(s) have no {field_name} recorded:", ""]
    lines += [f"- {_doc_line(d)}" for d in found[:SHOWN]]
    return _answer(
        "\n".join(lines) + _more(len(found), min(len(found), SHOWN)),
        "list_documents",
        args=args,
        ids=[d.document_id for d in found[:SHOWN]],
        rows=len(found),
    )


def concepts() -> RecordAnswer:
    """The concepts the person has taken on (`list_concepts`)."""
    listed = records.list_concepts()
    if not listed.concepts:
        return _answer("No concept has been taken on yet.", "list_concepts")
    shown = listed.concepts[:SHOWN]
    names = ", ".join(c.name for c in shown)
    return _answer(
        f"**{listed.total}** concept(s) taken on: {names}." + _more(listed.total, len(shown)),
        "list_concepts",
        ids=[c.concept_id for c in shown],
        rows=listed.total,
    )


def concept(name: str) -> RecordAnswer:
    """A concept's forms and chosen definition (`get_concept`); a term is not a concept."""
    name = name.strip().strip(_QUOTES)
    rec = records.get_concept(name)
    args = {"name": name}
    if rec is None:
        return _answer(
            f"“{name}” is not a concept you have taken on: no concept of that name has been "
            "taken on, and a term the library extracted is not a concept until you take it on.",
            "get_concept",
            args=args,
        )
    lines = [f"**{rec.name}**, a concept you have taken on.", ""]
    lines.append(f"- **Exact forms:** {', '.join(rec.exact_forms) or 'none recorded'}")
    lines.append(f"- **Broad forms:** {', '.join(rec.broad_forms) or 'none recorded'}")
    if rec.definition:
        src = rec.definition_source
        where = ""
        if src is not None and src.kind == "passage":
            where = f" (a passage of *{src.title or 'a document'}*)"
        elif src is not None:
            where = " (your own words)"
        lines.append(f"- **Definition{where}:** {rec.definition}")
    else:
        lines.append("- **Definition:** not recorded")
    return _answer("\n".join(lines), "get_concept", args=args, ids=[rec.concept_id], rows=1)


def library(text: str) -> RecordAnswer:
    """Today's count, health, latest and overview answers (`library_summary`)."""
    got = library_lookup(text)
    return _answer(got.answer, got.lookup, ids=got.ids, rows=got.rows)


# ============================================================
# The rules
# ============================================================


def _rx(*patterns: str) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_IN_FOLDER = r"in (?:the |my )?(?P<folder>.+?) folder"

_FOLDER_COUNT = _rx(
    rf"how many {_DOC_NOUN}(?: are| do i have)?(?: there)? {_IN_FOLDER}",
    rf"how many {_DOC_NOUN} (?:does|do) (?:the |my )?(?P<folder>.+?) folder (?:hold|have|contain)",
)
_FOLDER_LIST = _rx(
    rf"(?:which|what) {_DOC_NOUN}(?: are| do i have)? {_IN_FOLDER}",
    rf"(?:list|show|display)(?: me)?(?: all)?(?: the| my)?(?: {_DOC_NOUN})? {_IN_FOLDER}",
    rf"what(?:'s| is) {_IN_FOLDER}",
)
_FOLDERS = _rx(
    r"(?:which|what) folders (?:do i have|are there|does (?:my|the) library have)"
    r"(?: in (?:my|the) library)?",
    r"(?:list|show)(?: me)?(?: all)?(?: my| the)? folders",
    r"how many folders(?: do i have| are there)?(?: in (?:my|the) library)?",
)
_MISSING = _rx(
    rf"(?:which|what) {_DOC_NOUN}(?: in (?:my|the) library)? (?:have|has) (?:no|any missing) "
    rf"{_FIELD}(?: recorded| set| listed)?",
    rf"(?:which|what) {_DOC_NOUN}(?: in (?:my|the) library)? (?:are|is) (?:missing|without) "
    rf"(?:a |an |their |its )?{_FIELD}",
    rf"(?:list|show)(?: me)?(?: the)? {_DOC_NOUN} (?:with no|without|missing)(?: a| an)? {_FIELD}",
)
_CONCEPTS = _rx(
    r"(?:which|what) concepts (?:have i taken on|do i have|are there)(?: in (?:my|the) library)?",
    r"(?:list|show)(?: me)?(?: all)?(?: my| the)? concepts",
)
_CONCEPT = _rx(
    r"(?:what is|what's|show(?: me)?|give me) (?:my|the) (?:chosen |recorded )?definition of "
    r"the concept (?P<name>.+)",
    r"(?:what is|what's) (?:my|my chosen|the chosen|the recorded) definition of (?P<name>.+)",
    r"(?:what is|what's|show(?: me)?|describe) the concept (?P<name>.+)",
    r"what (?:forms|names|spellings) does the concept (?P<name>.+?) have",
)
_DOC_FIELD: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "year",
        _rx(
            r"(?:in )?what year (?:is|was) (?P<span>.+?)"
            r"(?: (?:from|published(?: in)?|written|released|out))?",
            r"when was (?P<span>.+?) (?:published|written|released)",
            r"what(?: is|'s) the (?:publication )?year of (?P<span>.+)",
        ),
    ),
    (
        "authors",
        _rx(
            r"who (?:are|is|were|was) the (?:authors?|writers?) of (?P<span>.+)",
            r"who (?:wrote|authored) (?P<span>.+)",
            r"what (?:are|is) the authors? of (?P<span>.+)",
        ),
    ),
    (
        "title",
        _rx(r"what(?: is|'s) the (?:recorded |full )?title of (?P<span>.+)"),
    ),
    (
        "type",
        _rx(
            rf"what (?:type|format|kind) of file is (?P<span>.+\b{_DOC_NOUN}\b.*)",
            rf"what (?:type|format) is (?P<span>.+\b{_DOC_NOUN}\b.*)",
            rf"what(?: is|'s) the (?:file )?(?:type|format) of (?P<span>.+\b{_DOC_NOUN}\b.*)",
        ),
    ),
]


def _field_of(match: re.Match[str]) -> str:
    raw = match.group("field").lower()
    return "authors" if raw.startswith("author") else raw.rstrip("s")


def _first(patterns: list[re.Pattern[str]], q: str) -> re.Match[str] | None:
    return next((m for p in patterns if (m := p.fullmatch(q)) is not None), None)


def answer_record_question(text: str) -> RecordAnswer | None:
    """The rule that catches ``text`` and its lookup's answer, or ``None``: retrieval answers.

    Folder rules come first, so a folder count is never the library's count (KI-95)."""
    q = _normal(text)
    if not q:
        return None
    if (m := _first(_FOLDER_COUNT, q)) is not None:
        return folder(m.group("folder"), count_only=True)
    if (m := _first(_FOLDER_LIST, q)) is not None:
        return folder(m.group("folder"))
    if _first(_FOLDERS, q) is not None:
        return folders()
    if (m := _first(_MISSING, q)) is not None:
        return documents_missing(_field_of(m))
    if _first(_CONCEPTS, q) is not None:
        return concepts()
    if (m := _first(_CONCEPT, q)) is not None:
        return concept(_strip_tails(m.group("name")))
    for field_name, patterns in _DOC_FIELD:
        if (m := _first(patterns, q)) is not None:
            span = _strip_tails(m.group("span"))
            if _TOPICAL_SPAN.search(span):
                return None
            return document_by_name(span, field_name)
    if is_library_query(text):
        return library(text)
    return None


# ============================================================
# The commands
# ============================================================


def _document_command(arg: str) -> RecordAnswer | None:
    """`/document <title>`: ``None`` for an id prefix, which keeps today's details view."""
    from doc_assistant.library import find_document_by_short_id

    if not arg:
        return None
    if re.fullmatch(r"[0-9a-f-]+", arg) and find_document_by_short_id(arg) is not None:
        return None
    found = document_by_name(arg)
    if found is not None:
        return found
    return _answer(
        f"No document is recorded under that name (“{arg}”). `/library` lists them all.",
        "list_documents",
        args={"name": arg, "field": None},
    )


_COMMANDS: dict[str, Callable[[str], RecordAnswer | None]] = {
    "folder": lambda arg: folder(arg) if arg else folders(),
    "concept": lambda arg: concept(arg) if arg else concepts(),
    "document": _document_command,
}


# ============================================================
# Scoped questions: `/folder <name>: <question>`, `/concept <name>: <question>` (A7b, G3)
# ============================================================

#: What separates the name from the question. The **first** one splits, so a question that
#: itself holds a further ``": "`` stays whole.
SCOPE_SEPARATOR = ": "


@dataclass(frozen=True)
class ScopedQuestion:
    """A question to run as an ordinary turn within one folder's or one concept's documents.

    ``kind`` is ``folder`` or ``concept``; ``target_id`` is that folder's or concept's id and
    ``name`` its recorded name; ``question`` is the text after the separator."""

    kind: str
    target_id: str
    name: str
    question: str


def split_scoped(cmd: str, arg: str) -> tuple[str, str] | None:
    """``(name, question)`` when ``/folder`` or ``/concept`` carries a question, else ``None``.

    The argument splits at its first :data:`SCOPE_SEPARATOR`. No separator, an empty name or an
    empty question → ``None``: A6's lookup runs on the argument unchanged."""
    if cmd not in ("folder", "concept") or SCOPE_SEPARATOR not in arg:
        return None
    name, question = arg.split(SCOPE_SEPARATOR, 1)
    name, question = name.strip(), question.strip()
    if not name or not question:
        return None
    return name, question


def _which_folder(name: str, matches: list[FolderItem], args: dict[str, Any]) -> RecordAnswer:
    lines = [f"**{len(matches)}** folders are named “{name}”. Which one do you mean?", ""]
    lines += [f"- {f.name} ({f.documents} documents)" for f in matches]
    lines += ["", "Choose it in the folder picker above the question box, then ask again."]
    return _answer(
        "\n".join(lines),
        "list_folders",
        args=args,
        ids=[f.folder_id for f in matches],
        rows=len(matches),
        asked=True,
    )


def _unknown_concept(name: str) -> RecordAnswer:
    """A6's unknown-concept sentence, followed by the concepts there are to choose from."""
    listed = records.list_concepts()
    shown = listed.concepts[:SHOWN]
    if shown:
        tail = f"The concepts you have taken on are: {', '.join(c.name for c in shown)}." + _more(
            listed.total, len(shown)
        )
    else:
        tail = "No concept has been taken on yet."
    return _answer(
        f"“{name}” is not a concept you have taken on, so no question was asked. {tail}",
        "list_concepts",
        args={"name": name},
        ids=[c.concept_id for c in shown],
        rows=listed.total,
    )


def scoped_question(cmd: str, arg: str) -> ScopedQuestion | RecordAnswer | None:
    """Read a scoped question from a command (G3).

    ``None``: not one (no separator, nothing after it, or another command), so A6's lookup runs.
    A :class:`RecordAnswer`: the name matched no folder or concept, or several folders, and the
    answer lists or asks; no question is asked and nothing is retrieved. A
    :class:`ScopedQuestion`: exactly one match, to run as an ordinary turn on its documents.

    Names resolve through A6's matchers: a folder by its name, case aside; a concept through
    `read_surface.records.get_concept` (its id, name or a form), which returns one concept or
    none, so a concept never asks which."""
    split = split_scoped(cmd, arg.strip())
    if split is None:
        return None
    name, question = split
    if cmd == "folder":
        listed = records.list_folders()
        matches = _folder_matches(name, listed)
        args: dict[str, Any] = {"folder": name, "question": question}
        if not matches:
            return _unknown_folder(name, listed, args)
        if len(matches) > 1:
            return _which_folder(name, matches, args)
        return ScopedQuestion("folder", matches[0].folder_id, matches[0].name, question)
    rec = records.get_concept(name.strip(_QUOTES))
    if rec is None:
        return _unknown_concept(name)
    return ScopedQuestion("concept", rec.concept_id, rec.name, question)


def answer_command(cmd: str, arg: str) -> RecordAnswer | None:
    """A lookup command's answer, or ``None`` when ``cmd`` is not one (or `/document` was given
    an id prefix): the caller then runs the command as before."""
    run = _COMMANDS.get(cmd)
    if run is None:
        return None
    arg = arg.strip()
    if cmd in ("folder", "concept") and arg.endswith(SCOPE_SEPARATOR.strip()):
        # A7b (G3): `/folder Thesis:` with no question after the colon is A6's lookup on the name.
        arg = arg[:-1].strip()
    return run(arg)
