"""BibTeX export for the document library (PR 1.5).

Projects ``Document`` rows into a single BibTeX file. Entry type is
chosen heuristically from the existing metadata — no schema change:

* Has ``authors`` AND ``year`` AND format in {pdf, epub, html} → ``@article``
* Format in {md, txt} → ``@misc`` flagged as a note (filename surfaced)
* Otherwise → ``@misc``

Citation keys are ``<surname>_<year>`` for papers and
``note_<safe_filename>`` / ``misc_<short_id>`` otherwise. Collisions are
resolved by appending ``a``, ``b``, ``c``, … in document-id order.

Pure functions over the SQLAlchemy ``Document`` model — no I/O.
The CLI runner and slash command both call into ``export_bibtex``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest.citations import _first_author_surname

# ============================================================
# Classification
# ============================================================

_PAPER_FORMATS = frozenset({"pdf", "epub", "html", "htm"})
_NOTE_FORMATS = frozenset({"md", "txt"})


@dataclass
class BibEntry:
    """One BibTeX entry ready for rendering."""

    entry_type: str  # "article" | "misc"
    key: str
    fields: dict[str, str]


# ============================================================
# Escaping
# ============================================================

_BRACE_RE = re.compile(r"([{}])")


def escape_bibtex(value: str) -> str:
    """Escape a string for safe inclusion inside ``{...}`` in a BibTeX field.

    Replaces unbalanced braces with their escaped form. BibTeX itself
    handles ``& % $ # _`` inside braced values, so we only need to
    sanitise braces. Newlines collapse to single spaces — multi-line
    fields confuse downstream BibTeX consumers.
    """
    if not value:
        return ""
    cleaned = " ".join(value.split())
    return _BRACE_RE.sub(r"\\\1", cleaned)


# ============================================================
# Key generation
# ============================================================

_SAFE_KEY = re.compile(r"[^A-Za-z0-9]+")


def _safe_key_fragment(s: str) -> str:
    return _SAFE_KEY.sub("_", s).strip("_").lower()


def _citation_key(doc: Document) -> str:
    """Return an un-de-duplicated citation key for ``doc``."""
    if doc.format in _NOTE_FORMATS:
        stem = doc.filename.rsplit(".", 1)[0]
        return f"note_{_safe_key_fragment(stem)[:40]}"

    surname = _first_author_surname(doc.authors) if doc.authors else None
    if surname and doc.year is not None:
        return f"{_safe_key_fragment(surname)}_{doc.year}"
    if surname:
        return _safe_key_fragment(surname)
    return f"misc_{doc.id[:8]}"


def _dedupe_keys(entries: list[BibEntry]) -> list[BibEntry]:
    """Append ``a``, ``b``, … to keys that collide. Order is stable."""
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.key] = counts.get(e.key, 0) + 1
    seen: dict[str, int] = {}
    out: list[BibEntry] = []
    for e in entries:
        if counts[e.key] == 1:
            out.append(e)
            continue
        idx = seen.get(e.key, 0)
        suffix = chr(ord("a") + idx) if idx < 26 else f"_{idx}"
        seen[e.key] = idx + 1
        out.append(BibEntry(entry_type=e.entry_type, key=f"{e.key}{suffix}", fields=e.fields))
    return out


# ============================================================
# Entry building
# ============================================================


def _build_entry(doc: Document) -> BibEntry:
    fmt = (doc.format or "").lower()
    fields: dict[str, str] = {}

    if fmt in _NOTE_FORMATS:
        title = doc.title or doc.filename.rsplit(".", 1)[0]
        fields["title"] = escape_bibtex(title)
        fields["howpublished"] = "Personal note"
        fields["note"] = escape_bibtex(f"filename: {doc.filename}; doc_id: {doc.id[:8]}")
        return BibEntry(entry_type="misc", key=_citation_key(doc), fields=fields)

    is_paper = fmt in _PAPER_FORMATS and doc.authors and doc.year is not None
    entry_type = "article" if is_paper else "misc"

    if doc.title:
        fields["title"] = escape_bibtex(doc.title)
    if doc.authors:
        fields["author"] = escape_bibtex(doc.authors)
    if doc.year is not None:
        fields["year"] = str(doc.year)
    if doc.doi:
        fields["doi"] = escape_bibtex(doc.doi)
    fields["note"] = escape_bibtex(f"filename: {doc.filename}; doc_id: {doc.id[:8]}")
    if not is_paper and "title" not in fields:
        fields["title"] = escape_bibtex(doc.filename)
    return BibEntry(entry_type=entry_type, key=_citation_key(doc), fields=fields)


def _render_entry(entry: BibEntry) -> str:
    lines = [f"@{entry.entry_type}{{{entry.key},"]
    for k, v in entry.fields.items():
        lines.append(f"  {k} = {{{v}}},")
    lines.append("}")
    return "\n".join(lines)


# ============================================================
# Top-level export
# ============================================================


def build_entries(docs: list[Document]) -> list[BibEntry]:
    """Pure transform: ``Document`` rows -> citation-key-deduplicated BibEntries."""
    entries = [_build_entry(d) for d in docs]
    return _dedupe_keys(entries)


def render(entries: list[BibEntry]) -> str:
    """Render entries as a single BibTeX file string (sorted by key)."""
    sorted_entries = sorted(entries, key=lambda e: e.key)
    blocks = [_render_entry(e) for e in sorted_entries]
    header = f"% Generated by doc_assistant\n% {len(entries)} entries\n\n"
    return header + "\n\n".join(blocks) + "\n"


def export_bibtex() -> str:
    """Read the library and return its BibTeX representation."""
    with session_scope() as session:
        docs = list(
            session.execute(
                select(Document).where(Document.is_archived.is_(False)).order_by(Document.filename)
            )
            .scalars()
            .all()
        )
        return render(build_entries(docs))
