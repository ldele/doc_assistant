"""Route user messages to the right handler.

Detects whether a message is asking about library metadata (document
counts, health, latest additions) versus asking about document *content*
(which should go to the RAG pipeline).

This module contains zero UI dependencies — it returns plain strings
that the UI layer can display however it wants.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from doc_assistant.library import library_summary, list_documents

# ============================================================
# Detection patterns
# ============================================================

# A trailing topical qualifier turns a metadata query into a *content* query:
# "show my papers" (list the library) vs "show my papers about RAG" (answer from
# content). When this negative lookahead matches, the pattern declines so the
# message falls through to the RAG pipeline and gets a real answer by default.
_TOPIC_WORDS = (
    "about|on|regarding|concerning|related to|discussing|"
    "covering|mentioning|that|which|with|where|involving"
)
_NOT_TOPICAL = rf"(?!\s+(?:{_TOPIC_WORDS})\b)"

_LIBRARY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(latest|newest|recent|last)\s+(document|file|paper|pdf)",
        r"how many\s+(document|file|paper|pdf|chunk)",
        r"(list|show|display)\s+(all\s+)?(my\s+)?(?:documents?|files?|papers?|pdfs?)\b"
        + _NOT_TOPICAL,
        r"what('s| is) in (?:my |the )?(?:library|database|collection)\b" + _NOT_TOPICAL,
        r"(library|database|collection)\s+(stats|statistics|summary|overview|status)",
        r"(broken|marginal|unhealthy)\s+(document|file|paper|pdf)",
        r"(document|file|paper)\s+(count|total|number)",
    ]
]


def is_library_query(text: str) -> bool:
    """Return True if the message is about library metadata, not content."""
    return any(p.search(text) for p in _LIBRARY_PATTERNS)


# ============================================================
# Health badge (shared by router + formatters)
# ============================================================


def health_badge(health: str | None) -> str:
    """Visual badge for document health."""
    return {
        "healthy": "🟢 healthy",
        "marginal": "🟡 marginal",
        "broken": "🔴 broken",
    }.get(health or "unknown", "⚪ unknown")


# ============================================================
# Library metadata responses
# ============================================================


@dataclass(frozen=True)
class LibraryLookup:
    """Today's library answer, with what it read: the lookup's name, the document ids it named
    and how many rows it returned (A6's `AnswerRecord.records_json`)."""

    answer: str
    lookup: str
    ids: list[str] = field(default_factory=list)
    rows: int = 0


def answer_library_query(text: str) -> str:
    """Answer a library metadata question from SQLite.

    Returns a markdown string. The UI layer sends it as-is.
    """
    return library_lookup(text).answer


def library_lookup(text: str) -> LibraryLookup:
    """Answer a library metadata question from SQLite, saying what was read."""
    lower = text.lower()
    summary = library_summary()

    # Latest/newest document
    if re.search(r"(latest|newest|recent|last)", lower):
        docs = list_documents()
        if not docs:
            return LibraryLookup(
                "Your library is empty. Add documents to `data/sources/` and run ingestion.",
                "list_documents",
            )
        with_dates = [d for d in docs if d.added_at is not None]
        if with_dates:
            latest = max(
                with_dates,
                key=lambda d: d.added_at or datetime.min,
            )
            date_str = latest.added_at.strftime("%Y-%m-%d %H:%M") if latest.added_at else "unknown"
            return LibraryLookup(
                f"**Most recently added:** `{latest.filename}` "
                f"({latest.format.upper()}, "
                f"{latest.chunk_count or 0} chunks, "
                f"{health_badge(latest.health)})\n\n"
                f"Added: {date_str}\n\n"
                f"Use `/document {latest.id[:8]}` for full details.",
                "list_documents",
                [latest.id],
                1,
            )
        return LibraryLookup(
            "Documents exist but have no recorded add dates. Try `/library` to browse.",
            "list_documents",
        )

    # Broken/marginal documents
    if re.search(r"(broken|marginal|unhealthy)", lower):
        health = "broken" if "broken" in lower else "marginal"
        docs = list_documents(health=health)
        if not docs:
            return LibraryLookup(f"No {health} documents in your library.", "list_documents")
        names = ", ".join(f"`{d.filename}`" for d in docs[:10])
        more = f" (and {len(docs) - 10} more)" if len(docs) > 10 else ""
        return LibraryLookup(
            f"**{len(docs)} {health} document(s):** {names}{more}\n\n"
            f"Use `/library {health}` for details.",
            "list_documents",
            [d.id for d in docs[:10]],
            len(docs),
        )

    # Count / how many
    if re.search(r"(how many|count|total|number)", lower):
        if summary.total_documents == 0:
            return LibraryLookup(
                "**0 documents**, 0 chunks: the library holds no documents yet.",
                "library_summary",
            )
        health_str = " · ".join(f"{h}: {n}" for h, n in sorted(summary.by_health.items()))
        return LibraryLookup(
            f"**{summary.total_documents} documents**, "
            f"{summary.total_chunks:,} chunks.\n\n"
            f"Health: {health_str}\n\n"
            f"Use `/library` for the full list.",
            "library_summary",
            rows=1,
        )

    # Generic library overview
    return LibraryLookup(
        f"**Library:** {summary.total_documents} documents, "
        f"{summary.total_chunks:,} chunks.\n\n"
        f"Use `/library` to browse, `/document <id>` for details, "
        f"or `/help` for all commands.",
        "library_summary",
        rows=1,
    )
