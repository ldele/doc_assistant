"""Route user messages to the right handler.

Detects whether a message is asking about library metadata (document
counts, health, latest additions) versus asking about document *content*
(which should go to the RAG pipeline).

This module contains zero UI dependencies — it returns plain strings
that the UI layer can display however it wants.
"""

import re
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


def answer_library_query(text: str) -> str:
    """Answer a library metadata question from SQLite.

    Returns a markdown string. The UI layer sends it as-is.
    """
    lower = text.lower()
    summary = library_summary()

    # Latest/newest document
    if re.search(r"(latest|newest|recent|last)", lower):
        docs = list_documents()
        if not docs:
            return "Your library is empty. Add documents to `data/sources/` and run ingestion."
        with_dates = [d for d in docs if d.added_at is not None]
        if with_dates:
            latest = max(
                with_dates,
                key=lambda d: d.added_at or datetime.min,
            )
            date_str = latest.added_at.strftime("%Y-%m-%d %H:%M") if latest.added_at else "unknown"
            return (
                f"**Most recently added:** `{latest.filename}` "
                f"({latest.format.upper()}, "
                f"{latest.chunk_count or 0} chunks, "
                f"{health_badge(latest.health)})\n\n"
                f"Added: {date_str}\n\n"
                f"Use `/document {latest.id[:8]}` for full details."
            )
        return "Documents exist but have no recorded add dates. Try `/library` to browse."

    # Broken/marginal documents
    if re.search(r"(broken|marginal|unhealthy)", lower):
        health = "broken" if "broken" in lower else "marginal"
        docs = list_documents(health=health)
        if not docs:
            return f"No {health} documents in your library."
        names = ", ".join(f"`{d.filename}`" for d in docs[:10])
        more = f" (and {len(docs) - 10} more)" if len(docs) > 10 else ""
        return (
            f"**{len(docs)} {health} document(s):** {names}{more}\n\n"
            f"Use `/library {health}` for details."
        )

    # Count / how many
    if re.search(r"(how many|count|total|number)", lower):
        health_str = " · ".join(f"{h}: {n}" for h, n in sorted(summary.by_health.items()))
        return (
            f"**{summary.total_documents} documents**, "
            f"{summary.total_chunks:,} chunks.\n\n"
            f"Health: {health_str}\n\n"
            f"Use `/library` for the full list."
        )

    # Generic library overview
    return (
        f"**Library:** {summary.total_documents} documents, "
        f"{summary.total_chunks:,} chunks.\n\n"
        f"Use `/library` to browse, `/document <id>` for details, "
        f"or `/help` for all commands."
    )
