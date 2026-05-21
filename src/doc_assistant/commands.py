"""Slash-command parsing and response formatting.

Handles /library, /document, /help commands. Returns markdown strings —
the UI layer decides how to display them.

No UI framework imports in this module.
"""

from doc_assistant.library import (
    DocumentDetails,
    DocumentSummary,
    LibrarySummary,
    find_document_by_short_id,
    get_document_details,
    library_summary,
    list_documents,
)
from doc_assistant.query_router import health_badge

# ============================================================
# Formatters
# ============================================================


def format_summary_message(
    summary: LibrarySummary,
    docs: list[DocumentSummary],
    filter_desc: str | None,
) -> str:
    """Build markdown for the library overview."""
    lines = ["## Library overview\n"]
    if filter_desc:
        lines.append(f"_Filtered: {filter_desc} ({len(docs)} of {summary.total_documents})_\n")

    health_strs = " · ".join(
        f"{health_badge(h)} {n}" for h, n in sorted(summary.by_health.items())
    )
    format_strs = ", ".join(f"{fmt.upper()} ({n})" for fmt, n in sorted(summary.by_format.items()))
    lines.append(f"**{summary.total_documents} documents**, {summary.total_chunks:,} chunks")
    lines.append(f"Health: {health_strs}")
    lines.append(f"Formats: {format_strs}")
    lines.append("")

    # Group docs by health for readability
    groups: dict[str, list[DocumentSummary]] = {
        "broken": [],
        "marginal": [],
        "healthy": [],
        "unknown": [],
    }
    for d in docs:
        groups[d.health or "unknown"].append(d)

    for health in ("broken", "marginal", "healthy", "unknown"):
        if not groups[health]:
            continue
        lines.append(f"### {health_badge(health)} ({len(groups[health])})")
        lines.append("")
        for d in groups[health]:
            short_id = d.id[:8]
            tags_str = f" · tags: {', '.join(d.tags)}" if d.tags else ""
            folders_str = f" · folders: {', '.join(d.folders)}" if d.folders else ""
            lines.append(
                f"- `{short_id}` **{d.filename}** "
                f"({d.format.upper()}, {d.chunk_count or 0} chunks)"
                f"{folders_str}{tags_str}"
            )
        lines.append("")

    return "\n".join(lines)


def format_document_details(details: DocumentDetails | None) -> str:
    """Build markdown for a single document detail view."""
    if details is None:
        return "Document not found."

    d = details
    lines = [f"## {d.filename}"]
    lines.append("")
    lines.append(f"**Health:** {health_badge(d.extraction_health)}")
    lines.append(f"**Format:** {d.format.upper()}")
    lines.append(f"**Chunks:** {d.chunk_count or 0}")
    if d.page_count:
        lines.append(f"**Pages:** {d.page_count}")
    if d.extractor_used:
        lines.append(f"**Extracted with:** {d.extractor_used}")
    lines.append("")

    if d.title:
        lines.append(f"**Title:** {d.title}")
    if d.authors:
        lines.append(f"**Authors:** {d.authors}")
    if d.year:
        lines.append(f"**Year:** {d.year}")
    if d.doi:
        lines.append(f"**DOI:** {d.doi}")
    lines.append("")

    if d.tags:
        lines.append(f"**Tags:** {', '.join(d.tags)}")
    if d.folders:
        lines.append(f"**Folders:** {', '.join(d.folders)}")
    if d.keywords:
        lines.append(
            f"**Keywords:** {', '.join(d.keywords[:10])}"
            + (f" (and {len(d.keywords) - 10} more)" if len(d.keywords) > 10 else "")
        )
    lines.append("")

    lines.append(f"**Document ID:** `{d.id}`")
    lines.append(f"**Source:** `{d.source_original}`")
    lines.append(f"**Hash:** `{d.doc_hash}`")
    lines.append("")

    if d.notes:
        lines.append("### Notes")
        lines.append(d.notes)
        lines.append("")

    if d.ingestion_history:
        lines.append("### Ingestion history")
        for event in d.ingestion_history[:10]:
            ts = event["timestamp"].strftime("%Y-%m-%d %H:%M") if event["timestamp"] else "?"
            lines.append(
                f"- {ts}: **{event['event_type']}** "
                f"(extractor: {event.get('extractor') or '-'}, "
                f"chunks: {event.get('chunks_produced') or '-'}, "
                f"health: {event.get('health_status') or '-'})"
            )
            if event.get("notes"):
                lines.append(f"  - notes: {event['notes']}")

    return "\n".join(lines)


def help_message() -> str:
    """Return the /help response."""
    return """## Available commands

- `/library` — show all documents
- `/library broken` — show only broken documents
- `/library marginal` — show only marginal documents
- `/library healthy` — show only healthy documents
- `/library pdf` (or epub, etc.) — show documents of a specific format
- `/document <id>` — full details for one document (use first 8 chars of ID)
- `/help` — this message

Anything else is treated as a normal question to the library.
"""


# ============================================================
# Command dispatcher
# ============================================================

# Valid format filters for the /library command
_FORMAT_FILTERS = frozenset(["pdf", "epub", "html", "htm", "docx", "txt", "md", "odt", "rtf"])


def parse_command(message: str) -> tuple[str, str] | None:
    """Parse a slash command from the message.

    Returns (command, arg) if the message is a command, else None.
    """
    msg = message.strip()
    if not msg.startswith("/"):
        return None
    parts = msg[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip().lower() if len(parts) > 1 else ""
    return cmd, arg


def execute_command(cmd: str, arg: str) -> str:
    """Execute a parsed command, returning the markdown response."""
    if cmd == "help":
        return help_message()

    if cmd == "library":
        health = format_filter = None
        if arg in ("broken", "marginal", "healthy"):
            health = arg
        elif arg in _FORMAT_FILTERS:
            format_filter = arg
        elif arg:
            return f"Unknown library filter: `{arg}`. Try `/help`."

        docs = list_documents(health=health, format=format_filter)
        summary = library_summary()

        filter_desc = None
        if health:
            filter_desc = f"health: {health}"
        elif format_filter:
            filter_desc = f"format: {format_filter}"

        return format_summary_message(summary, docs, filter_desc)

    if cmd == "document":
        if not arg:
            return "Usage: `/document <id>` — provide the document ID prefix."
        full_id = find_document_by_short_id(arg) if len(arg) < 36 else arg
        if not full_id:
            return f"No document found matching `{arg}`. Try `/library` to see IDs."
        details = get_document_details(full_id)
        if not details:
            return f"Could not load details for `{arg}`."
        return format_document_details(details)

    return f"Unknown command: `/{cmd}`. Try `/help`."
