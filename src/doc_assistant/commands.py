"""Slash-command parsing and response formatting.

Handles /library, /document, /help, /cites, /cited-by, /graph commands.
Returns markdown strings — the UI layer decides how to display them.

No UI framework imports in this module.
"""

from doc_assistant.library import (
    CitationEdge,
    DocumentDetails,
    DocumentSummary,
    LibrarySummary,
    cited_by,
    cites_out,
    find_document_by_short_id,
    get_document_details,
    graph_subgraph,
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
- `/cites <id>` — citations this document makes (Phase 4)
- `/cited-by <id>` — documents in the library that cite this one (Phase 4)
- `/graph <id>` — small citation subgraph around this document (Phase 4)
- `/help` — this message

Anything else is treated as a normal question to the library.
"""


# ============================================================
# Phase 4 citation formatters
# ============================================================


def _ref_one_line(e: CitationEdge) -> str:
    """One-line markdown for an outgoing citation."""
    parts: list[str] = []
    if e.target_authors:
        a = e.target_authors[:50] + ("…" if len(e.target_authors) > 50 else "")
        parts.append(a)
    if e.target_year is not None:
        parts.append(f"({e.target_year})")
    if e.target_title:
        t = e.target_title[:80] + ("…" if len(e.target_title) > 80 else "")
        parts.append(f"*{t}*")
    if e.target_doi:
        parts.append(f"[doi:{e.target_doi}]")
    line = " ".join(parts) if parts else (e.raw_text or "(unparsed)")
    if e.target_document_id:
        line = f"🔗 **`{e.target_document_id[:8]}`** {line}"
    else:
        line = f"  {line}"
    return f"- {line}"


def format_cites_out(filename: str, edges: list[CitationEdge]) -> str:
    """Build markdown for `/cites <id>`."""
    if not edges:
        return (
            f"**{filename}** — no citations extracted yet. "
            "Run `python -m scripts.extract_citations --apply`."
        )

    internal = [e for e in edges if e.target_document_id]
    external = [e for e in edges if not e.target_document_id]

    lines = [f"## {filename} cites {len(edges)} works"]
    lines.append(
        f"_{len(internal)} resolved to library docs · {len(external)} external_"
    )
    lines.append("")

    if internal:
        lines.append(f"### 🔗 In library ({len(internal)})")
        lines.append("")
        for e in internal:
            lines.append(_ref_one_line(e))
        lines.append("")

    cap = 30
    if external:
        lines.append(f"### 📄 External ({len(external)})")
        lines.append("")
        for e in external[:cap]:
            lines.append(_ref_one_line(e))
        if len(external) > cap:
            lines.append(f"_…and {len(external) - cap} more._")

    return "\n".join(lines)


def format_cited_by(filename: str, rows: list[tuple[str, str, str | None]]) -> str:
    """Build markdown for `/cited-by <id>`."""
    if not rows:
        return (
            f"**{filename}** — no library documents cite this one yet. "
            "Either no internal citations have been resolved, or extraction "
            "hasn't been run. Try `python -m scripts.extract_citations --apply`."
        )
    lines = [f"## {len(rows)} library document(s) cite {filename}"]
    lines.append("")
    for src_id, src_fn, raw in rows:
        snippet = (raw or "")[:120] + ("…" if raw and len(raw) > 120 else "")
        lines.append(f"- `{src_id[:8]}` **{src_fn}** — {snippet}")
    return "\n".join(lines)


def format_graph(filename: str, graph: dict[str, object]) -> str:
    """Build markdown for `/graph <id>` — Mermaid subgraph for small N."""
    nodes_obj = graph.get("nodes", [])
    edges_obj = graph.get("edges", [])
    assert isinstance(nodes_obj, list) and isinstance(edges_obj, list)

    if not nodes_obj or len(nodes_obj) == 1:
        return (
            f"**{filename}** has no internal citation edges yet. "
            "Once `scripts/extract_citations.py --apply` is run and resolves "
            "library-internal citations, this graph will populate."
        )

    if len(nodes_obj) > 25:
        return (
            f"## {filename} — citation subgraph\n\n"
            f"_{len(nodes_obj)} nodes, {len(edges_obj)} edges._ "
            "Too large for inline rendering. Use the data API "
            "(`library.graph_subgraph`) for visualization."
        )

    lines = [f"## {filename} — citation subgraph", "", "```mermaid", "graph LR"]
    for n in nodes_obj:
        assert isinstance(n, dict)
        nid = str(n["id"])[:8]
        label = str(n.get("filename") or "")[:30].replace('"', "'")
        if n.get("is_center"):
            lines.append(f'  {nid}["**{label}**"]:::center')
        else:
            lines.append(f'  {nid}["{label}"]')
    for e in edges_obj:
        assert isinstance(e, dict)
        src = str(e["source"])[:8]
        tgt = str(e["target"])[:8]
        lines.append(f"  {src} --> {tgt}")
    lines.append("  classDef center fill:#fef3c7,stroke:#92400e,stroke-width:2px;")
    lines.append("```")
    return "\n".join(lines)


# ============================================================
# Command dispatcher
# ============================================================

_FORMAT_FILTERS = frozenset(["pdf", "epub", "html", "htm", "docx", "txt", "md", "odt", "rtf"])


def parse_command(message: str) -> tuple[str, str] | None:
    """Parse a slash command. Returns (command, arg) or None."""
    msg = message.strip()
    if not msg.startswith("/"):
        return None
    parts = msg[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip().lower() if len(parts) > 1 else ""
    return cmd, arg


def execute_command(cmd: str, arg: str) -> str:
    """Execute a parsed command, returning a markdown response."""
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

    if cmd in ("cites", "cited-by", "cited_by", "graph"):
        if not arg:
            return f"Usage: `/{cmd} <id>` — provide the document ID prefix."
        full_id = find_document_by_short_id(arg) if len(arg) < 36 else arg
        if not full_id:
            return f"No document found matching `{arg}`. Try `/library` to see IDs."
        details = get_document_details(full_id)
        filename = details.filename if details else arg
        if cmd == "cites":
            return format_cites_out(filename, cites_out(full_id))
        if cmd in ("cited-by", "cited_by"):
            return format_cited_by(filename, cited_by(full_id))
        if cmd == "graph":
            return format_graph(filename, graph_subgraph(full_id, depth=1))

    return f"Unknown command: `/{cmd}`. Try `/help`."
