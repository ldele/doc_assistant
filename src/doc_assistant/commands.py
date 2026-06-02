"""Slash-command parsing and response formatting.

Handles /library, /document, /help, /cites, /cited-by, /graph commands.
Returns markdown strings — the UI layer decides how to display them.

No UI framework imports in this module.
"""

from doc_assistant.library import (
    CitationEdge,
    CitationGraph,
    DocumentDetails,
    DocumentSummary,
    LibrarySummary,
    SimilarDoc,
    cited_by,
    cites_out,
    find_document_by_short_id,
    get_document_details,
    graph_subgraph,
    library_summary,
    list_documents,
    similar_docs,
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
- `/similar <id>` — top-N semantically-similar documents (Phase 4)
- `/bibtex` — render the whole library as BibTeX (writes to `docs/library.bib` from the CLI)
- `/export-record <id>` — export the full provenance record for one answer as JSON (Phase 5)
- `/records` — list the most recent answer records
- `/review <id>` — run the LLM reviewer on any past answer (Phase 6)
- `/help` — this message

Anything else is treated as a normal question to the library.

_One command per message — chaining (e.g. `/cites X then /similar X`)
is not supported; the parser treats everything after the command as a
single argument._
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
    line = f"🔗 **`{e.target_document_id[:8]}`** {line}" if e.target_document_id else f"  {line}"
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
    lines.append(f"_{len(internal)} resolved to library docs · {len(external)} external_")
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


def format_similar(filename: str, neighbours: list[SimilarDoc]) -> str:
    """Build markdown for `/similar <id>`."""
    if not neighbours:
        return (
            f"**{filename}** — no similarity edges yet. "
            "Run `python -m scripts.compute_doc_vectors --apply`."
        )
    lines = [f"## {len(neighbours)} document(s) most similar to {filename}"]
    lines.append("")
    for n in neighbours:
        title = f" — *{n.target_title}*" if n.target_title else ""
        lines.append(
            f"- `{n.target_document_id[:8]}` **{n.target_filename}**{title}  "
            f"(cosine {n.score:.3f})"
        )
    return "\n".join(lines)


def format_graph(filename: str, graph: CitationGraph) -> str:
    """Build markdown for `/graph <id>` — Mermaid subgraph for small N."""
    if not graph.nodes or len(graph.nodes) == 1:
        return (
            f"**{filename}** has no internal citation edges yet. "
            "Once `scripts/extract_citations.py --apply` is run and resolves "
            "library-internal citations, this graph will populate."
        )

    if len(graph.nodes) > 25:
        return (
            f"## {filename} — citation subgraph\n\n"
            f"_{len(graph.nodes)} nodes, {len(graph.edges)} edges._ "
            "Too large for inline rendering. Use the data API "
            "(`library.graph_subgraph`) for visualization."
        )

    lines = [f"## {filename} — citation subgraph", "", "```mermaid", "graph LR"]
    for n in graph.nodes:
        nid = n.id[:8]
        label = (n.filename or "")[:30].replace('"', "'")
        if n.is_center:
            lines.append(f'  {nid}["**{label}**"]:::center')
        else:
            lines.append(f'  {nid}["{label}"]')
    for e in graph.edges:
        lines.append(f"  {e.source[:8]} --> {e.target[:8]}")
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

    if cmd == "bibtex":
        from doc_assistant.bibtex import export_bibtex

        text = export_bibtex()
        entry_count = text.count("\n@")
        return (
            f"### BibTeX export — {entry_count} entries\n\n"
            "```bibtex\n"
            f"{text}"
            "```\n"
            "_Run `python -m scripts.export_bibtex` to write this to `docs/library.bib`._"
        )

    if cmd in ("export-record", "export_record"):
        import json as _json

        from doc_assistant.provenance import find_record_by_short_id, get_record

        if not arg:
            return (
                "Usage: `/export-record <id>` — id is the 8-char prefix "
                "from any answer's provenance card."
            )
        prov = get_record(arg) if len(arg) >= 36 else find_record_by_short_id(arg)
        if prov is None:
            return f"No answer record matching `{arg}`. Try `/records` to list recent answers."
        return (
            f"### Answer record `{prov.id[:8]}`\n\n"
            "```json\n"
            f"{_json.dumps(prov.to_json_dict(), indent=2, ensure_ascii=False)}\n"
            "```"
        )

    if cmd == "records":
        from doc_assistant.provenance import list_recent_records

        records = list_recent_records(limit=20)
        if not records:
            return "No answer records yet. Ask a content question to generate one."
        lines = [f"## {len(records)} recent answer record(s)\n"]
        for r in records:
            ts = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "?"
            q = (r.query or "")[:80] + ("…" if r.query and len(r.query) > 80 else "")
            lines.append(f"- `{r.id[:8]}` · {ts} · {q}")
        return "\n".join(lines)

    if cmd == "review":
        from doc_assistant.config import REVIEWER_MODEL, REVIEWER_PROVIDER
        from doc_assistant.llm import get_reviewer_client, reviewer_available
        from doc_assistant.provenance import find_record_by_short_id, get_record
        from doc_assistant.reviewer import persist_review, review_answer

        if not arg:
            return "Usage: `/review <id>` — id is the 8-char prefix from a provenance card."
        if not reviewer_available():
            return (
                f"`/review` needs `ANTHROPIC_API_KEY` set — the reviewer "
                f"(`{REVIEWER_PROVIDER}`) is an LLM call. Set `REVIEWER_PROVIDER=ollama` "
                f"to review locally without a key."
            )
        prov = get_record(arg) if len(arg) >= 36 else find_record_by_short_id(arg)
        if prov is None:
            return f"No answer record matching `{arg}`. Try `/records` to list recent answers."
        try:
            result = review_answer(prov, get_reviewer_client())
            persist_review(prov.id, result, reviewer_kind="llm_haiku", model_name=REVIEWER_MODEL)
        except Exception as e:
            return f"Reviewer failed: {type(e).__name__}: {e}"
        if result.error:
            return f"Reviewer error: {result.error}"
        notes = f"\n\n> {result.notes}" if result.notes else ""
        return (
            f"## Reviewer assessment for `{prov.id[:8]}`\n\n"
            f"- **faithfulness:** {result.faithfulness}/5\n"
            f"- **citation density:** {result.citation_density}/5\n"
            f"- **hedging adequacy:** {result.hedging_adequacy}/5\n"
            f"- **unsupported claims:** {result.unsupported_claims_count}"
            f"{notes}"
        )

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

    if cmd in ("cites", "cited-by", "cited_by", "graph", "similar"):
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
        if cmd == "similar":
            return format_similar(filename, similar_docs(full_id))

    return f"Unknown command: `/{cmd}`. Try `/help`."
