import chainlit as cl
from doc_assistant.pipeline import RAGPipeline, format_citation
from doc_assistant.tracking import TokenCounter

from doc_assistant.config import TOP_K

from doc_assistant.library import (
    list_documents,
    get_document_details,
    library_summary,
    find_document_by_short_id,
)

rag = RAGPipeline()

# ============================================================
# Command rendering
# ============================================================

def _health_badge(health: str | None) -> str:
    """Visual badge for document health."""
    return {
        "healthy": "🟢 healthy",
        "marginal": "🟡 marginal",
        "broken": "🔴 broken",
    }.get(health or "unknown", "⚪ unknown")


def _format_summary_message(summary, docs, filter_desc:str | None) -> str:
    """Build the markdown for the library overview."""
    lines = ["## Library overview\n"]
    if filter_desc:
        lines.append(f"_Filtered: {filter_desc} ({len(docs)} of {summary.total_documents})_\n")

    # Top-level counts
    health_strs = " · ".join(
        f"{_health_badge(h)} {n}"
        for h, n in sorted(summary.by_health.items())
    )
    format_strs = ", ".join(f"{fmt.upper()} ({n})" for fmt, n in sorted(summary.by_format.items()))
    lines.append(f"**{summary.total_documents} documents**, {summary.total_chunks:,} chunks")
    lines.append(f"Health: {health_strs}")
    lines.append(f"Formats: {format_strs}")
    lines.append("")

    # Group docs by health for readability
    groups = {"broken": [], "marginal": [], "healthy": [], "unknown": []}
    for d in docs:
        groups[d.health or "unknown"].append(d)

    for health in ("broken", "marginal", "healthy", "unknown"):
        if not groups[health]:
            continue
        lines.append(f"### {_health_badge(health)} ({len(groups[health])})")
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


def _format_document_details(details) -> str:
    """Build the markdown for a single document detail view."""
    lines = [f"## {details.filename}"]
    lines.append("")
    lines.append(f"**Health:** {_health_badge(details.extraction_health)}")
    lines.append(f"**Format:** {details.format.upper()}")
    lines.append(f"**Chunks:** {details.chunk_count or 0}")
    if details.page_count:
        lines.append(f"**Pages:** {details.page_count}")
    if details.extractor_used:
        lines.append(f"**Extracted with:** {details.extractor_used}")
    lines.append("")

    if details.title:
        lines.append(f"**Title:** {details.title}")
    if details.authors:
        lines.append(f"**Authors:** {details.authors}")
    if details.year:
        lines.append(f"**Year:** {details.year}")
    if details.doi:
        lines.append(f"**DOI:** {details.doi}")
    lines.append("")

    if details.tags:
        lines.append(f"**Tags:** {', '.join(details.tags)}")
    if details.folders:
        lines.append(f"**Folders:** {', '.join(details.folders)}")
    if details.keywords:
        lines.append(f"**Keywords:** {', '.join(details.keywords[:10])}"
                     + (f" (and {len(details.keywords) - 10} more)" if len(details.keywords) > 10 else ""))
    lines.append("")

    lines.append(f"**Document ID:** `{details.id}`")
    lines.append(f"**Source:** `{details.source_original}`")
    lines.append(f"**Hash:** `{details.doc_hash}`")
    lines.append("")

    if details.notes:
        lines.append("### Notes")
        lines.append(details.notes)
        lines.append("")

    if details.ingestion_history:
        lines.append("### Ingestion history")
        for event in details.ingestion_history[:10]:
            ts = event["timestamp"].strftime("%Y-%m-%d %H:%M") if event["timestamp"] else "?"
            lines.append(f"- {ts}: **{event['event_type']}** "
                         f"(extractor: {event.get('extractor') or '-'}, "
                         f"chunks: {event.get('chunks_produced') or '-'}, "
                         f"health: {event.get('health_status') or '-'})")
            if event.get("notes"):
                lines.append(f"  - notes: {event['notes']}")

    return "\n".join(lines)


def _help_message() -> str:
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
# Command parser
# ============================================================

async def handle_command(message: str) -> bool:
    """If the message is a command, handle it and return True.
    Returns False if not a command, so normal chat handling proceeds.
    """
    msg = message.strip()
    if not msg.startswith("/"):
        return False

    parts = msg[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "help":
        await cl.Message(content=_help_message()).send()
        return True

    arg = arg.lower() if arg else ""

    if cmd == "library":
        # Parse the arg as either a health filter, format filter, or nothing
        health = format_filter = None
        if arg in ("broken", "marginal", "healthy"):
            health = arg
        elif arg in ("pdf", "epub", "html", "htm", "docx", "txt", "md", "odt", "rtf"):
            format_filter = arg
        elif arg:
            await cl.Message(content=f"Unknown library filter: `{arg}`. Try `/help`.").send()
            return True

        docs = list_documents(health=health, format=format_filter)
        summary = library_summary()
        
        filter_desc = None
        if health:
            filter_desc = f"health: {health}"
        elif format_filter:
            filter_desc = f"format: {format_filter}"

        await cl.Message(content=_format_summary_message(summary, docs, filter_desc)).send()
        return True

    if cmd == "document":
        if not arg:
            await cl.Message(content="Usage: `/document <id>` — provide the document ID prefix.").send()
            return True
        full_id = find_document_by_short_id(arg) if len(arg) < 36 else arg
        if not full_id:
            await cl.Message(content=f"No document found matching `{arg}`. Try `/library` to see IDs.").send()
            return True
        details = get_document_details(full_id)
        if not details:
            await cl.Message(content=f"Could not load details for `{arg}`.").send()
            return True
        await cl.Message(content=_format_document_details(details)).send()
        return True

    await cl.Message(content=f"Unknown command: `/{cmd}`. Try `/help`.").send()
    return True

# ============================================================
# CLI
# ============================================================

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("counter", TokenCounter())
    await cl.Message(
        content=f"📚 **Document assistant ready.** {rag.chunk_count()} chunks indexed.",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Try to handle command first
    if await handle_command(message.content):
        return
    
    history = cl.user_session.get("history") or []
    counter: TokenCounter = cl.user_session.get("counter")
    user_question = message.content

    # Snapshot before this turn (for per-turn delta)
    pre_in, pre_out = counter.input_tokens, counter.output_tokens

    if history:
        async with cl.Step(name="Understanding context", type="tool") as step:
            standalone = rag.rewrite(user_question, history, counter=counter)
            step.output = f"Searching for: {standalone}"
    else:
        standalone = user_question

    async with cl.Step(name="Searching documents", type="retrieval") as step:
        docs = rag.retrieve(standalone, top_k=TOP_K)
        step.output = f"Found {len(docs)} relevant passages"

    source_elements = []
    for i, doc in enumerate(docs):
        preview = doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else "")
        source_elements.append(
            cl.Text(
                name=f"[{i + 1}]",
                content=f"**{format_citation(doc, i + 1)}**\n\n{preview}",
                display="side",
            )
        )

    msg = cl.Message(content="", elements=source_elements)
    full_answer = ""
    for token in rag.stream_answer(standalone, docs, counter=counter):
        full_answer += token
        await msg.stream_token(token)

    # Token usage for this turn
    turn_in = counter.input_tokens - pre_in
    turn_out = counter.output_tokens - pre_out
    turn_total = turn_in + turn_out

    sources_block = "\n\n---\n**Sources:**\n" + "\n".join(
        format_citation(doc, i + 1) for i, doc in enumerate(docs)
    )
    
    usage_block = (
        f"\n\n---\n"
        f"📊 **This turn:** {turn_in:,} in + {turn_out:,} out = {turn_total:,} tokens "
        f"(~${(turn_in * 1.0 + turn_out * 5.0) / 1_000_000:.4f})  \n"
        f"**Session total:** {counter.total():,} tokens "
        f"(~${counter.cost_usd():.4f})"
    )
    
    msg.content = full_answer + sources_block + usage_block
    await msg.update()

    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("history", history)



