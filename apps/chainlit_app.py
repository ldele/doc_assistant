"""Chainlit web UI — thin shell over doc_assistant core.

All business logic lives in src/doc_assistant/. This file only does:
1. Chainlit lifecycle hooks (@cl.on_chat_start, @cl.on_message)
2. Streaming token delivery
3. Source element rendering
"""

import chainlit as cl

from doc_assistant.commands import execute_command, parse_command
from doc_assistant.config import TOP_K
from doc_assistant.pipeline import RAGPipeline, format_citation
from doc_assistant.query_router import answer_library_query, is_library_query
from doc_assistant.tracking import TokenCounter

rag = RAGPipeline()


# ============================================================
# Lifecycle
# ============================================================


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])
    cl.user_session.set("counter", TokenCounter())
    await cl.Message(
        content=(f"📚 **Document assistant ready.** {rag.chunk_count()} chunks indexed."),
    ).send()


# ============================================================
# Message handler
# ============================================================


@cl.on_message
async def on_message(message: cl.Message) -> None:
    # --- Slash commands ---
    parsed = parse_command(message.content)
    if parsed is not None:
        cmd, arg = parsed
        response = execute_command(cmd, arg)
        await cl.Message(content=response).send()
        return

    # --- Library metadata questions (answered from SQLite) ---
    if is_library_query(message.content):
        answer = answer_library_query(message.content)
        await cl.Message(content=answer).send()
        return

    # --- RAG pipeline ---
    history = cl.user_session.get("history") or []
    counter: TokenCounter = cl.user_session.get("counter")
    user_question = message.content

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

    turn_in = counter.input_tokens - pre_in
    turn_out = counter.output_tokens - pre_out
    turn_total = turn_in + turn_out

    sources_block = "\n\n---\n**Sources:**\n" + "\n".join(
        format_citation(doc, i + 1) for i, doc in enumerate(docs)
    )

    usage_block = (
        f"\n\n---\n"
        f"📊 **This turn:** {turn_in:,} in + {turn_out:,} out "
        f"= {turn_total:,} tokens "
        f"(~${(turn_in * 1.0 + turn_out * 5.0) / 1_000_000:.4f})  \n"
        f"**Session total:** {counter.total():,} tokens "
        f"(~${counter.cost_usd():.4f})"
    )

    msg.content = full_answer + sources_block + usage_block
    await msg.update()

    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("history", history)
