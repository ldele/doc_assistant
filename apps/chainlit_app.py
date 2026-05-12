import chainlit as cl
from doc_assistant.pipeline import RAGPipeline, format_citation
from doc_assistant.tracking import TokenCounter

rag = RAGPipeline()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("counter", TokenCounter())
    await cl.Message(
        content=f"📚 **Document assistant ready.** {rag.chunk_count()} chunks indexed.",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
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
        docs = rag.retrieve(standalone)
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