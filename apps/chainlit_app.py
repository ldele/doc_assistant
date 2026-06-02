"""Chainlit web UI — thin shell over doc_assistant core.

All business logic lives in src/doc_assistant/. This file only does:
1. Chainlit lifecycle hooks (@cl.on_chat_start, @cl.on_message)
2. Streaming token delivery
3. Source element rendering
4. Provenance capture (per-answer record + collapsible card)
"""

import time

import chainlit as cl

from doc_assistant.commands import execute_command, parse_command
from doc_assistant.config import TOP_K, USE_PARENT_CHILD
from doc_assistant.embeddings import get_active_model_name
from doc_assistant.pipeline import RAGPipeline, format_citation
from doc_assistant.prompts import ANSWER_PROMPT
from doc_assistant.provenance import (
    AnswerProvenance,
    ConfidenceSignals,
    RetrievedChunk,
    compute_confidence_signals,
    prompt_version_hash,
    record_answer,
    template_hash,
)
from doc_assistant.query_router import answer_library_query, is_library_query
from doc_assistant.reviewer import ReviewResult, persist_review, review_answer
from doc_assistant.tracking import TokenCounter

rag = RAGPipeline()

# Cached at startup — prompt template doesn't change between turns.
_ANSWER_TEMPLATE_HASH = template_hash(str(ANSWER_PROMPT))


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
# Provenance card formatting (inline, collapsible)
# ============================================================


def _format_review_block(review: ReviewResult | None) -> str:
    """Render the reviewer's verdict as a sub-section of the provenance card."""
    if review is None:
        return ""
    if review.error:
        return f"\n\n**Reviewer:** _failed — {review.error}_"
    bits = [
        f"faithfulness `{review.faithfulness}/5`",
        f"citation density `{review.citation_density}/5`",
        f"hedging `{review.hedging_adequacy}/5`",
        f"unsupported claims: `{review.unsupported_claims_count}`",
    ]
    notes = f"  \n_Reviewer notes:_ {review.notes}" if review.notes else ""
    return "\n\n**Reviewer assessment:** " + " · ".join(bits) + notes


def _format_provenance_card(
    prov: AnswerProvenance,
    signals: ConfidenceSignals,
    *,
    expanded: bool,
    review: ReviewResult | None = None,
) -> str:
    """Render an AnswerProvenance as a collapsible markdown card.

    When ``signals.any()`` is True, the summary line leads with a ⚠
    chip listing the fired reasons and the card opens expanded by
    default. Otherwise it stays collapsed with a neutral summary.
    """
    chunks_lines = []
    for i, c in enumerate(prov.retrieved_chunks):
        score = f"{c.reranker_score:.3f}" if c.reranker_score is not None else "-"
        page = f" p.{c.page}" if c.page is not None else ""
        section = f' "{c.section}"' if c.section else ""
        chunks_lines.append(
            f"- **[{i + 1}]** `{score}` · {c.filename or 'unknown'}{page}{section}"
        )

    cost_in = prov.token_input or 0
    cost_out = prov.token_output or 0
    latency_s = (prov.latency_ms or 0.0) / 1000.0

    if signals.any():
        summary_prefix = f"⚠ <b>Low confidence</b>: {', '.join(signals.reasons)}"
        signals_block = (
            "\n\n**Confidence signals:**  \n"
            f"- max reranker score: `{signals.max_score:.3f}`"
            f"{' ⚠' if signals.weak_retrieval else ''}  \n"
            f"- top-3 score span: `{signals.top3_span:.3f}`"
            f"{' ⚠' if signals.score_cluster_concern else ''}  \n"
            f"- unique source documents: `{signals.unique_sources}`"
            f"{' ⚠' if signals.single_source_risk else ''}"
        )
    else:
        summary_prefix = "🔍 <b>Provenance</b>"
        signals_block = ""

    open_attr = " open" if expanded else ""

    review_block = _format_review_block(review)

    return (
        f"\n\n<details{open_attr}>\n<summary>{summary_prefix} — "
        f"<code>{prov.id[:8]}</code> · {latency_s:.1f}s · "
        f"{cost_in + cost_out:,} tokens</summary>\n\n"
        f"**Model:** `{prov.model_name or '?'}` · "
        f"**Embedding:** `{prov.embedding_model or '?'}` · "
        f"**top_k:** {prov.top_k} · "
        f"**parent-child:** {prov.use_parent_child}  \n"
        f"**Prompt version:** `{prov.prompt_version or '?'}`  \n"
        f"**Tokens:** {cost_in:,} in + {cost_out:,} out"
        f"{signals_block}"
        f"{review_block}\n\n"
        "**Retrieved chunks (with reranker scores):**\n\n"
        + "\n".join(chunks_lines)
        + f"\n\n_Export full record:_ `/export-record {prov.id[:8]}`\n\n"
        "</details>"
    )


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
    turn_start = time.monotonic()

    if history:
        async with cl.Step(name="Understanding context", type="tool") as step:
            standalone = rag.rewrite(user_question, history, counter=counter)
            step.output = f"Searching for: {standalone}"
    else:
        standalone = user_question

    async with cl.Step(name="Searching documents", type="retrieval") as step:
        scored = rag.retrieve_with_scores(standalone, top_k=TOP_K)
        step.output = f"Found {len(scored)} relevant passages"

    docs = [doc for doc, _ in scored]

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
    latency_ms = (time.monotonic() - turn_start) * 1000.0

    # --- Provenance capture (sidecar; never blocks the answer) ---
    embedding_model = get_active_model_name()
    prov_version = prompt_version_hash(
        template_hash=_ANSWER_TEMPLATE_HASH,
        top_k=TOP_K,
        use_parent_child=USE_PARENT_CHILD,
        embedding_model=embedding_model,
    )
    retrieved_chunks = [
        RetrievedChunk(
            filename=doc.metadata.get("filename"),
            doc_id=doc.metadata.get("document_id") or doc.metadata.get("doc_hash"),
            page=doc.metadata.get("page"),
            section=doc.metadata.get("section"),
            reranker_score=float(score),
            chunk_excerpt=doc.page_content[:300],
        )
        for doc, score in scored
    ]
    try:
        record_id = record_answer(
            query=standalone,
            original_query=user_question if standalone != user_question else None,
            answer=full_answer,
            retrieved_chunks=retrieved_chunks,
            model_name=getattr(rag.llm, "model", None) or getattr(rag.llm, "model_name", None),
            embedding_model=embedding_model,
            prompt_version=prov_version,
            top_k=TOP_K,
            use_parent_child=USE_PARENT_CHILD,
            token_input=turn_in,
            token_output=turn_out,
            latency_ms=latency_ms,
        )
        prov = AnswerProvenance(
            id=record_id,
            query=standalone,
            original_query=user_question if standalone != user_question else None,
            answer=full_answer,
            retrieved_chunks=retrieved_chunks,
            model_name=getattr(rag.llm, "model", None) or getattr(rag.llm, "model_name", None),
            embedding_model=embedding_model,
            prompt_version=prov_version,
            top_k=TOP_K,
            use_parent_child=USE_PARENT_CHILD,
            token_input=turn_in,
            token_output=turn_out,
            latency_ms=latency_ms,
        )
        signals = compute_confidence_signals(prov)
        # PR 5.1 — quiet UI on clean answers, loud on flagged ones.
        # Record always persists (visible via /records and /export-record).
        review: ReviewResult | None = None
        if signals.any():
            # PR 6 — when heuristic flags fire AND a reviewer is available,
            # run the LLM reviewer to add depth. ~$0.001 + ~1-2s per flagged
            # answer (free + local under Ollama). Clean answers skip the call.
            from doc_assistant.llm import get_reviewer_client, reviewer_available

            if reviewer_available():
                try:
                    from doc_assistant.config import REVIEWER_MODEL

                    review = review_answer(prov, get_reviewer_client())
                    persist_review(
                        record_id, review, reviewer_kind="llm_haiku", model_name=REVIEWER_MODEL
                    )
                except Exception as e:
                    review = ReviewResult(error=f"reviewer setup failed: {e}")
            provenance_block = _format_provenance_card(prov, signals, expanded=True, review=review)
        else:
            provenance_block = ""
    except Exception as e:
        # Never let provenance failure break the answer.
        provenance_block = f"\n\n_⚠ Provenance capture failed: {e}_"

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

    msg.content = full_answer + sources_block + usage_block + provenance_block
    await msg.update()

    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("history", history)
