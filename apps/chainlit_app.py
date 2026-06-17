"""Chainlit web UI — thin shell over doc_assistant core.

All business logic lives in src/doc_assistant/. This file only does:
1. Chainlit lifecycle hooks (@cl.on_chat_start, @cl.on_message)
2. Streaming token delivery
3. Source element rendering
4. Provenance capture (per-answer record + collapsible card)
"""

import contextlib
import time

import chainlit as cl

from doc_assistant import export
from doc_assistant.commands import execute_command, parse_command
from doc_assistant.config import (
    LLM_MODEL,
    LLM_PROVIDER,
    REVIEWER_EVIDENCE_CHARS,
    SYNTHESIS_MODE,
    TOP_K,
    USE_PARENT_CHILD,
)
from doc_assistant.embeddings import get_active_model_name
from doc_assistant.figures import load_figure_image_paths
from doc_assistant.pipeline import RAGPipeline, format_citation
from doc_assistant.prompts import ANSWER_PROMPT
from doc_assistant.provenance import (
    AnswerProvenance,
    ConfidenceSignals,
    RetrievedChunk,
    adjudicate_claim,
    compute_confidence_signals,
    prompt_version_hash,
    record_answer,
    record_claims,
    template_hash,
)
from doc_assistant.query_router import answer_library_query, is_library_query
from doc_assistant.reviewer import ReviewResult, persist_review, review_answer
from doc_assistant.synthesis import (
    MARKER_OK,
    Claim,
    audit_citations,
    render_evidence_markdown,
    segment_claims,
)
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
    cl.user_session.set("export_turns", [])
    cl.user_session.set("session_id", time.strftime("%Y%m%d-%H%M%S"))
    await cl.Message(
        content=(
            f"📚 **Document assistant ready.** {rag.chunk_count()} chunks indexed.  \n"
            f"🤖 Generation model: `{LLM_PROVIDER}/{LLM_MODEL}` · "
            f"🧬 Embeddings: `{get_active_model_name()}`  \n"
            "_Tip: `/export` downloads this conversation; `/export-debug` writes a dev "
            "bundle (sources + scores + figures + log) to `data/exports/`._"
        ),
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


def _token_suffix(prov: AnswerProvenance) -> str:
    """Header token tag — provider-aware. Local models report no usage, so a
    `0 tokens` figure would be misleading; show `local` instead."""
    if LLM_PROVIDER.lower() == "ollama":
        return " · local"
    total = (prov.token_input or 0) + (prov.token_output or 0)
    return f" · {total:,} tokens"


def _format_provenance_card(
    prov: AnswerProvenance,
    signals: ConfidenceSignals,
    *,
    review: ReviewResult | None = None,
) -> str:
    """Render an AnswerProvenance as a plain-markdown card (no raw HTML).

    Clean answers get a compact three-line block; when a confidence signal
    fires the block expands with the signal breakdown, the reviewer verdict,
    and the full per-source reranker scores, led by a ⚠ chip. Filenames are
    not repeated — they live in the always-visible "Sources:" block; the card
    keys scores by source number. Full per-chunk metadata is in the DB /
    `/export-record`.
    """
    id8 = prov.id[:8]
    latency_s = (prov.latency_ms or 0.0) / 1000.0
    meta = (
        f"**Model** `{prov.model_name or '?'}` · "
        f"**Embedding** `{prov.embedding_model or '?'}` · "
        f"**top_k** {prov.top_k} · **parent-child** {prov.use_parent_child}"
    )
    hint = f"_Review:_ `/review {id8}` · _Export:_ `/export-record {id8}`"

    if not signals.any():
        top = (
            f" · **top reranker** `{signals.max_score:.3f}`"
            if signals.max_score is not None
            else ""
        )
        return (
            f"\n\n---\n"
            f"🔍 **Provenance** — `{id8}` · {latency_s:.1f}s{_token_suffix(prov)}{top}  \n"
            f"{meta}  \n"
            f"{hint}"
        )

    sig_lines = (
        f"- max reranker score: `{signals.max_score:.3f}`"
        f"{' ⚠' if signals.weak_retrieval else ''}  \n"
        f"- top-3 score span: `{signals.top3_span:.3f}`"
        f"{' ⚠' if signals.score_cluster_concern else ''}  \n"
        f"- unique source documents: `{signals.unique_sources}`"
        f"{' ⚠' if signals.single_source_risk else ''}"
    )
    score_lines = "\n".join(
        f"- [{i + 1}] reranker `{c.reranker_score:.3f}`"
        if c.reranker_score is not None
        else f"- [{i + 1}] reranker `-`"
        for i, c in enumerate(prov.retrieved_chunks)
    )
    review_block = _format_review_block(review)
    return (
        f"\n\n---\n"
        f"⚠ **Low confidence: {', '.join(signals.reasons)}** — "
        f"`{id8}` · {latency_s:.1f}s{_token_suffix(prov)}  \n"
        f"{meta}  \n"
        f"**Prompt version** `{prov.prompt_version or '?'}`\n\n"
        f"**Confidence signals**  \n{sig_lines}"
        f"{review_block}\n\n"
        f"**Reranker scores** (by source number above)\n{score_lines}\n\n"
        f"{hint}"
    )


# ============================================================
# Chunk 2a — dual interpretation (evidence vs AI synthesis)
# ============================================================


def _build_retrieved_chunks(scored: list[tuple[object, float]]) -> list[RetrievedChunk]:
    """Build the provenance RetrievedChunk list from (doc, score) pairs."""
    chunks: list[RetrievedChunk] = []
    for doc, score in scored:
        meta = doc.metadata  # type: ignore[attr-defined]
        chunks.append(
            RetrievedChunk(
                filename=meta.get("filename"),
                doc_id=meta.get("document_id") or meta.get("doc_hash"),
                page=meta.get("page"),
                section=meta.get("section"),
                reranker_score=float(score),
                chunk_excerpt=doc.page_content[:300],  # type: ignore[attr-defined]
                # Wider grounding for the reviewer (not persisted/displayed).
                full_text=doc.page_content[:REVIEWER_EVIDENCE_CHARS],  # type: ignore[attr-defined]
            )
        )
    return chunks


def _build_claim_review(claims: list[Claim], claim_ids: list[str]) -> tuple[str, list[cl.Action]]:
    """Render the adjudication section + per-claim buttons for *flagged* claims only.

    Quiet on clean answers (UX: inform, don't clutter): claims marked ``ok`` get
    no buttons; only ``weak``/``unsupported`` claims surface accept/reject/edit.
    All claims are persisted regardless (the eager adjudication log).
    """
    flagged = [(c, cid) for c, cid in zip(claims, claim_ids, strict=True) if c.marker != MARKER_OK]
    if not flagged:
        return (
            f"\n\n---\n🔎 **Interpretation** — all {len(claims)} claim(s) grounded "
            "in cited evidence.",
            [],
        )
    lines = [f"\n\n---\n⚠ **{len(flagged)} claim(s) to review** (evidence vs interpretation):"]
    actions: list[cl.Action] = []
    for c, cid in flagged:
        n = c.claim_index + 1
        badge = "unsupported" if c.marker != "weak" else "weakly grounded"
        lines.append(f"- **#{n}** {c.text}  _({badge})_")
        actions.append(
            cl.Action(name="claim_accept", payload={"id": cid, "n": n}, label=f"✓ #{n}")
        )
        actions.append(
            cl.Action(name="claim_reject", payload={"id": cid, "n": n}, label=f"✗ #{n}")
        )
        actions.append(cl.Action(name="claim_edit", payload={"id": cid, "n": n}, label=f"✎ #{n}"))
    return "\n".join(lines), actions


async def _resolve_claim(action: cl.Action, decision: str) -> None:
    p = action.payload
    try:
        adjudicate_claim(str(p["id"]), decision)
        mark = "✓ accepted" if decision == "accepted" else "✗ rejected"
        await cl.Message(content=f"Claim #{p['n']} {mark}.").send()
    except Exception as e:
        await cl.Message(content=f"⚠ Could not record claim #{p['n']}: {e}").send()


@cl.action_callback("claim_accept")
async def _on_claim_accept(action: cl.Action) -> None:
    await _resolve_claim(action, "accepted")


@cl.action_callback("claim_reject")
async def _on_claim_reject(action: cl.Action) -> None:
    await _resolve_claim(action, "rejected")


@cl.action_callback("claim_edit")
async def _on_claim_edit(action: cl.Action) -> None:
    p = action.payload
    cl.user_session.set("awaiting_edit", {"id": str(p["id"]), "n": p["n"]})
    await cl.Message(content=f"✏️ Send the corrected text for claim #{p['n']}.").send()


# ============================================================
# Export — markdown transcript (user) + dev bundle + per-turn log
# ============================================================


def _export_sources(
    scored: list[tuple[object, float]], fig_paths: dict[str, str]
) -> list[export.ExportSource]:
    """Map (doc, score) pairs to the export's source view (figure paths attached)."""
    sources: list[export.ExportSource] = []
    for i, (doc, score) in enumerate(scored):
        meta = doc.metadata  # type: ignore[attr-defined]
        fig_id = meta.get("figure_id", "")
        is_figure = meta.get("chunk_type") == "figure"
        sources.append(
            export.ExportSource(
                n=i + 1,
                filename=meta.get("filename"),
                page=meta.get("page"),
                section=meta.get("section"),
                reranker_score=float(score),
                is_figure=is_figure,
                image_path=fig_paths.get(fig_id) if is_figure else None,
                excerpt=doc.page_content[:300],  # type: ignore[attr-defined]
            )
        )
    return sources


def _append_export_turn(turn: export.ExportTurn) -> None:
    """Stash a turn in the session transcript and append its event to the session log."""
    turns = cl.user_session.get("export_turns") or []
    turns.append(turn)
    cl.user_session.set("export_turns", turns)
    session_id = cl.user_session.get("session_id") or "session"
    with contextlib.suppress(Exception):  # the log is a sidecar — never break a turn
        export.append_log_event(session_id, export.log_event(turn))


async def _send_conversation_export(*, dev: bool) -> None:
    """Render the session's turns to markdown, write to data/exports/, offer download."""
    turns = cl.user_session.get("export_turns") or []
    if not turns:
        await cl.Message(content="Nothing to export yet — ask a question first.").send()
        return
    session_id = cl.user_session.get("session_id") or "session"
    flavour = "debug" if dev else "transcript"
    md = export.render_conversation_markdown(
        turns, title=f"doc_assistant session {session_id}", dev=dev
    )
    path = export.write_markdown(f"{session_id}-{flavour}.md", md)
    await cl.Message(
        content=f"📄 Exported {len(turns)} turn(s) — {flavour}. Saved to `{path}`.",
        elements=[cl.File(name=path.name, path=str(path), display="inline")],
    ).send()


@cl.action_callback("export_conversation")
async def _on_export(action: cl.Action) -> None:
    await _send_conversation_export(dev=bool(action.payload.get("dev", False)))


def _export_action() -> cl.Action:
    return cl.Action(
        name="export_conversation",
        payload={"dev": False},
        label="⬇ Export chat",
        tooltip="Download this conversation as markdown",
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
        # Export commands need the live session transcript, so they're handled here
        # (stateful) rather than in the stateless commands.execute_command dispatcher.
        if cmd in ("export", "export-conversation", "export_conversation"):
            await _send_conversation_export(dev=False)
            return
        if cmd in ("export-debug", "export_debug"):
            await _send_conversation_export(dev=True)
            return
        response = execute_command(cmd, arg)
        await cl.Message(content=response).send()
        return

    # --- Chunk 2a: claim edit follow-up (a prior "✎ Edit" set this) ---
    pending_edit = cl.user_session.get("awaiting_edit")
    if pending_edit is not None:
        cl.user_session.set("awaiting_edit", None)
        try:
            adjudicate_claim(pending_edit["id"], "edited", edited_text=message.content)
            await cl.Message(content=f"✏️ Claim #{pending_edit['n']} updated.").send()
        except Exception as e:
            await cl.Message(content=f"⚠ Edit failed: {e}").send()
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

    # Feature 4c: a retrieved figure chunk (chunk_type='figure') carries the Figure
    # sidecar id — render its cropped PNG inline beside the text card so figures are
    # visible, not just described. Batch the path lookup (one DB read for the turn).
    fig_ids = [
        fid
        for doc in docs
        if doc.metadata.get("chunk_type") == "figure" and (fid := doc.metadata.get("figure_id"))
    ]
    fig_paths = load_figure_image_paths(fig_ids) if fig_ids else {}

    source_elements: list[cl.Text | cl.Image] = []
    for i, doc in enumerate(docs):
        preview = doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else "")
        source_elements.append(
            cl.Text(
                name=f"[{i + 1}]",
                content=f"**{format_citation(doc, i + 1)}**\n\n{preview}",
                display="side",
            )
        )
        fig_path = fig_paths.get(doc.metadata.get("figure_id", ""))
        if fig_path:
            source_elements.append(
                cl.Image(name=f"[{i + 1}] figure", path=fig_path, display="inline", size="medium")
            )

    retrieved_chunks = _build_retrieved_chunks(scored)

    # --- SYNTHESIS_MODE=human: evidence only; skip the interpretation call ---
    if SYNTHESIS_MODE == "human":
        with contextlib.suppress(Exception):  # provenance is a sidecar, never blocks
            record_answer(
                query=standalone,
                original_query=user_question if standalone != user_question else None,
                answer="(human synthesis mode — evidence only; no AI interpretation)",
                retrieved_chunks=retrieved_chunks,
                embedding_model=get_active_model_name(),
                top_k=TOP_K,
                use_parent_child=USE_PARENT_CHILD,
                latency_ms=(time.monotonic() - turn_start) * 1000.0,
            )
        await cl.Message(
            content=(
                "🧑 **Human synthesis mode** — evidence only; the interpretation is yours.\n\n"
                + render_evidence_markdown(retrieved_chunks)
            ),
            elements=source_elements,
            actions=[_export_action()],
        ).send()
        _append_export_turn(
            export.ExportTurn(
                question=user_question,
                answer="(human synthesis mode — evidence only; no AI interpretation)",
                standalone_query=standalone,
                sources=_export_sources(scored, fig_paths),
                embedding_model=get_active_model_name(),
            )
        )
        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": "(human mode: evidence only)"})
        cl.user_session.set("history", history)
        return

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
    record_id: str | None = None
    review: ReviewResult | None = None
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
        # PR 5.1 — quiet UI on clean answers, loud on flagged ones. The card
        # ALWAYS renders (so the provenance id for `/review`/`/export-record`
        # and the active model are visible on every answer): a compact neutral
        # line on clean answers, a full ⚠ block when a confidence signal fires.
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
        provenance_block = _format_provenance_card(prov, signals, review=review)
    except Exception as e:
        # Never let provenance failure break the answer.
        provenance_block = f"\n\n_⚠ Provenance capture failed: {e}_"

    sources_block = "\n\n---\n**Sources:**\n" + "\n".join(
        format_citation(doc, i + 1) for i, doc in enumerate(docs)
    )

    if LLM_PROVIDER.lower() == "ollama":
        # Local models report no token usage to the LangChain callback, so the
        # real counts are zero — showing "0 tokens / $0.0000" reads as broken.
        # Be honest: no metered cost, with a rough output estimate from text.
        est_out = max(0, len(full_answer) // 4)
        usage_block = (
            f"\n\n---\n"
            f"🖥 **Local model** (`{LLM_PROVIDER}/{LLM_MODEL}`) — no metered token "
            f"cost; provider reports no usage. (~{est_out:,} output tokens, estimated.)"
        )
    else:
        usage_block = (
            f"\n\n---\n"
            f"📊 **This turn:** {turn_in:,} in + {turn_out:,} out "
            f"= {turn_total:,} tokens "
            f"(~${(turn_in * 1.0 + turn_out * 5.0) / 1_000_000:.4f})  \n"
            f"**Session total:** {counter.total():,} tokens "
            f"(~${counter.cost_usd():.4f})"
        )

    # --- Chunk 2a: segment + eager-persist claims; surface flagged ones with buttons ---
    review_block = ""
    actions: list[cl.Action] = []
    if record_id is not None:
        try:
            claims = segment_claims(full_answer, retrieved_chunks)
            claim_ids = record_claims(record_id, claims)
            review_block, actions = _build_claim_review(claims, claim_ids)
        except Exception as e:
            review_block = f"\n\n_⚠ Claim adjudication unavailable: {e}_"
    actions.append(_export_action())  # always offer the download button
    msg.actions = actions

    # Post-hoc citation audit — quiet unless the model cited badly (out-of-range
    # numbers or malformed forms the [n] parser silently drops). Surface, don't rewrite.
    citation = audit_citations(full_answer, len(docs))
    citation_block = "" if citation.clean else f"\n\n---\n⚠ **Citation check:** {citation.note()}"

    msg.content = (
        full_answer
        + sources_block
        + usage_block
        + provenance_block
        + review_block
        + citation_block
    )
    await msg.update()

    # --- Export: stash this turn + append the per-turn debug log event ---
    reviewer_summary = None
    if review is not None and not review.error:
        reviewer_summary = (
            f"faithfulness {review.faithfulness}/5 · citation {review.citation_density}/5 · "
            f"hedging {review.hedging_adequacy}/5"
        )
    _append_export_turn(
        export.ExportTurn(
            question=user_question,
            answer=full_answer,
            standalone_query=standalone,
            sources=_export_sources(scored, fig_paths),
            reviewer_summary=reviewer_summary,
            failure_tag=(review.failure_tag if review is not None else None),
            citation_note=citation.note(),
            token_input=turn_in,
            token_output=turn_out,
            latency_ms=latency_ms,
            model_name=getattr(rag.llm, "model", None) or getattr(rag.llm, "model_name", None),
            embedding_model=embedding_model,
            record_id=record_id,
        )
    )

    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": full_answer})
    cl.user_session.set("history", history)
