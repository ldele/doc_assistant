"""``ChatController`` — the turn orchestration itself.

Slash-command dispatch, library-query routing, history-aware rewrite, retrieval, figure lookup,
source assembly, answer streaming, provenance capture, confidence gating + (flagged-only) reviewer,
claim segmentation + persistence, citation audit, usage accounting and export stashing."""

from __future__ import annotations

import contextlib
import hashlib
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import structlog
from langchain_core.documents import Document

from doc_assistant import app_settings, compare, conversations, corpus_stats, export, library
from doc_assistant.chat_controller.events import Result, Step, Token, TurnEvent
from doc_assistant.chat_controller.helpers import (
    _build_claims_block,
    _build_retrieved_chunks,
    _build_source_views,
    _export_sources,
    _format_provenance_card,
    _is_local,
    _ProvenanceInputs,
    _ProvenanceOutcome,
    _resolve_scope,
    _resolve_turn_knobs,
    _scope_dict,
    _scope_label,
    _scope_note,
    _sources_block,
)
from doc_assistant.chat_controller.session import RagOverrides, Session
from doc_assistant.chat_controller.views import (
    ClaimView,
    ScopeView,
    SourceEpistemics,
    SourceEvalSummary,
    SourceView,
    TurnResult,
    UsageView,
)
from doc_assistant.commands import execute_command, parse_command
from doc_assistant.config import (
    EPISTEMICS_MARKERS_ENABLED,
    TOP_K,
    USE_MULTI_QUERY,
    USE_PARENT_CHILD,
)
from doc_assistant.embeddings import get_active_model_name
from doc_assistant.ingest.figures import load_figure_image_paths
from doc_assistant.knowledge.epistemics import (
    ChunkEval,
    current_graph_version,
    derive_markers,
    load_source_evaluations,
)
from doc_assistant.library import document_years
from doc_assistant.pipeline import RAGPipeline, format_citation
from doc_assistant.prompts import ANSWER_PROMPT
from doc_assistant.provenance import (
    AnswerProvenance,
    RetrievedChunk,
    adjudicate_claim,
    compute_confidence_signals,
    prompt_version_hash,
    record_answer,
    template_hash,
)
from doc_assistant.query_router import answer_library_query, is_library_query
from doc_assistant.reviewer import ReviewResult, persist_review, review_answer
from doc_assistant.synthesis import audit_citations, render_evidence_markdown
from doc_assistant.tracking import TokenCounter

log = structlog.get_logger(__name__)


# ============================================================
# Controller
# ============================================================


class ChatController:
    """Owns the turn orchestration. Stateless across turns (the injected ``Session``
    carries per-conversation state). Imports the same library functions the old
    the original UI handler did; no UI-framework import."""

    def __init__(self, rag: RAGPipeline | None = None) -> None:
        if rag is not None:
            self.rag = rag  # test seam (cpc §13) — a fake; never carries a persisted selection
        else:
            self.rag = RAGPipeline()
            # ADR-011 (U1c): apply any persisted provider/model selection so a restart restores
            # it. A fresh RAGPipeline boots on the config default; only swap if the persisted
            # choice actually differs (skip a needless rebuild on the common no-switch boot).
            provider, model = app_settings.get_llm_selection()
            if (
                provider is not None
                and model is not None
                and (provider, model) != (self.rag.provider, self.rag.model)
            ):
                self.rag.set_chat_model(provider, model)
        # Cached once — the prompt template doesn't change between turns.
        self._answer_template_hash = template_hash(str(ANSWER_PROMPT))

    # -- public API -------------------------------------------------------

    def chunk_count(self) -> int:
        return self.rag.chunk_count()

    def corpus_stats(self) -> corpus_stats.CorpusStats:
        """What this corpus costs on this machine (ADR-037) — the Settings "Corpus" panel's facts.

        Assembled here rather than in the API shell because it needs the **live pipeline's** arm
        (whether the on-disk keyword index is the one actually serving), and reaching through a
        controller into a pipeline's private state from a router would put logic in the shell.
        """
        return corpus_stats.corpus_stats(
            documents=library.count_documents(),
            chunks=self.chunk_count(),
            keyword_index_on_disk=self.rag.sparse_index_active,
        )

    def rebuild_keyword_index(self) -> int:
        """Rebuild the on-disk keyword index and swap it in; returns the chunk count (ADR-037)."""
        return self.rag.rebuild_sparse_index()

    def reconfigure(self, provider: str, model: str) -> None:
        """Switch the live generation provider/model (ADR-011, U1c desktop provider switch).

        Validates and persists the choice via ``app_settings`` (raises :class:`ValueError` for
        an unknown or keyless provider — the API maps that to 400), then swaps the pipeline's
        generation model with a **direct method call** — never a module-global mutation. An
        in-flight turn already holds its own chain reference (``pipeline.set_chat_model``'s own
        guarantee) and finishes on the old model; the very next turn picks up the new one.
        """
        app_settings.set_llm_selection(provider, model)
        self.rag.set_chat_model(provider, model)

    def refresh_chat_model(self) -> tuple[str, str]:
        """Rebuild the generation model from the *current* credentials, changing no selection.

        ADR-034: a key saved in the app while the process runs must reach the next turn. The
        pipeline's chat model was constructed with whatever key existed at boot (for a first-run
        install: none), so the credential change is only live once the model is rebuilt — hence a
        method distinct from :meth:`reconfigure`, which persists a *choice* the user did not make
        here. Returns the effective ``(provider, model)`` for the caller to report.
        """
        provider, model = app_settings.effective_llm()
        self.rag.set_chat_model(provider, model)
        log.info("chat_model_refreshed", provider=provider, model=model)
        return provider, model

    def compare_retrieval(
        self,
        text: str,
        overrides: RagOverrides,
        scope_folder_id: str | None = None,
    ) -> compare.CompareResult:
        """Retrieval-only A/B compare (U6, ``feature-ab-compare-sandbox.md``).

        Runs ``retrieve_with_scores`` twice on the same raw query — A = locked defaults, B = the
        session ``overrides`` — and returns both ranked source sets + the diff + note. **$0**:
        retrieval only, no generation, no ``self.llm`` touch. Request-scoped: ``overrides`` rides
        the call, no module-global assigned (the ADR-010 isolation invariant). Only
        ``top_k``/``use_multi_query`` affect retrieval; the rest are answer-time (see the note).

        ``scope_folder_id`` (ADR-025 F2) is applied to **both** sides. The comparison exists to
        isolate a knob, so the document set is held constant — and an unscoped diff shown while a
        folder scope is active would describe retrieval the next answer will not perform.
        """
        scope, scope_view = _resolve_scope(scope_folder_id)
        eff_a: dict[str, int | bool] = {"top_k": TOP_K, "use_multi_query": USE_MULTI_QUERY}
        eff_b: dict[str, int | bool] = {
            "top_k": overrides.top_k if overrides.top_k is not None else TOP_K,
            "use_multi_query": (
                overrides.use_multi_query
                if overrides.use_multi_query is not None
                else USE_MULTI_QUERY
            ),
        }
        # A follows the global default (use_multi_query=None); B forces the override's value.
        pairs_a = self.rag.retrieve_with_scores(
            text, top_k=int(eff_a["top_k"]), use_multi_query=None, scope=scope
        )
        pairs_b = self.rag.retrieve_with_scores(
            text,
            top_k=int(eff_b["top_k"]),
            use_multi_query=overrides.use_multi_query,
            scope=scope,
        )
        return compare.build_result(
            text,
            [self._to_compare_source(d, s, i + 1) for i, (d, s) in enumerate(pairs_a)],
            [self._to_compare_source(d, s, i + 1) for i, (d, s) in enumerate(pairs_b)],
            eff_a,
            eff_b,
            _scope_label(scope_view),
        )

    @staticmethod
    def _to_compare_source(doc: Document, score: float, rank: int) -> compare.CompareSource:
        """Map one retrieved ``(Document, score)`` to a :class:`compare.CompareSource`.

        ``identity`` reuses the pipeline's dedup key (``doc_hash + "_" + sha256(page_content)``)
        so a source appearing on both sides matches exactly."""
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        identity = f"{doc.metadata.get('doc_hash', '')}_{content_hash}"
        return compare.CompareSource(
            rank=rank,
            filename=str(doc.metadata.get("filename", "unknown")),
            page=doc.metadata.get("page"),
            section=doc.metadata.get("section"),
            score=float(score),
            excerpt=doc.page_content[:240].strip(),  # a short preview for the compare card
            citation=format_citation(doc, rank),
            identity=identity,
        )

    def adjudicate(self, claim_id: str, decision: str, edited_text: str | None = None) -> None:
        """Record the user's verdict on one flagged claim. Lifts ``_resolve_claim``'s
        core; the renderer owns the success/error messaging (it catches and displays)."""
        adjudicate_claim(claim_id, decision, edited_text=edited_text)

    def export_conversation(self, session: Session, *, dev: bool) -> tuple[str, Path | None]:
        """Render a conversation to markdown, write to ``data/exports/``, and return
        ``(message, path)``. ``path`` is ``None`` when there is nothing to export.

        Sources from the **durable ``AnswerRecord`` transcript** by ``session_id`` — so a
        reopened or resumed past chat exports the same as a live one (the earlier turns live
        only in the store). The in-memory ``export_turns`` are richer (reviewer, figures,
        citation audit), so the dev bundle prefers them when this is a live session."""
        in_memory = session.export_turns
        durable = conversations.conversation_export_turns(session.session_id)
        turns = in_memory if (dev and in_memory) else (durable or in_memory)
        if not turns:
            return ("Nothing to export yet — ask a question first.", None)
        flavour = "debug" if dev else "transcript"
        subtitle = (
            f"Exported {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · session "
            f"{session.session_id}"
        )
        md = export.render_conversation_markdown(
            turns, title="Provenote conversation", subtitle=subtitle, dev=dev
        )
        path = export.write_markdown(f"{session.session_id}-{flavour}.md", md)
        return (f"📄 Exported {len(turns)} turn(s) — {flavour}. Saved to `{path}`.", path)

    def handle_message(
        self,
        session: Session,
        text: str,
        *,
        overrides: RagOverrides | None = None,
        scope_folder_id: str | None = None,
    ) -> Iterator[TurnEvent]:
        """Drive one turn. Ports ``on_message``'s dispatch order verbatim:
        (a) slash command, (b) pending claim-edit, (c) library query, (d) RAG path.

        ``overrides`` (ADR-010) only affects the RAG path — commands/library queries/claim
        edits have no retrieval or synthesis-mode knobs to override. Default ``None`` is
        byte-identical to before this feature existed.

        ``scope_folder_id`` (ADR-025 F2) restricts retrieval to one folder's documents for this
        turn only — request-scoped like ``overrides``, never stored on the session: a scope the
        backend remembered would be a scope the user could forget (spec S9). Same RAG-path-only
        carve."""
        # --- Slash commands ---
        parsed = parse_command(text)
        if parsed is not None:
            cmd, arg = parsed
            try:
                # Export commands need the live session transcript, so they're handled here
                # (stateful) rather than in the stateless commands.execute_command dispatcher.
                if cmd in ("export", "export-conversation", "export_conversation"):
                    msg, path = self.export_conversation(session, dev=False)
                    yield Result(self._command_result(msg, download_path=path))
                elif cmd in ("export-debug", "export_debug"):
                    msg, path = self.export_conversation(session, dev=True)
                    yield Result(self._command_result(msg, download_path=path))
                else:
                    yield Result(self._command_result(execute_command(cmd, arg)))
            except Exception as e:
                # A failing command (empty/missing DB, no API key, …) must not break the
                # turn or the SSE stream — surface it as a normal result.
                yield Result(self._command_result(f"⚠ `/{cmd}` failed: {e}"))
            return

        # --- Chunk 2a: claim edit follow-up (a prior "✎ Edit" set this) ---
        pending_edit = session.awaiting_edit
        if pending_edit is not None:
            session.awaiting_edit = None
            try:
                adjudicate_claim(pending_edit["id"], "edited", edited_text=text)
                yield Result(self._command_result(f"✏️ Claim #{pending_edit['n']} updated."))
            except Exception as e:
                yield Result(self._command_result(f"⚠ Edit failed: {e}"))
            return

        # --- Library metadata questions (answered from SQLite) ---
        if is_library_query(text):
            try:
                yield Result(self._command_result(answer_library_query(text)))
            except Exception as e:
                yield Result(self._command_result(f"⚠ Library query failed: {e}"))
            return

        # --- RAG pipeline ---
        yield from self._handle_rag(session, text, overrides, scope_folder_id)

    # -- internal ---------------------------------------------------------

    def _command_result(self, answer: str, *, download_path: Path | None = None) -> TurnResult:
        """Wrap a command/library/edit string as a minimal TurnResult (no sources,
        claims, or telemetry blocks)."""
        return TurnResult(
            answer=answer,
            mode="ai",
            sources=[],
            flagged_claims=[],
            usage=UsageView(0, 0, 0, None, _is_local(self.rag.provider)),
            standalone_query="",
            record_id=None,
            provenance_card_md="",
            claim_review_md="",
            sources_md="",
            usage_md="",
            citation_note_md="",
            download_path=download_path,
        )

    def _attach_source_evaluation(
        self,
        sources: list[SourceView],
        scored: list[tuple[Document, float]],
        *,
        markers_enabled: bool = EPISTEMICS_MARKERS_ENABLED,
    ) -> SourceEvalSummary | None:
        """ADR-027 **D3**: attach the always-on per-source epistemic evaluation (coverage/direction
        + doc year) to every retrieved source, and return the strip-level freshness. Independent of
        the D2/E3 answer-influence toggle — the strip **always** renders when a concept graph
        exists; ``markers_enabled`` only governs whether the *answer-surface* marker chips
        (``sv.markers``, E1.1) are populated from the same read. Scoped, indexed reads (KI-18), no
        LLM, no provider touched.

        The per-source join is a direct ``chunk_key`` lookup (E1.1 re-projection — flat + PC both
        resolve). Freshness: ``stale`` when the epistemics sidecar's
        ``graph_version`` differs from the current skeleton's — the graph was rebuilt without a
        ``compute_epistemics`` re-run. Returns ``None`` (no strip) when no concept graph is built.

        Advisory: any failure logs a **WARNING** and returns ``None`` rather than breaking the turn
        — a silent failure under an always-on strip is a silently-lying UI."""
        try:
            current = current_graph_version()
            if current is None:
                return None  # no concept graph built → nothing to assess, no strip
            chunk_keys = [sv.chunk_key for sv in sources if sv.chunk_key is not None]
            evals, sidecar_version = load_source_evaluations(chunk_keys)
            document_ids = [
                str(d) for d in (doc.metadata.get("document_id") for doc, _ in scored) if d
            ]
            years = document_years(document_ids)
            for sv, (doc, _score) in zip(sources, scored, strict=True):
                document_id = doc.metadata.get("document_id")
                year = years.get(str(document_id)) if document_id is not None else None
                ev: ChunkEval | None = (
                    evals.get(sv.chunk_key) if sv.chunk_key is not None else None
                )
                sv.evaluation = SourceEpistemics(
                    coverage=ev.coverage if ev is not None else None,
                    superseded=ev.superseded if ev is not None else False,
                    n_claims=ev.n_claims if ev is not None else 0,
                    year=year,
                )
                if ev is not None and markers_enabled:
                    sv.markers = derive_markers(
                        1 if ev.coverage == "contested" else 0, 1 if ev.superseded else 0
                    )
            return SourceEvalSummary(
                graph_version=sidecar_version or current,
                stale=sidecar_version is not None and sidecar_version != current,
            )
        except Exception as exc:
            # Advisory strip must never break a turn — but never silently, either (see above).
            log.warning("attach_source_evaluation_failed", error=str(exc))
            return None

    def _capture_provenance_and_review(self, pin: _ProvenanceInputs) -> _ProvenanceOutcome:
        """Record the answer's provenance + (when a heuristic signal fires and a reviewer is
        available) run the confined LLM reviewer, returning the rendered card block (E1.2 — the
        88-line block lifted verbatim out of ``_handle_rag``). Never blocks the answer: any failure
        collapses to a "Provenance capture failed" card and an empty ``record_id`` (the caller then
        skips claim adjudication). The ``overrides_note``/``scope_note`` suffix is appended by the
        caller, which owns those turn knobs."""
        prov_version = prompt_version_hash(
            template_hash=self._answer_template_hash,
            top_k=pin.top_k,
            use_parent_child=USE_PARENT_CHILD,
            embedding_model=pin.embedding_model,
        )
        record_id: str | None = None
        review: ReviewResult | None = None
        try:
            record_id = record_answer(
                query=pin.standalone,
                original_query=pin.original_query,
                answer=pin.full_answer,
                retrieved_chunks=pin.retrieved_chunks,
                model_name=pin.model_name,
                embedding_model=pin.embedding_model,
                prompt_version=prov_version,
                top_k=pin.top_k,
                use_parent_child=USE_PARENT_CHILD,
                token_input=pin.token_input,
                token_output=pin.token_output,
                latency_ms=pin.latency_ms,
                session_id=pin.session_id,
                retrieval_scope=_scope_dict(pin.scope_view),
                epistemics_markers_enabled=pin.markers_enabled,
            )
            prov = AnswerProvenance(
                id=record_id,
                query=pin.standalone,
                original_query=pin.original_query,
                answer=pin.full_answer,
                retrieved_chunks=pin.retrieved_chunks,
                model_name=pin.model_name,
                embedding_model=pin.embedding_model,
                prompt_version=prov_version,
                top_k=pin.top_k,
                use_parent_child=USE_PARENT_CHILD,
                token_input=pin.token_input,
                token_output=pin.token_output,
                latency_ms=pin.latency_ms,
            )
            signals = compute_confidence_signals(prov)
            # PR 5.1 — quiet UI on clean answers, loud on flagged ones. The card ALWAYS
            # renders (so the provenance id and active model are visible on every answer):
            # a compact neutral line on clean answers, a full ⚠ block when a signal fires.
            if signals.any():
                # PR 6 — when heuristic flags fire AND a reviewer is available, run the LLM
                # reviewer to add depth. ~$0.001 + ~1-2s per flagged answer (free + local
                # under Ollama). Clean answers skip the call. ADR-011 (U1c): the reviewer
                # follows the effective generation provider unless REVIEWER_PROVIDER is
                # explicitly pinned in the environment (resolve_reviewer's own rule).
                from doc_assistant.llm import (
                    get_reviewer_client,
                    resolve_reviewer,
                    reviewer_available,
                )

                reviewer_provider, reviewer_model = resolve_reviewer(
                    pin.turn_provider, pin.turn_model
                )
                if reviewer_available(reviewer_provider):
                    try:
                        review = review_answer(
                            prov, get_reviewer_client(pin.turn_provider, pin.turn_model)
                        )
                        # ADR-011: the recorded kind must match the instrument that actually ran.
                        # A followed switch to Ollama is no longer the Haiku reviewer — labeling it
                        # "llm_haiku" beside an ollama model_name would be a provenance lie.
                        reviewer_kind = (
                            "llm_haiku"
                            if reviewer_provider == "anthropic"
                            else f"llm_{reviewer_provider}"
                        )
                        persist_review(
                            record_id,
                            review,
                            reviewer_kind=reviewer_kind,
                            model_name=reviewer_model,
                        )
                    except Exception as e:
                        review = ReviewResult(error=f"reviewer setup failed: {e}")
            provenance_block = _format_provenance_card(
                prov,
                signals,
                review=review,
                is_local=_is_local(pin.turn_provider),
                source_strip_rendered=pin.source_strip_rendered,
            )
        except Exception as e:
            # Never let provenance failure break the answer.
            provenance_block = f"\n\n_⚠ Provenance capture failed: {e}_"
        return _ProvenanceOutcome(
            record_id=record_id, provenance_block=provenance_block, review=review
        )

    def _handle_rag(
        self,
        session: Session,
        text: str,
        overrides: RagOverrides | None = None,
        scope_folder_id: str | None = None,
    ) -> Iterator[TurnEvent]:
        rag = self.rag
        history = session.history
        counter = session.counter
        user_question = text

        # --- ADR-011: snapshot the generation instrument for the whole turn. A live provider
        # switch (RAGPipeline.set_chat_model) can land mid-turn; the answer must stream on —
        # and every recorded label (model_name, usage, reviewer resolution) must name — the
        # SAME instrument, so read the trio once here and never through ``rag`` again. ---
        turn_llm = rag.llm
        turn_provider = rag.provider
        turn_model = rag.model

        # --- ADR-010: resolve effective per-turn knobs (None = locked default; never a
        # module-global assignment — request-scoped so concurrent turns can't leak). ---
        knobs = _resolve_turn_knobs(overrides)

        # --- ADR-025 F2: resolve the retrieval scope ONCE for the turn. Membership lives in
        # SQLite and is editable at any moment, so the hash set is read here and then frozen —
        # the answer, the chip, and the provenance record all describe the same set. An unknown
        # or empty folder yields an empty scope and an honest zero-source turn; it never falls
        # back to the whole library (spec S3). ---
        scope, scope_view = _resolve_scope(scope_folder_id)

        pre_in, pre_out = counter.input_tokens, counter.output_tokens
        turn_start = time.monotonic()

        if history:
            standalone = rag.rewrite(user_question, history, counter=counter)
            yield Step("Understanding context", f"Searching for: {standalone}")
        else:
            standalone = user_question

        scored = rag.retrieve_with_scores(
            standalone,
            top_k=knobs.top_k,
            use_multi_query=(overrides.use_multi_query if overrides else None),
            scope=scope,
        )
        yield Step("Searching documents", f"Found {len(scored)} relevant passages")

        docs = [doc for doc, _ in scored]

        # Feature 4c: a retrieved figure chunk (chunk_type='figure') carries the Figure
        # sidecar id — resolve its cropped PNG so a renderer can show it inline. Batch the
        # path lookup (one DB read for the turn).
        fig_ids = [
            fid
            for doc in docs
            if doc.metadata.get("chunk_type") == "figure"
            and (fid := doc.metadata.get("figure_id"))
        ]
        fig_paths = load_figure_image_paths(fig_ids) if fig_ids else {}

        sources = _build_source_views(scored, fig_paths)
        # ADR-027 D3: the always-on source-evaluation strip (per-source coverage/year + freshness);
        # markers_enabled= is U1b's per-turn override over the D2 answer-surface marker chips only.
        source_eval = self._attach_source_evaluation(
            sources, scored, markers_enabled=knobs.markers_enabled
        )
        retrieved_chunks = _build_retrieved_chunks(
            scored, reviewer_evidence_chars=knobs.reviewer_evidence_chars
        )

        # --- synthesis_mode=human (locked default or a per-turn override): evidence only;
        # skip the interpretation call ---
        if knobs.synthesis_mode == "human":
            yield Result(
                self._human_result(
                    session,
                    user_question=user_question,
                    standalone=standalone,
                    scored=scored,
                    fig_paths=fig_paths,
                    sources=sources,
                    retrieved_chunks=retrieved_chunks,
                    turn_start=turn_start,
                    eff_top_k=knobs.top_k,
                    overrides_note=knobs.overrides_note,
                    scope=scope_view,
                    source_eval=source_eval,
                    markers_enabled=knobs.markers_enabled,
                )
            )
            return

        full_answer = ""
        for tok in rag.stream_answer(standalone, docs, counter=counter, llm=turn_llm):
            full_answer += tok
            yield Token(tok)

        turn_in = counter.input_tokens - pre_in
        turn_out = counter.output_tokens - pre_out
        latency_ms = (time.monotonic() - turn_start) * 1000.0

        # --- Provenance capture + reviewer (sidecar; never blocks the answer — E1.2) ---
        embedding_model = get_active_model_name()
        model_name = getattr(turn_llm, "model", None) or getattr(turn_llm, "model_name", None)
        prov_out = self._capture_provenance_and_review(
            _ProvenanceInputs(
                standalone=standalone,
                original_query=user_question if standalone != user_question else None,
                full_answer=full_answer,
                retrieved_chunks=retrieved_chunks,
                model_name=model_name,
                embedding_model=embedding_model,
                top_k=knobs.top_k,
                token_input=turn_in,
                token_output=turn_out,
                latency_ms=latency_ms,
                session_id=session.session_id,
                scope_view=scope_view,
                turn_provider=turn_provider,
                turn_model=turn_model,
                markers_enabled=knobs.markers_enabled,
                # The strip renders iff we have both a summary and sources to list (Turn.svelte's
                # own condition). When it does, it already shows the per-source relevance scores,
                # so the card drops its duplicate copy of them.
                source_strip_rendered=source_eval is not None and bool(sources),
            )
        )
        record_id = prov_out.record_id
        review = prov_out.review
        provenance_block = (
            prov_out.provenance_block + knobs.overrides_note + _scope_note(scope_view)
        )

        sources_block = _sources_block(sources)
        usage_block = self._usage_block(
            full_answer, turn_in, turn_out, counter, provider=turn_provider, model=turn_model
        )

        # --- Chunk 2a: segment + eager-persist claims; surface flagged ones ---
        claim_review_block = ""
        flagged_claims: list[ClaimView] = []
        if record_id is not None:
            claim_review_block, flagged_claims = _build_claims_block(
                record_id, full_answer, retrieved_chunks
            )

        # Post-hoc citation audit — quiet unless the model cited badly (out-of-range
        # numbers or malformed forms the [n] parser silently drops). Surface, don't rewrite.
        citation = audit_citations(full_answer, len(docs))
        citation_block = (
            "" if citation.clean else f"\n\n---\n⚠ **Citation check:** {citation.note()}"
        )

        # --- Export: stash this turn + append the per-turn debug log event ---
        reviewer_summary = None
        if review is not None and not review.error:
            reviewer_summary = (
                f"faithfulness {review.faithfulness}/5 · citation {review.citation_density}/5 · "
                f"hedging {review.hedging_adequacy}/5"
            )
        self._append_export_turn(
            session,
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
                model_name=model_name,
                embedding_model=embedding_model,
                record_id=record_id,
            ),
        )

        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": full_answer})

        yield Result(
            TurnResult(
                answer=full_answer,
                mode="ai",
                sources=sources,
                flagged_claims=flagged_claims,
                usage=UsageView(
                    turn_input=turn_in,
                    turn_output=turn_out,
                    session_total=counter.total(),
                    cost_usd=None if _is_local(turn_provider) else counter.cost_usd(),
                    is_local=_is_local(turn_provider),
                ),
                standalone_query=standalone,
                record_id=record_id,
                provenance_card_md=provenance_block,
                claim_review_md=claim_review_block,
                sources_md=sources_block,
                usage_md=usage_block,
                citation_note_md=citation_block,
                scope=scope_view,
                source_eval=source_eval,
            )
        )

    def _human_result(
        self,
        session: Session,
        *,
        user_question: str,
        standalone: str,
        scored: list[tuple[Document, float]],
        fig_paths: dict[str, str],
        sources: list[SourceView],
        retrieved_chunks: list[RetrievedChunk],
        turn_start: float,
        eff_top_k: int = TOP_K,
        overrides_note: str = "",
        scope: ScopeView | None = None,
        source_eval: SourceEvalSummary | None = None,
        markers_enabled: bool | None = None,
    ) -> TurnResult:
        """``synthesis_mode=human`` (locked default or a per-turn ADR-010 override) —
        evidence only; no interpretation call. Records provenance silently (no card shown),
        stashes the export turn, updates history."""
        human_answer = "(human synthesis mode — evidence only; no AI interpretation)"
        with contextlib.suppress(Exception):  # provenance is a sidecar, never blocks
            record_answer(
                query=standalone,
                original_query=user_question if standalone != user_question else None,
                answer=human_answer,
                retrieved_chunks=retrieved_chunks,
                embedding_model=get_active_model_name(),
                top_k=eff_top_k,
                use_parent_child=USE_PARENT_CHILD,
                latency_ms=(time.monotonic() - turn_start) * 1000.0,
                session_id=session.session_id,
                retrieval_scope=_scope_dict(scope),
                epistemics_markers_enabled=markers_enabled,
            )
        self._append_export_turn(
            session,
            export.ExportTurn(
                question=user_question,
                answer=human_answer,
                standalone_query=standalone,
                sources=_export_sources(scored, fig_paths),
                embedding_model=get_active_model_name(),
            ),
        )
        session.history.append({"role": "user", "content": user_question})
        session.history.append({"role": "assistant", "content": "(human mode: evidence only)"})
        return TurnResult(
            answer=(
                "🧑 **Human synthesis mode** — evidence only; the interpretation is yours.\n\n"
                + render_evidence_markdown(retrieved_chunks)
                + overrides_note
                # Human mode renders no provenance card, so the scope note rides the answer —
                # otherwise a scoped evidence-only turn would state its scope nowhere.
                + _scope_note(scope)
            ),
            mode="human",
            sources=sources,
            flagged_claims=[],
            usage=UsageView(0, 0, session.counter.total(), None, _is_local(self.rag.provider)),
            standalone_query=standalone,
            record_id=None,
            provenance_card_md="",
            claim_review_md="",
            sources_md="",
            usage_md="",
            citation_note_md="",
            scope=scope,
            source_eval=source_eval,
        )

    def _usage_block(
        self,
        full_answer: str,
        turn_in: int,
        turn_out: int,
        counter: TokenCounter,
        *,
        provider: str,
        model: str,
    ) -> str:
        if _is_local(provider):
            # Local models report no token usage to the LangChain callback, so the real
            # counts are zero — showing "0 tokens / $0.0000" reads as broken. Be honest:
            # no metered cost, with a rough output estimate from text. ``provider``/``model``
            # are the caller's turn snapshot (ADR-011) — never read live off ``self.rag``,
            # which a mid-turn switch may already have moved.
            est_out = max(0, len(full_answer) // 4)
            return (
                f"\n\n---\n"
                f"🖥 **Local model** (`{provider}/{model}`) — no metered token "
                f"cost; provider reports no usage. (~{est_out:,} output tokens, estimated.)"
            )
        turn_total = turn_in + turn_out
        return (
            f"\n\n---\n"
            f"📊 **This turn:** {turn_in:,} in + {turn_out:,} out "
            f"= {turn_total:,} tokens "
            f"(~${(turn_in * 1.0 + turn_out * 5.0) / 1_000_000:.4f})  \n"
            f"**Session total:** {counter.total():,} tokens "
            f"(~${counter.cost_usd():.4f})"
        )

    def _append_export_turn(self, session: Session, turn: export.ExportTurn) -> None:
        """Stash a turn in the session transcript and append its event to the session log."""
        session.export_turns.append(turn)
        with contextlib.suppress(Exception):  # the log is a sidecar — never break a turn
            export.append_log_event(session.session_id, export.log_event(turn))
