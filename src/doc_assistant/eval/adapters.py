"""doc_assistant-specific adapters for the generic eval harness.

**This is the only module in ``doc_assistant.eval`` that depends on
the rest of ``doc_assistant``.** Extracting the harness into a
standalone repo (Feature 5) means dropping this file and writing a new
one in the consumer project.

Two adapters, two ways into the app (spec ``eval-harness-chat.md``, decision 6):

* ``rag_pipeline_adapter`` calls ``RAGPipeline`` directly: retrieval and the answer, nothing of
  the chat around them. Old run ids were measured this way, so it stays and stays comparable.
  It can also see the order before the rerank.
* ``controller_adapter`` drives the headless ``ChatController`` the API uses, turn by turn: the
  records router, the conversation history and the prompt framing are all on its path, and it
  replays a conversation's earlier turns. **It writes**: every turn is recorded in the library
  it runs on, as in the app. Run it on a data home of its own, never on a library you keep.

Both put the app's own verdicts on ``EvalOutput.raw`` for the generic scorers to read (the keys
are listed in ``scorers.py``): the citation audit from ``synthesis.audit_citations``, the
filenames the answer validly cited, the route the turn took. The scorers never restate them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from doc_assistant.eval.cases import EvalCase
from doc_assistant.eval.results import EvalOutput
from doc_assistant.tracking import TokenCounter

if TYPE_CHECKING:
    from doc_assistant.chat_controller import ChatController
    from doc_assistant.pipeline import RAGPipeline


def _audit_fields(answer: str, filenames: list[str | None]) -> dict[str, Any]:
    """The app's citation audit of ``answer`` against sources numbered 1..n, and the filenames
    of the sources it validly cited, as ``raw`` keys."""
    from doc_assistant.synthesis import audit_citations

    audit = audit_citations(answer, len(filenames))
    cited = [filenames[n - 1] for n in audit.valid if filenames[n - 1]]
    return {"citation_audit": asdict(audit), "cited_filenames": cited}


def rag_pipeline_adapter(pipeline: RAGPipeline) -> Callable[[str], EvalOutput]:
    """Return a callable ``(query: str) -> EvalOutput`` over ``pipeline``.

    Per-query lifecycle:

    1. Spin up a fresh ``TokenCounter`` so token cost is per-case, not
       per-run.
    2. Run retrieval, collect filenames of returned chunks as citations.
    3. Stream the answer to exhaustion, accumulating the text.
    4. Return ``EvalOutput`` with answer, citations, token counts, and
       a ``raw`` dict carrying the original query (the store uses it).
    """

    def _call(query: str) -> EvalOutput:
        counter = TokenCounter()
        trace: dict[str, Any] = {}
        docs = [doc for doc, _ in pipeline.retrieve_with_scores(query, trace=trace)]
        citations: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            filename = doc.metadata.get("filename")
            if filename and filename not in seen:
                seen.add(filename)
                citations.append(str(filename))

        # Per-chunk descriptors so a retrieval-shape scorer (e.g. the Feature 4c
        # FigureRetrievalScorer) can see *what kind* of chunk came back, not just
        # the filenames. Kept generic — plain dicts, no doc_assistant types.
        retrieved = [
            {
                "filename": doc.metadata.get("filename"),
                "page": doc.metadata.get("page"),
                "chunk_type": doc.metadata.get("chunk_type"),
                "figure_id": doc.metadata.get("figure_id"),
            }
            for doc in docs
        ]

        chunks: list[str] = []
        for chunk in pipeline.stream_answer(query, docs, counter=counter):
            chunks.append(chunk)
        answer = "".join(chunks).strip()

        return EvalOutput(
            answer=answer,
            citations=citations,
            token_input=counter.input_tokens or None,
            token_output=counter.output_tokens or None,
            raw={
                "query": query,
                "n_retrieved": len(docs),
                "retrieved": retrieved,
                "retrieved_pre_rerank": list(trace.get("pre_rerank", [])),
                "route": "rag",
                **_audit_fields(answer, [doc.metadata.get("filename") for doc in docs]),
            },
        )

    return _call


def controller_adapter(controller: ChatController) -> Callable[[EvalCase], EvalOutput]:
    """Return a callable ``(case) -> EvalOutput`` that plays the case through ``controller``.

    Per case: a fresh ``Session`` (no history leaks between cases), the earlier turns replayed
    in order and their results discarded, then the scored turn. A turn that ends in a
    ``Failure`` raises, so the runner records the error rather than a score on an answer that
    was never written; a failed *earlier* turn raises too, since the scored turn would then be
    answered against a different conversation.

    ``retrieved`` is the scored turn's sources in rank order. The controller does not expose the
    order before the rerank, so ``retrieved_pre_rerank`` is not recorded and those rank scorers
    report it as not scored.
    """
    from doc_assistant.chat_controller import Session
    from doc_assistant.chat_controller.events import Failure, Result

    def _turn(session: Session, text: str) -> Any:
        for event in controller.handle_message(session, text):
            if isinstance(event, Failure):
                raise RuntimeError(f"turn failed: {event.message}")
            if isinstance(event, Result):
                return event.result
        raise RuntimeError("turn ended without a result")

    def _call(case: EvalCase) -> EvalOutput:
        session = Session()
        for earlier in case.prior_turns:
            _turn(session, earlier)
        before_in = session.counter.input_tokens
        before_out = session.counter.output_tokens
        result = _turn(session, case.query)
        filenames = [s.filename for s in result.sources]
        token_in = session.counter.input_tokens - before_in
        token_out = session.counter.output_tokens - before_out
        return EvalOutput(
            answer=result.answer.strip(),
            citations=list(dict.fromkeys(f for f in filenames if f)),
            token_input=token_in or None,
            token_output=token_out or None,
            raw={
                "query": case.query,
                "n_turns": len(case.prior_turns) + 1,
                "n_retrieved": len(result.sources),
                "retrieved": [
                    {"filename": s.filename, "figure_id": s.figure_id} for s in result.sources
                ],
                "route": result.route,
                "standalone_query": result.standalone_query,
                **_audit_fields(result.answer, filenames),
            },
        )

    return _call


def embedding_callable(pipeline: RAGPipeline) -> Callable[[str], list[float]]:
    """Adapt the pipeline's loaded embedder for ``EmbeddingSimilarityScorer``.

    Avoids loading a second HuggingFace model just for scoring. Returns
    a callable ``(text: str) -> list[float]``.
    """

    def _embed(text: str) -> list[float]:
        return [float(x) for x in pipeline.embeddings.embed_query(text)]

    return _embed
