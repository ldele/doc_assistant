"""Self-eval: run a set of questions through the pipeline locally and judge them.

Drives the RAG pipeline over a question set with **local** generation (Ollama by
default — free), runs the reference-free reviewer for a verdict on each answer
(pass / concern / fail, graded against the answer's own retrieved evidence), and
writes a dev bundle (sources + reranker scores + figures + reviewer + verdict) plus a
per-turn JSONL log to ``data/exports/``. A fast way to sanity-check answer quality
across many questions without billing the API or driving the UI by hand.

Generation is forced local via ``pipeline.build_chat_model`` (no ``.env`` edit); the
reviewer uses the same provider. Defaults to Ollama and routes through the cost guard,
so ``--provider anthropic`` warns before spending. Read-only over the corpus — writes
only the export sidecar.

Usage:
    python -m scripts.self_eval                        # built-in dev questions, Ollama
    python -m scripts.self_eval --questions qs.txt     # one question per line
    python -m scripts.self_eval --model llama3.1:8b --limit 5
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from doc_assistant import export
from doc_assistant.config import CONCEPT_GRAPH_LLM_MODEL, TOP_K
from doc_assistant.embeddings import get_active_model_name
from doc_assistant.figures import load_figure_image_paths
from doc_assistant.provenance import AnswerProvenance, RetrievedChunk
from doc_assistant.reviewer import review_answer, verdict_from_review

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# A small corpus-relevant default set (the library is RAG / IR papers). Override with
# --questions to point at your own list (one per line).
_DEFAULT_QUESTIONS = [
    "What is dense passage retrieval and how does it differ from BM25?",
    "How does a cross-encoder reranker improve retrieval quality?",
    "What does the ColBERT late-interaction architecture do?",
    "Explain HyDE (hypothetical document embeddings).",
    "How is retrieval-augmented generation (RAG) structured?",
    "What risks does the responsible-AI-usage work raise about LLMs?",
]


def _load_questions(path: str | None) -> list[str]:
    if not path:
        return list(_DEFAULT_QUESTIONS)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [q.strip() for q in lines if q.strip() and not q.lstrip().startswith("#")]


def _build_turn(rag: object, question: str, reviewer: object, model: str) -> export.ExportTurn:
    """Run one question end-to-end (local) and judge it into an ExportTurn."""
    scored = rag.retrieve_with_scores(question, top_k=TOP_K)  # type: ignore[attr-defined]
    docs = [doc for doc, _ in scored]
    answer = "".join(rag.stream_answer(question, docs))  # type: ignore[attr-defined]

    fig_ids = [
        fid
        for doc, _ in scored
        if doc.metadata.get("chunk_type") == "figure" and (fid := doc.metadata.get("figure_id"))
    ]
    fig_paths = load_figure_image_paths(fig_ids)

    sources: list[export.ExportSource] = []
    rchunks: list[RetrievedChunk] = []
    for i, (doc, score) in enumerate(scored):
        meta = doc.metadata
        is_fig = meta.get("chunk_type") == "figure"
        sources.append(
            export.ExportSource(
                n=i + 1,
                filename=meta.get("filename"),
                page=meta.get("page"),
                section=meta.get("section"),
                reranker_score=float(score),
                is_figure=is_fig,
                image_path=fig_paths.get(meta.get("figure_id", "")) if is_fig else None,
                excerpt=doc.page_content[:300],
            )
        )
        rchunks.append(
            RetrievedChunk(
                filename=meta.get("filename"),
                doc_id=meta.get("document_id") or meta.get("doc_hash"),
                page=meta.get("page"),
                section=meta.get("section"),
                reranker_score=float(score),
                chunk_excerpt=doc.page_content[:300],
            )
        )

    prov = AnswerProvenance(
        id="self-eval",
        query=question,
        answer=answer,
        retrieved_chunks=rchunks,
        model_name=model,
        embedding_model=get_active_model_name(),
    )
    review = review_answer(prov, reviewer)  # type: ignore[arg-type]
    label, reason = verdict_from_review(review)
    reviewer_summary = (
        None
        if review.error
        else (
            f"faithfulness {review.faithfulness}/5 · citation {review.citation_density}/5 · "
            f"hedging {review.hedging_adequacy}/5"
        )
    )
    return export.ExportTurn(
        question=question,
        answer=answer,
        sources=sources,
        reviewer_summary=reviewer_summary,
        failure_tag=None if review.error else review.failure_tag,
        verdict=f"{label} — {reason}",
        model_name=model,
        embedding_model=get_active_model_name(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=str, help="File with one question per line")
    parser.add_argument(
        "--provider", type=str, default="ollama", help="LLM provider (ollama | anthropic)"
    )
    parser.add_argument(
        "--model", type=str, default=CONCEPT_GRAPH_LLM_MODEL, help="Model (default %(default)s)"
    )
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N questions")
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]
    if not questions:
        print("No questions to run.")
        return 1

    # Cost guard: ollama is a no-op; anthropic warns + needs a key before spending.
    from doc_assistant.llm import ProviderCostError, assert_provider_intent, make_client

    try:
        assert_provider_intent(
            args.provider,
            operation="self-eval (local generation + reviewer)",
            model=args.model,
            scope=f"{len(questions)} question(s), 2 calls each",
        )
    except ProviderCostError as e:
        print(e)
        return 1

    from doc_assistant.pipeline import RAGPipeline, build_chat_model

    print(f"Running {len(questions)} question(s) with {args.provider}/{args.model}...")
    rag = RAGPipeline()
    rag.llm = build_chat_model(args.provider, args.model)  # force local generation
    reviewer = make_client(args.provider, args.model)

    session_id = time.strftime("%Y%m%d-%H%M%S")
    turns: list[export.ExportTurn] = []
    for i, q in enumerate(questions, 1):
        turn = _build_turn(rag, q, reviewer, args.model)
        turns.append(turn)
        export.append_log_event(session_id, export.log_event(turn))
        print(f"  [{i}/{len(questions)}] {turn.verdict}  — {q[:60]}")

    md = export.render_conversation_markdown(
        turns, title=f"Self-eval {session_id} ({args.provider}/{args.model})", dev=True
    )
    path = export.write_markdown(f"self-eval-{session_id}.md", md)

    tally = Counter(str(t.verdict).split(" ")[0] for t in turns)
    print("\n" + "=" * 60)
    print(f"Verdicts: {dict(tally)}")
    print(f"Bundle:   {path}")
    print(f"Log:      data/exports/session-{session_id}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
