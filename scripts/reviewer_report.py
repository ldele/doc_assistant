"""Reviewer self-improvement report (Phase 6 / Integrity Chunk 2c).

Aggregates the per-answer reviewer verdicts (``answer_reviews``) into a
min-N-gated failure-tag report, and — with ``--anchor`` — adjudicates each
recurring tag as **reviewer bias** vs **real system fault** by reviewer-scoring
the verified golden set.

Read-only over the sidecar tables (Enrichment-Layer Pattern). **Instrumentation,
not action**: it surfaces recurring faults; a human decides the fix.

Usage:
    python -m scripts.reviewer_report                 # production aggregation (free)
    python -m scripts.reviewer_report --anchor        # + golden-set bias-vs-fault (paid)
    python -m scripts.reviewer_report --anchor --cases tests/eval/cases.public.yaml

The default report is free and offline. ``--anchor`` loads the RAG pipeline and
runs the reviewer over the golden set (generation + reviewer calls — costs money),
so it requires ``ANTHROPIC_API_KEY`` (unless the reviewer is configured local).
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.config import PROJECT_ROOT
from doc_assistant.reviewer_aggregate import (
    ReviewTagRow,
    aggregate_tags,
    classify_bias_vs_fault,
    format_bias_vs_fault,
    format_by_prompt_version,
    format_tag_report,
    golden_tag_rates,
    load_review_tags,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "eval" / "cases.public.yaml"


def _golden_anchor_rows(cases_path: str) -> list[ReviewTagRow]:
    """Reviewer-score the golden set → ``ReviewTagRow`` per case. Paid + online.

    Runs the real pipeline (retrieve + generate) on each golden question, then
    the pinned reviewer on the answer. The golden answers are known-good, so a
    tag the reviewer assigns here is a false-positive (bias) signal.
    """
    from doc_assistant.eval.cases import load_cases_yaml
    from doc_assistant.llm import get_reviewer_client
    from doc_assistant.pipeline import RAGPipeline
    from doc_assistant.provenance import AnswerProvenance, RetrievedChunk
    from doc_assistant.reviewer import review_answer

    cases = load_cases_yaml(cases_path)
    if not cases:
        print(f"No golden cases in {cases_path}")
        return []

    print(f"Loading pipeline + reviewing {len(cases)} golden case(s) (paid)...")
    pipeline = RAGPipeline()
    client = get_reviewer_client()

    rows: list[ReviewTagRow] = []
    for i, case in enumerate(cases):
        print(f"  [{i + 1:>2}/{len(cases)}] {case.id}")
        try:
            scored = pipeline.retrieve_with_scores(case.query)
            chunks = [
                RetrievedChunk(
                    filename=doc.metadata.get("filename"),
                    page=doc.metadata.get("page"),
                    section=doc.metadata.get("section"),
                    reranker_score=score,
                    chunk_excerpt=(doc.page_content or "")[:300],
                )
                for doc, score in scored
            ]
            answer = "".join(pipeline.stream_answer(case.query, [d for d, _ in scored]))
            prov = AnswerProvenance(
                id=f"golden:{case.id}", query=case.query, answer=answer, retrieved_chunks=chunks
            )
            review = review_answer(prov, client)
            if review.error:
                print(f"      reviewer error, skipped: {review.error}")
                continue
            rows.append(
                ReviewTagRow(failure_tag=review.failure_tag or "none", answer_record_id=case.id)
            )
        except Exception as e:  # per-case isolation
            print(f"      error, skipped: {type(e).__name__}: {e}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="Reviewer-score the golden set to adjudicate bias vs fault (paid)",
    )
    parser.add_argument(
        "--cases", type=str, default=str(DEFAULT_GOLDEN), help="Golden cases YAML for --anchor"
    )
    args = parser.parse_args()

    rows = load_review_tags()
    stats, total = aggregate_tags(rows)

    print(format_tag_report(stats, total))
    by_pv = format_by_prompt_version(rows)
    if by_pv:
        print()
        print(by_pv)

    golden_rates = None
    golden_n: int | None = None
    if args.anchor:
        from doc_assistant.config import ANTHROPIC_API_KEY, REVIEWER_PROVIDER

        if REVIEWER_PROVIDER.lower() == "anthropic" and not ANTHROPIC_API_KEY:
            print("\n--anchor needs ANTHROPIC_API_KEY (or set REVIEWER_PROVIDER=ollama).")
            return 1
        golden_rows = _golden_anchor_rows(args.cases)
        golden_rates = golden_tag_rates(golden_rows)
        golden_n = len(golden_rows)

    verdicts = classify_bias_vs_fault(stats, total, golden_rates)
    print()
    print(format_bias_vs_fault(verdicts, golden_n=golden_n))

    if not args.anchor:
        print(
            "\n_Run with `--anchor` to split recurring tags into reviewer-bias vs system-fault._"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
