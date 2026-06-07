"""Diagnostic: does same-source crowding actually occur in retrieval?

Read-only. Tests the premise behind the proposed per-source diversity cap
(``MAX_PARENTS_PER_DOC``) BEFORE building it. Across an eval set it measures,
for each query, how many of the top-k parent passages come from the same
source document.

Decision rule:
  * Crowding rare  -> the cap is a no-op; don't build it. Premise falsified.
  * Crowding common -> build it, and prefer a focused-vs-broad split over a
    flat cap (a naive cap regresses single-paper-deep questions, where the
    answer genuinely needs several passages from one source).

Behaviour-neutral: only calls ``RAGPipeline.retrieve_with_scores`` and counts
metadata. Touches no store, writes nothing. Safe to delete.

Usage::

    uv run --no-sync python -m scripts.diagnose_crowding
    uv run --no-sync python -m scripts.diagnose_crowding \
        --cases tests/eval/cases.public.yaml --threshold 3
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from doc_assistant.config import PROJECT_ROOT, TOP_K
from doc_assistant.eval import load_cases_yaml
from doc_assistant.pipeline import RAGPipeline

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CASES = PROJECT_ROOT / "tests" / "eval" / "cases.public.yaml"


def _source_key(meta: dict) -> str:
    """Stable per-source identity. Prefer filename (readable); fall back to hash."""
    return str(meta.get("filename") or meta.get("doc_hash") or "<unknown>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="eval YAML to drive queries")
    ap.add_argument(
        "--top-k", type=int, default=TOP_K, help=f"passages per query (default {TOP_K})"
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="a query 'crowds' when one source contributes >= this many parents (default 3)",
    )
    args = ap.parse_args()

    cases = load_cases_yaml(args.cases)
    print(f"Loaded {len(cases)} cases from {args.cases}")
    print("Building pipeline (embeddings + reranker + LLM-free retrieval)...")
    pipe = RAGPipeline()

    crowded_queries = 0
    max_per_query: list[int] = []
    worst_offenders: Counter[str] = Counter()

    print("\n--- per-query source distribution ---")
    for case in cases:
        docs = pipe.retrieve(case.query, top_k=args.top_k)
        counts = Counter(_source_key(d.metadata) for d in docs)
        top_source, top_count = (counts.most_common(1)[0] if counts else ("<none>", 0))
        max_per_query.append(top_count)
        crowded = top_count >= args.threshold
        crowded_queries += int(crowded)
        if crowded:
            worst_offenders[top_source] += 1
        flag = "  <-- CROWDED" if crowded else ""
        print(
            f"  {case.id:<28} k={len(docs):<2} sources={len(counts):<2} "
            f"max/source={top_count} ({Path(top_source).name}){flag}"
        )

    n = len(cases)
    hist = Counter(max_per_query)
    print("\n--- summary ---")
    print(f"queries:                 {n}")
    print(f"top_k:                   {args.top_k}")
    print(f"crowd threshold:         >= {args.threshold} parents from one source")
    print(f"crowded queries:         {crowded_queries}/{n} ({crowded_queries / n:.0%})")
    print("max-parents-from-one-source histogram (count of queries):")
    for k in sorted(hist):
        bar = "#" * hist[k]
        print(f"  {k:>2} parents : {hist[k]:>3}  {bar}")
    if worst_offenders:
        print("sources that crowd most often:")
        for src, times in worst_offenders.most_common(5):
            print(f"  {times:>3}x  {Path(src).name}")

    verdict = (
        "CROWDING IS COMMON -> the cap could help; build it with a focused/broad split."
        if crowded_queries / n >= 0.2
        else "CROWDING IS RARE -> the cap would be a no-op; skip it."
    )
    print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
