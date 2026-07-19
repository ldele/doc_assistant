"""Suggest concepts semantically (#2) — grounds the vocabulary in meaning, not frequency.

Three modes (combine freely):
  --from-abstracts   candidate concepts from each paper's title + abstract (scientific papers;
                     a document with no abstract falls back — nothing is invented for it)
  --anchor-ranked    full-text candidates re-ranked by cosine to the title+abstract anchor —
                     full-text recall + abstract precision (the unified extractor)
  --near             curated concepts that are near-duplicates by embedding cosine (merge hints)

None of them writes anything — these are curation aids. Pair them with `seed_concepts --add`.
The embedder for the two semantic modes defaults to SPECTER2 (academic); override with --model.

Usage:
    python -m scripts.suggest_concepts --from-abstracts
    python -m scripts.suggest_concepts --anchor-ranked --top-k 12 --model specter2
    python -m scripts.suggest_concepts --near --threshold 0.85 --model specter2
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.config import (
    ABSTRACT_CONCEPTS_TOP_K,
    CONCEPT_EMBED_MODEL,
    CONCEPT_MERGE_COSINE,
)
from doc_assistant.knowledge.concept_semantics import (
    anchor_ranked_candidates,
    concept_merge_suggestions,
    suggest_from_abstracts,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-abstracts", action="store_true", help="Candidate concepts from title + abstract"
    )
    parser.add_argument(
        "--anchor-ranked",
        action="store_true",
        help="Full-text candidates re-ranked by title+abstract cosine (recall + precision)",
    )
    parser.add_argument(
        "--near",
        action="store_true",
        help="Near-duplicate curated concept pairs (embedding cosine)",
    )
    parser.add_argument(
        "--doc", default=None, metavar="ID", help="Restrict to one document id (candidate modes)"
    )
    parser.add_argument(
        "--top-k", type=int, default=ABSTRACT_CONCEPTS_TOP_K, help="Candidates per document"
    )
    parser.add_argument(
        "--pool-k",
        type=int,
        default=80,
        help="Full-text pool size before re-ranking (--anchor-ranked)",
    )
    parser.add_argument(
        "--threshold", type=float, default=CONCEPT_MERGE_COSINE, help="Cosine threshold (--near)"
    )
    parser.add_argument(
        "--model", default=CONCEPT_EMBED_MODEL, help="Embedding model for the semantic modes"
    )
    args = parser.parse_args()

    if not (args.from_abstracts or args.anchor_ranked or args.near):
        parser.error("choose --from-abstracts, --anchor-ranked, and/or --near")

    docs = [args.doc] if args.doc else None

    if args.from_abstracts:
        results = suggest_from_abstracts(docs, top_k=args.top_k)
        print(f"=== title+abstract candidates ({len(results)} document(s)) ===")
        for _doc_id, filename, candidates in results:
            shown = ", ".join(candidates) if candidates else "(no extractable abstract — skipped)"
            print(f"\n{filename}\n  {shown}")

    if args.anchor_ranked:
        results = anchor_ranked_candidates(
            docs, top_k=args.top_k, pool_k=args.pool_k, model=args.model
        )
        print(
            f"\n=== anchor-ranked candidates (model={args.model}, {len(results)} document(s)) ==="
        )
        for _doc_id, filename, scored in results:
            if not scored:
                print(f"\n{filename}\n  (no anchor/pool — skipped)")
                continue
            shown = ", ".join(f"{s.term} [{s.anchor_cosine:.2f}]" for s in scored)
            print(f"\n{filename}\n  {shown}")

    if args.near:
        pairs = concept_merge_suggestions(threshold=args.threshold, model=args.model)
        print(
            f"\n=== near-duplicate concepts (model={args.model}, cosine >= {args.threshold}) ==="
        )
        if not pairs:
            print("  none above threshold")
        for p in pairs:
            print(f"  {p.cosine:.3f}  {p.label_a}  ~  {p.label_b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
