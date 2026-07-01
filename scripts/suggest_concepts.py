"""Suggest concepts semantically (#2) — grounds the vocabulary in meaning, not frequency.

Two modes (combine freely):
  --from-abstracts   candidate concepts from each paper's title + abstract (scientific papers;
                     a document with no abstract falls back — nothing is invented for it)
  --near             curated concepts that are near-duplicates by embedding cosine (merge hints)

Neither writes anything — these are curation aids. Pair them with `seed_concepts --add`.

Usage:
    python -m scripts.suggest_concepts --from-abstracts
    python -m scripts.suggest_concepts --from-abstracts --doc <id> --top-k 10
    python -m scripts.suggest_concepts --near --threshold 0.85
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.concept_semantics import concept_merge_suggestions, suggest_from_abstracts
from doc_assistant.config import ABSTRACT_CONCEPTS_TOP_K, CONCEPT_MERGE_COSINE

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-abstracts", action="store_true", help="Candidate concepts from title + abstract"
    )
    parser.add_argument(
        "--near",
        action="store_true",
        help="Near-duplicate curated concept pairs (embedding cosine)",
    )
    parser.add_argument(
        "--doc", default=None, metavar="ID", help="Restrict --from-abstracts to one document id"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=ABSTRACT_CONCEPTS_TOP_K,
        help="Candidates per doc (--from-abstracts)",
    )
    parser.add_argument(
        "--threshold", type=float, default=CONCEPT_MERGE_COSINE, help="Cosine threshold (--near)"
    )
    args = parser.parse_args()

    if not args.from_abstracts and not args.near:
        parser.error("choose --from-abstracts and/or --near")

    if args.from_abstracts:
        docs = [args.doc] if args.doc else None
        results = suggest_from_abstracts(docs, top_k=args.top_k)
        print(f"=== title+abstract candidates ({len(results)} document(s)) ===")
        for _doc_id, filename, candidates in results:
            shown = ", ".join(candidates) if candidates else "(no extractable abstract — skipped)"
            print(f"\n{filename}\n  {shown}")

    if args.near:
        pairs = concept_merge_suggestions(threshold=args.threshold)
        print(f"\n=== near-duplicate concepts (cosine >= {args.threshold}) ===")
        if not pairs:
            print("  none above threshold")
        for p in pairs:
            print(f"  {p.cosine:.3f}  {p.label_a}  ~  {p.label_b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
