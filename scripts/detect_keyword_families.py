"""Detect candidate keyword families — report only, nothing is written to the DB.

Runs the two zero-LLM detection tiers (feature-tag-families.md, PR-2) over every keyword not
already a family's canonical or alias: Tier 1 morphological stem-matching (``llm``/``llms``),
Tier 2 bge-embedding cosine clustering (``connectome``/``connectomics``). Prints proposals for a
human to review — via the desktop app's Manage-keywords view, or the family CRUD directly. This
script never creates, renames, or deletes a family; there is no ``--apply``.

Usage::

    python -m scripts.detect_keyword_families                 # both tiers (loads bge)
    python -m scripts.detect_keyword_families --no-embeddings  # Tier 1 only, instant, $0
    python -m scripts.detect_keyword_families --threshold 0.9  # stricter Tier 2
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from doc_assistant.knowledge.keyword_families import DEFAULT_EMBEDDING_THRESHOLD, FamilyProposal
from doc_assistant.library import detect_family_candidates

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(proposals: list[FamilyProposal], *, embeddings_used: bool) -> str:
    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Proposals:      {len(proposals)}")
    out.append(f"Embedding tier: {'ran' if embeddings_used else 'skipped (--no-embeddings)'}")
    out.append("=" * 76)
    out.append("")
    if not proposals:
        out.append(
            "Nothing to propose (every keyword is already familied, or no group met the "
            "detection thresholds)."
        )
        return "\n".join(out)
    for p in proposals:
        out.append(f"  [{p.tier:13s} conf={p.confidence:4.2f}]  {p.canonical}")
        for m in p.members:
            out.append(f"      + {m}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip Tier 2 (no model load) — Tier 1 morphological proposals only",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_EMBEDDING_THRESHOLD,
        help="Tier-2 cosine similarity threshold (default %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model registry key for Tier 2 (default: the active retrieval model)",
    )
    args = parser.parse_args()

    embed_fn: Callable[[list[str]], list[list[float]]] | None = None
    if not args.no_embeddings:
        from doc_assistant.knowledge.concept_semantics import embed_texts

        model = args.model

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return embed_texts(texts, model=model)

    print("Detecting keyword families (report only — nothing is written to the DB)...")
    proposals = detect_family_candidates(embed_fn=embed_fn, embedding_threshold=args.threshold)
    print(_format_report(proposals, embeddings_used=embed_fn is not None))
    print("Review + accept proposals via the Manage-keywords view or the family CRUD routes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
