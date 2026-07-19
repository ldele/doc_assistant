"""Build the self-organizing wiki / synthesis layer (Feature 6).

Clusters the library (connected components over the ``DocSimilarity`` graph),
summarises each topic with the configured generator, and writes Obsidian-
compatible topic notes under ``WIKI_DIR`` (``{topic_id}.md`` + ``.manifest.json``).

Enrichment-Layer Pattern: idempotent, sidecar markdown only — never the chunk
store. Notes are regenerated each run; the drift report flags topics added /
removed since the last build.

Summarisation is the *generator* role and is fully local-capable — point it at a
local model (`--provider ollama` or `WIKI_LLM_PROVIDER=ollama`) to build for free.

Clustering primitive: by default, connected components over `DocSimilarity` at/above
`--min-similarity` (corpus-dependent — saturates on same-domain libraries). Pass
`--use-concept-communities` (or `WIKI_USE_CONCEPT_COMMUNITIES=true`) to cluster by
the concept-skeleton's Louvain communities instead (threshold-free); that path reads
`data/skeleton/skeleton.json`, so run `build_concept_skeleton --apply` first, and it
falls back to cosine clustering if the sidecar is absent.

Usage:
    python -m scripts.build_wiki                       # dry-run: clusters only, no LLM
    python -m scripts.build_wiki --apply               # summarise + write notes
    python -m scripts.build_wiki --apply --force       # wipe + rebuild all notes
    python -m scripts.build_wiki --apply --provider ollama --model llama3
    python -m scripts.build_wiki --apply --use-concept-communities  # skeleton clustering
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.config import (
    WIKI_LLM_MODEL,
    WIKI_LLM_PROVIDER,
    WIKI_MIN_SIMILARITY,
    WIKI_USE_CONCEPT_COMMUNITIES,
)
from doc_assistant.knowledge.wiki import WikiBuildResult, build_wiki

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(result: WikiBuildResult) -> str:
    notes = result.notes
    multi = [n for n in notes if len(n.docs) > 1]
    flagged = [n for n in notes if n.gap.any()]

    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Clustering:                {result.clustering}")
    out.append(f"Topics (clusters):         {len(notes)}")
    out.append(f"  Multi-document topics:    {len(multi)}")
    out.append(f"  Singletons:               {len(notes) - len(multi)}")
    out.append(f"  With gap signals:         {len(flagged)}")
    if result.applied:
        out.append(f"Notes written:             {result.written}")
        out.append(f"Stale notes removed:       {result.removed_files}")
        out.append(f"Drift: +{len(result.drift.added)} topic(s) / -{len(result.drift.removed)}")
    out.append("=" * 76)
    out.append("")
    out.append(f"{'topic':<14} {'docs':>4} {'links':>5} {'gap':<22} title")
    out.append("-" * 76)
    for n in notes:
        gap = ", ".join(n.gap.reasons) or "-"
        out.append(
            f"{n.topic_id:<14} {len(n.docs):>4} {len(n.links):>5} {gap[:21]:<22} {n.title[:28]}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Summarise + write notes (LLM calls)")
    parser.add_argument("--force", action="store_true", help="Wipe existing notes and rebuild all")
    parser.add_argument(
        "--provider", type=str, default=WIKI_LLM_PROVIDER, help="LLM provider (anthropic | ollama)"
    )
    parser.add_argument("--model", type=str, default=WIKI_LLM_MODEL, help="LLM model")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=WIKI_MIN_SIMILARITY,
        help="Cluster-merge cosine threshold (default %(default)s)",
    )
    parser.add_argument(
        "--use-concept-communities",
        action=argparse.BooleanOptionalAction,
        default=WIKI_USE_CONCEPT_COMMUNITIES,
        help=(
            "Cluster by the concept-skeleton's communities (threshold-free) "
            "instead of --min-similarity; falls back to cosine if data/skeleton is "
            "absent (default %(default)s). Run `build_concept_skeleton --apply` first."
        ),
    )
    args = parser.parse_args()

    client = None
    if args.apply:
        from doc_assistant.llm import ProviderCostError, assert_provider_intent, make_client

        try:
            assert_provider_intent(
                args.provider,
                operation="wiki topic summarisation",
                model=args.model,
                scope="every topic cluster",
            )
        except ProviderCostError as e:
            print(e)
            return 1
        client = make_client(args.provider, args.model)
        print(f"Summarising topics with {args.provider}/{args.model}...")

    result = build_wiki(
        apply=args.apply,
        force=args.force,
        client=client,
        min_similarity=args.min_similarity,
        use_concept_communities=args.use_concept_communities,
    )
    if args.use_concept_communities and result.clustering != "concept-graph":
        print(
            "Note: --use-concept-communities requested but the concept skeleton was "
            "absent; fell back to cosine clustering. Run "
            "`python -m scripts.build_concept_skeleton --apply` first."
        )
    print(_format_report(result))
    if not args.apply:
        print("\nDry run. Pass --apply to summarise topics and write notes.")
    else:
        from doc_assistant.config import WIKI_DIR

        print(f"\nWiki written to {WIKI_DIR} — opens directly in Obsidian.")
        if result.drift.any():
            print(
                f"Drift since last build: added {result.drift.added or '-'}, "
                f"removed {result.drift.removed or '-'}."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
