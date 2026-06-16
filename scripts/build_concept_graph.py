"""Build the cross-document concept graph (Feature 7, PR 16).

Extracts the salient concepts + stated relations from each document, merges them
into one corpus graph (edges tagged EXTRACTED / INFERRED / AMBIGUOUS — structural,
no self-reported confidence), runs Louvain community detection + god-node ranking
+ graph-gap signals, and writes a ``graph.json`` sidecar under ``CONCEPT_GRAPH_DIR``
(plus a per-document extraction cache). Enrichment-Layer Pattern: idempotent,
sidecar only — never the chunk store. Not a graph database — a JSON artifact.

Extraction runs on **local Ollama by default** (free) — NOT the analysis provider,
so it cannot silently bill your Anthropic key the way the wiki/chat paths can.
A document already extracted (cache keyed by doc_hash) is reused unless --force;
re-running rebuilds the graph from cache with zero LLM calls.

Usage:
    python -m scripts.build_concept_graph                  # dry-run: graph from cache, no LLM
    python -m scripts.build_concept_graph --apply          # extract (Ollama) + write graph.json
    python -m scripts.build_concept_graph --apply --force  # re-extract every document
    python -m scripts.build_concept_graph --apply --doc <hash>  # one document only
    python -m scripts.build_concept_graph --apply --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.concept_graph import ConceptGraphResult, build_concept_graph
from doc_assistant.config import (
    CONCEPT_GRAPH_DIR,
    CONCEPT_GRAPH_LLM_MODEL,
    CONCEPT_GRAPH_LLM_PROVIDER,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(result: ConceptGraphResult) -> str:
    g = result.graph
    summ = g.integrity_summary
    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Documents:                 {g.meta.get('n_documents', '?')}")
    out.append(
        f"  Extracted this run:       {result.extracted}   "
        f"(cached {result.cached}, skipped {result.skipped}, errors {result.errors})"
    )
    out.append(f"Concept nodes:             {len(g.nodes)}")
    out.append(f"Relations (edges):         {len(g.edges)}")
    out.append(
        f"  EXTRACTED / INFERRED / AMBIGUOUS: "
        f"{summ['EXTRACTED']} / {summ['INFERRED']} / {summ['AMBIGUOUS']}"
    )
    out.append(f"Communities:               {len(g.communities)}")
    out.append(f"Isolated concepts:         {len(g.gaps.isolated_nodes)}")
    out.append(f"Thin bridges:              {len(g.gaps.thin_bridges)}")
    out.append("=" * 76)

    if g.god_nodes:
        out.append("")
        out.append("Top hub concepts (god nodes):")
        by_id = {n.id: n for n in g.nodes}
        for nid in g.god_nodes:
            n = by_id.get(nid)
            if n:
                out.append(f"  {n.degree:>3} deg  c{n.community:<3} {n.label}")

    if g.communities:
        out.append("")
        out.append("Communities (size — label):")
        for c in g.communities[:12]:
            out.append(f"  c{c.id:<3} {c.size:>3} concepts  — {c.label}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Extract + write graph.json (LLM calls)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-extract every document (ignore cache)"
    )
    parser.add_argument("--doc", type=str, help="Limit extraction to one doc_hash or id prefix")
    parser.add_argument(
        "--provider",
        type=str,
        default=CONCEPT_GRAPH_LLM_PROVIDER,
        help="LLM provider for extraction (default %(default)s — local, free)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CONCEPT_GRAPH_LLM_MODEL,
        help="LLM model (default %(default)s)",
    )
    args = parser.parse_args()

    client = None
    if args.apply:
        from doc_assistant.llm import ProviderCostError, assert_provider_intent, make_client

        try:
            assert_provider_intent(
                args.provider,
                operation="concept-graph extraction",
                model=args.model,
                scope="every document",
            )
        except ProviderCostError as e:
            print(e)
            return 1
        client = make_client(args.provider, args.model)
        print(f"Extracting concepts with {args.provider}/{args.model} ...")

    result = build_concept_graph(
        apply=args.apply,
        force=args.force,
        client=client,
        doc_filter=args.doc,
        provider=args.provider,
        model=args.model,
    )
    print(_format_report(result))
    if not args.apply:
        print("\nDry run (graph assembled from cache only). Pass --apply to extract + write.")
    else:
        print(f"\nConcept graph written to {CONCEPT_GRAPH_DIR / 'graph.json'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
