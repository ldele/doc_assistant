"""Build the deterministic concept skeleton (Node A) — curated vocabulary, zero LLM.

Computes concept presence (curated label/alias string match), the chunk-level
co-occurrence edge skeleton, and citation/similarity provenance over the library's
existing zero-LLM graphs, then Louvain communities — all deterministic and free. Writes
the ``concept_edges`` / ``concept_presence`` sidecar tables + ``skeleton.json`` (Enrichment-
Layer Pattern: regenerable, never mutates the chunk store). Idempotent + byte-stable.

The curated vocabulary must exist first (``python -m scripts.seed_concepts``) and
``DocSimilarity`` must be populated (``python -m scripts.compute_doc_vectors --apply``) or
similarity-provenance is silently absent. The LLM relation/stance pass (Node B) is a
separate, deferred runner (PR-B) — this Node-A runner makes no LLM calls.

Usage:
    python -m scripts.build_concept_skeleton                   # dry run (no writes)
    python -m scripts.build_concept_skeleton --apply           # write sidecar + skeleton.json
    python -m scripts.build_concept_skeleton --apply --force   # force a full rebuild
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from doc_assistant.concept_skeleton import SkeletonResult, build_concept_skeleton
from doc_assistant.config import CONCEPT_SKELETON_DIR

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(result: SkeletonResult) -> str:
    sk = result.skeleton
    prov = result.provenance_counts
    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Documents (with presence):  {result.n_documents}")
    out.append(f"Concept nodes:              {result.n_concepts}")
    out.append(f"Skeleton edges:             {result.n_edges}")
    out.append(
        f"  by provenance: cooccurrence {prov.get('cooccurrence', 0)} · "
        f"citation {prov.get('citation', 0)} · similarity {prov.get('similarity', 0)} · "
        f"llm_relation {prov.get('llm_relation', 0)}"
    )
    out.append(f"Communities:                {len(sk.communities)}")
    out.append(f"Isolated concepts:          {result.n_isolated}")
    out.append("=" * 76)
    if sk.communities:
        out.append("")
        out.append("Communities (size — label):")
        for c in sk.communities[:12]:
            out.append(f"  c{c.id:<3} {c.size:>3} concepts  — {c.label}")
    return "\n".join(out)


def _format_node_b_report(skeleton: Any, contested: list[Any]) -> str:
    meta = skeleton.meta
    label = {n.id: n.label for n in skeleton.nodes}
    annotated = [e for e in skeleton.edges if e.relation]
    n_stances = sum(len(e.stance_by_doc) for e in skeleton.edges)
    out = [
        "",
        "=" * 76,
        "Node B — LLM relation/stance enrichment",
        f"LLM calls:                  {meta.get('node_b_calls', 0)}",
        f"Edges annotated:            {len(annotated)}",
        f"Stance assertions:          {n_stances}",
        f"Contested edges:            {len(contested)}",
        "=" * 76,
    ]
    if annotated:
        out.append("")
        out.append("Annotated edges (source —[relation]→ target : stances):")
        for e in annotated[:15]:
            stances = ", ".join(pol for _, pol in e.stance_by_doc)
            src = label.get(e.source_concept_id, e.source_concept_id)
            tgt = label.get(e.target_concept_id, e.target_concept_id)
            out.append(f"  {src} —[{e.relation}]→ {tgt}  ({stances})")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write concept_edges/concept_presence + skeleton.json (no LLM)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a full rebuild of the derived tables",
    )
    parser.add_argument(
        "--min-cooccurrence",
        type=int,
        default=None,
        help="Override CONCEPT_SKELETON_MIN_COOCCURRENCE for this run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the Louvain seed for this run",
    )
    parser.add_argument(
        "--presence-mode",
        choices=("boundary", "substring"),
        default=None,
        help="Presence match: boundary=whole-word (default), substring=raw (A/B lever)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Node B: LLM relation/stance pass over existing edges (needs --apply to run)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Node B LLM provider (default CONCEPT_SKELETON_LLM_PROVIDER=ollama; local/free)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Node B LLM model (default CONCEPT_SKELETON_LLM_MODEL=llama3.1:8b)",
    )
    args = parser.parse_args()

    result = build_concept_skeleton(
        apply=args.apply,
        force=args.force,
        min_cooccurrence=args.min_cooccurrence,
        seed=args.seed,
        presence_mode=args.presence_mode,
    )
    print(_format_report(result))
    if not args.apply:
        print("\nDry run (skeleton computed, nothing written). Pass --apply to write.")
    else:
        print(f"\nConcept skeleton written to {CONCEPT_SKELETON_DIR / 'skeleton.json'}.")

    if args.enrich:
        return _run_node_b(args, result)
    return 0


def _run_node_b(args: argparse.Namespace, result: SkeletonResult) -> int:
    """Node B — confined LLM relation/stance enrichment over the just-built Node A skeleton.

    Provider-isolated (local Ollama default) + apply-gated: the LLM is called only on
    ``--apply``, so a dry run costs nothing and a paid provider always trips the cost guard.
    """
    from doc_assistant import config
    from doc_assistant.concept_skeleton import contested_edges, write_skeleton
    from doc_assistant.concept_skeleton_enrich import (
        annotate_relations,
        load_presence_rows,
        present_by_doc,
    )
    from doc_assistant.llm import assert_provider_intent, make_client

    provider = args.provider or config.CONCEPT_SKELETON_LLM_PROVIDER
    model = args.model or config.CONCEPT_SKELETON_LLM_MODEL
    assert_provider_intent(
        provider,
        operation="concept-skeleton Node B (LLM relation/stance)",
        apply=args.apply,
        model=model,
        scope=str(CONCEPT_SKELETON_DIR / "skeleton.json"),
    )
    if not args.apply:
        print(
            "\nNode B is apply-gated (zero LLM calls in a dry run). Re-run with "
            f"--apply --enrich --provider {provider} to annotate."
        )
        return 0

    presences = load_presence_rows()
    pbd = present_by_doc(presences)
    if not pbd:
        print("\nNode B: no presence rows found — run Node A with --apply first.")
        return 1

    client = make_client(provider, model)
    print(f"\nNode B: annotating with {provider}:{model} over {len(pbd)} document(s)...")
    enriched = annotate_relations(result.skeleton, pbd, client)
    write_skeleton(enriched, presences)
    print(_format_node_b_report(enriched, contested_edges(list(enriched.edges))))
    print(f"\nEnriched skeleton written to {CONCEPT_SKELETON_DIR / 'skeleton.json'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
