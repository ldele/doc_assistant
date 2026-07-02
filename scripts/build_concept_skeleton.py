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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
