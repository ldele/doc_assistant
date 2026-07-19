"""Backfill the ADR-018 ``graph_include`` flag on concepts that have none.

Tag families (ADR-015) and concept-graph nodes are the same ``Concept`` rows, so the graph
scopes itself with an opt-in flag instead of the two features fighting over one vocabulary.
This runner applies that rule retroactively: ``source == "manual"`` (deliberate glossary
curation) opts **in**, everything else — promoted keyword candidates, keyword families — opts
**out**.

Only rows whose flag ``IS NULL`` are touched, so a later override survives and re-running is a
no-op. Dry-run by default: this decides what the graph is built over.

The flag takes effect on the next skeleton build. Rebuild with ``--apply --enrich`` together —
``--apply`` alone rebuilds the edges with no Node-B stance annotations, wiping them (see
``tests/eval/baselines/superseded_year_rule_2026-07.md``).

Usage:
    python -m scripts.backfill_graph_include              # dry run — report the split
    python -m scripts.backfill_graph_include --apply      # write it
    python -m scripts.build_concept_skeleton --apply --enrich   # then rebuild
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.knowledge.concept_skeleton import backfill_graph_include

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the flags (default: dry run — report only)",
    )
    args = parser.parse_args()

    n_in, n_out = backfill_graph_include(apply=args.apply)
    verb = "Set" if args.apply else "Would set"
    total = n_in + n_out
    if total == 0:
        print("Nothing to backfill — every concept already has a graph_include value.")
        return 0
    print(f"{verb} graph_include on {total} concept(s) with no value:")
    print(f"  include (source='manual'): {n_in}")
    print(f"  exclude (other sources):   {n_out}")
    if not args.apply:
        print("\nDry run — re-run with --apply to write.")
    else:
        print("\nRebuild the skeleton for this to take effect:")
        print("  python -m scripts.build_concept_skeleton --apply --enrich")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
