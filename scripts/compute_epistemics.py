"""Compute the knowledge-currency / claim-corroboration sidecar (Feature 7d).

Projects the concept skeleton's per-node corroboration weights onto the baseline
chunks (structural attribution — a concept is "in" a chunk if its label occurs in
the text), and writes a `chunk_epistemics` row per chunk that carries a weighted
claim. The row records contested / superseded-trending claim counts so the evidence
layer can surface them at answer time.

Free + read-only: no LLM call, never mutates the chunk store. Enrichment-Layer
Pattern — idempotent, regenerable: re-running replaces the table from the current
skeleton.

Run `build_concept_skeleton --apply` first (this projects over
`data/skeleton/skeleton.json`). The skeleton carries no publication years, so
`superseded_trend` never appears today (contested / stable / unique only) — see
`concept_skeleton.node_weights_for_epistemics`.

Usage:
    python -m scripts.compute_epistemics              # dry-run: compute + report
    python -m scripts.compute_epistemics --apply      # write the chunk_epistemics table
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant.knowledge.epistemics import EpistemicsResult, build_epistemics

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(result: EpistemicsResult) -> str:
    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Graph version:             {result.graph_version}")
    out.append(f"Concept nodes weighted:    {result.n_nodes}")
    out.append(f"  Contested nodes:          {result.n_contested_nodes}")
    out.append(f"  Superseded-trend nodes:   {result.n_superseded_nodes}")
    out.append(f"Chunks with a claim:       {len(result.rows)}")
    out.append(f"  Chunks marked:            {result.n_chunks_marked}")
    if result.applied:
        out.append(f"Rows written:              {len(result.rows)}")
    out.append("=" * 76)
    marked = [r for r in result.rows if r.markers]
    if marked:
        out.append("")
        out.append(f"{'chunk_key':<28} {'claims':>6} {'markers'}")
        out.append("-" * 76)
        for r in marked[:40]:
            out.append(f"{r.chunk_key:<28} {r.n_claims:>6} {', '.join(r.markers)}")
        if len(marked) > 40:
            out.append(f"... and {len(marked) - 40} more marked chunk(s)")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Write the chunk_epistemics sidecar table"
    )
    args = parser.parse_args()

    try:
        result = build_epistemics(apply=args.apply)
    except FileNotFoundError as e:
        print(e)
        return 1

    print(_format_report(result))
    if not args.apply:
        print("\nDry run. Pass --apply to write the chunk_epistemics sidecar.")
    else:
        print("\nchunk_epistemics sidecar written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
