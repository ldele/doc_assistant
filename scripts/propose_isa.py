"""Propose `is_a` edges between concepts from their labels' heads (ROADMAP 51, ADR-028 D2/D8).

Deterministic and free: no model, no network. A label whose tokens end with another concept's
whole label is proposed as narrower than it (`beta oscillations` is_a `oscillations`). Rows are
written with `origin="proposed"` — the taxonomy view accepts or rejects them; nothing here is
curated fact, and a curated edge is never overwritten. Why a shared *prefix* and an alias match
are both refused: `doc_assistant/knowledge/isa_propose.py`.

Usage:
    python -m scripts.propose_isa                  # dry-run: the candidate list, nothing written
    python -m scripts.propose_isa --apply          # write them as proposals
    python -m scripts.propose_isa --graph-only     # only the graph vocabulary (small here)
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant import config
from doc_assistant.knowledge.isa_propose import IsaProposeResult, run_propose_isa

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(run: IsaProposeResult) -> str:
    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Concepts read:                  {run.n_concepts}")
    out.append(f"Candidates (shared head):       {len(run.candidates)}")
    if run.applied:
        out.append(f"Proposed edges written:         {run.n_written}")
        out.append(f"Refused by the write seam:      {run.n_skipped}")
    else:
        out.append("Nothing written — dry run (--apply writes them as proposals)")
    out.append("=" * 76)
    if run.candidates:
        out.append("")
        out.append(f"{'narrower':<32} {'broader':<24} {'shared head':<20}")
        out.append("-" * 76)
        for c in run.candidates:
            out.append(f"{c.narrow_label[:31]:<32} {c.broad_label[:23]:<24} {c.head[:19]:<20}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the candidates as origin='proposed' is_a edges (default: dry-run)",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Restrict to the graph vocabulary (graph_include) — 13 of 357 concepts here, so the "
        "shared heads that make a spine are mostly outside it",
    )
    args = parser.parse_args()

    from doc_assistant.logging_config import configure_logging

    configure_logging(json=config.LOG_JSON, level=config.LOG_LEVEL)

    run = run_propose_isa(apply=args.apply, graph_only=args.graph_only)
    print(_format_report(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
