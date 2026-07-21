"""Compute the gap-detection sidecar — Tier 1, the Tier-2a deterministic floor, and
(optionally) the Tier-2a stochastic ceiling (ADR-004 / docs/specs/feature-gap-detection.md).

Runs the deterministic detectors over the concept skeleton (isolated / single-source
/ thin-bridge / under-connected concepts) plus the deterministic floor over persisted
answer-claim data (`unsupported`-marked claims aggregated onto the curated concept(s)
they mention), and writes a `gaps` row per finding. Free + read-only: no LLM call,
never mutates the chunk store or the curated vocabulary. Enrichment-Layer Pattern —
idempotent, regenerable: re-running replaces the *deterministic* rows from the
current skeleton + claims; stochastic rows persist their `status` across the rebuild.

`--suggest` additionally routes the `under_connected` gaps through the Tier-2a
stochastic ceiling (`gap_suggest.suggest_for_thin`, SPRINT-005) — one quarantined LLM
call per concept, suggestion-only, never written as fact. Provider-isolated like Node B
(`build_concept_skeleton --enrich`): defaults to LOCAL Ollama (`GAP_SUGGEST_LLM_PROVIDER`/
`_MODEL`), `--apply` routes through `llm.assert_provider_intent` *before* any client is
constructed, and a dry run (`--suggest` without `--apply`) makes zero LLM calls.

Run `build_concept_skeleton --apply` first (this projects over
`data/skeleton/skeleton.json`).

`--min-degree` is corpus-derived, not a guessed absolute — see
`tests/eval/baselines/gap_min_degree_2026-07.md` for how the default below was set.

Usage:
    python -m scripts.build_gaps                          # dry-run: compute + report
    python -m scripts.build_gaps --apply                   # write the gaps table
    python -m scripts.build_gaps --apply --min-degree 4
    python -m scripts.build_gaps --apply --suggest         # + the stochastic ceiling (local)
    python -m scripts.build_gaps --apply --suggest --provider anthropic --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant import config
from doc_assistant.knowledge.gaps import GapsResult, build_gaps
from doc_assistant.llm import assert_provider_intent, make_client

# Q1 (first quartile) of the degree distribution on the validated 2026-07 corpus
# baseline (26 concepts, degree range 1-20, Q1=3.0) — see the baseline note. A
# concept below this many edges routes into the Tier-2a stochastic ceiling (--suggest)
# as a "thin" candidate.
_DEFAULT_MIN_DEGREE = 3

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(result: GapsResult) -> str:
    by_kind: dict[str, int] = {}
    for g in result.gaps:
        by_kind[g.kind] = by_kind.get(g.kind, 0) + 1

    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Graph version:             {result.graph_version}")
    out.append(f"Tier-1 gaps:               {result.n_t1}")
    out.append(f"Tier-2a floor gaps:        {result.n_t2a}")
    out.append(f"Tier-2a suggested (stoch): {result.n_suggested}")
    if result.n_reconciled:
        out.append(
            f"Orphaned stochastic reaped:{result.n_reconciled:>3}  (concept left the graph)"
        )
    out.append(f"Total gaps:                {len(result.gaps)}")
    if result.applied:
        out.append(f"Rows written:              {len(result.gaps) + result.n_suggested}")
    out.append("=" * 76)
    if by_kind:
        out.append("")
        out.append(f"{'kind':<18} {'count':>6}")
        out.append("-" * 76)
        for kind, count in sorted(by_kind.items()):
            out.append(f"{kind:<18} {count:>6}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the gaps sidecar table")
    parser.add_argument(
        "--min-degree",
        type=int,
        default=_DEFAULT_MIN_DEGREE,
        help="Degree floor for `under_connected` (default %(default)s; corpus-derived — "
        "see tests/eval/baselines/gap_min_degree_2026-07.md)",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Tier-2a stochastic ceiling: one quarantined LLM call per under-connected "
        "concept, suggestion-only (never written as fact). Zero LLM calls without --apply.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="--suggest LLM provider (default GAP_SUGGEST_LLM_PROVIDER=ollama; local/free)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="--suggest LLM model (default GAP_SUGGEST_LLM_MODEL=llama3.1:8b)",
    )
    args = parser.parse_args()

    provider = args.provider or config.GAP_SUGGEST_LLM_PROVIDER
    model = args.model or config.GAP_SUGGEST_LLM_MODEL
    client = None
    if args.suggest:
        assert_provider_intent(
            provider,
            operation="gap-detection Tier-2a ceiling (--suggest)",
            apply=args.apply,
            model=model,
            scope="gaps sidecar (under_connected concepts)",
        )
        if args.apply:
            client = make_client(provider, model)

    try:
        result = build_gaps(
            apply=args.apply,
            min_degree=args.min_degree,
            suggest=args.suggest,
            client=client,
        )
    except FileNotFoundError as e:
        print(e)
        return 1

    print(_format_report(result))
    if not args.apply:
        print("\nDry run. Pass --apply to write the gaps sidecar.")
    else:
        print("\ngaps sidecar written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
