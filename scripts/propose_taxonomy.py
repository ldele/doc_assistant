"""Auto-propose taxonomy placements for unplaced concepts + unclassified documents (ADR-028 D8).

One quarantined local-LLM pass per item, two-stage (division -> group within it), written as
`origin="proposed"` links the user accepts or deletes — never as curated fact. Spec:
`docs/specs/feature-taxonomy-auto-propose.md`.

Provider-isolated like Node B / gap-suggest: defaults to LOCAL Ollama
(`TAXONOMY_PROPOSE_LLM_PROVIDER`/`_MODEL`, KI-4), `--apply` routes through
`llm.assert_provider_intent` *before* any client is constructed, and a dry run makes **zero** LLM
calls (the guard no-ops on a dry run, so "no --apply" has to mean "no spend").

Run `seed_taxonomy --apply` first — with no field nodes there is nothing to place into.

Usage:
    python -m scripts.propose_taxonomy                       # dry-run: scope + call budget only
    python -m scripts.propose_taxonomy --apply               # propose (local Ollama, $0)
    python -m scripts.propose_taxonomy --apply --limit 10    # bounded first pass
    python -m scripts.propose_taxonomy --apply --documents-only
    python -m scripts.propose_taxonomy --apply --all-concepts  # incl. non-graph keyword concepts
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant import config
from doc_assistant.knowledge.taxonomy_propose import ProposeRunResult, run_propose
from doc_assistant.llm import assert_provider_intent, make_client

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _format_report(run: ProposeRunResult, *, provider: str, model: str) -> str:
    result = run.result
    out: list[str] = []
    out.append("=" * 76)
    # Name the instrument in the report: a "local, $0" claim should be readable off the run
    # itself, not inferred from the defaults (KI-4).
    out.append(f"Provider / model:               {provider} / {model}")
    out.append(f"Unplaced concepts (in scope):   {run.n_unplaced_concepts}")
    if run.n_concepts_out_of_scope:
        out.append(
            f"  + outside graph vocabulary:   {run.n_concepts_out_of_scope}"
            "  (--all-concepts to include)"
        )
    out.append(f"Unclassified documents:         {run.n_unclassified_documents}")
    out.append(f"Items this run:                 {result.n_items}")
    if run.n_truncated:
        out.append(f"  dropped by --limit:           {run.n_truncated}")
    if run.applied:
        out.append(f"LLM calls made:                 {result.n_calls}")
        out.append(f"Proposals:                      {len(result.proposals)}")
        out.append(f"  placed at division only:      {result.n_division_only}")
        out.append(f"Abstained / no answer:          {result.n_abstained}")
        out.append(f"Rows written (in_field):        {run.n_hierarchy_written}")
        out.append(f"Rows written (document_field):  {run.n_document_fields_written}")
    else:
        out.append(
            f"Call budget if applied:         <= {result.n_items * 2}  (two stages per item)"
        )
    out.append("=" * 76)

    if result.proposals:
        out.append("")
        out.append(f"{'item':<34} {'kind':<9} {'placement':<34} {'conf':>5}")
        out.append("-" * 76)
        for p in result.proposals:
            placement = p.field_label if p.field_id != p.division_id else f"{p.field_label} (div)"
            confidence = "  -  " if p.confidence is None else f"{p.confidence:>5.2f}"
            out.append(
                f"{p.item_label[:33]:<34} {p.item_kind:<9} {placement[:33]:<34} {confidence}"
            )
    if result.misses:
        out.append("")
        out.append("No placement proposed (reported, not silently dropped):")
        for label, why in result.misses:
            out.append(f"  - {label[:56]:<58} {why}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run the LLM pass and write origin='proposed' links (default: dry-run, zero calls)",
    )
    parser.add_argument(
        "--concepts-only", action="store_true", help="Propose for concepts only, not documents"
    )
    parser.add_argument(
        "--documents-only", action="store_true", help="Propose for documents only, not concepts"
    )
    parser.add_argument(
        "--all-concepts",
        action="store_true",
        help="Include unplaced concepts outside the graph vocabulary (graph_include false) — "
        "on a keyword-flooded corpus this is a much larger run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the items this run (the drop count is printed)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider (default TAXONOMY_PROPOSE_LLM_PROVIDER=ollama; local/free)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model (default TAXONOMY_PROPOSE_LLM_MODEL=qwen3.5:9b — RG-015 measured)",
    )
    args = parser.parse_args()

    if args.concepts_only and args.documents_only:
        parser.error("--concepts-only and --documents-only are mutually exclusive")

    from doc_assistant.logging_config import configure_logging

    configure_logging(json=config.LOG_JSON, level=config.LOG_LEVEL)

    provider = args.provider or config.TAXONOMY_PROPOSE_LLM_PROVIDER
    model = args.model or config.TAXONOMY_PROPOSE_LLM_MODEL
    client = None
    if args.apply:
        assert_provider_intent(
            provider,
            operation="taxonomy auto-propose (--apply)",
            apply=True,
            model=model,
            scope="unplaced concepts + unclassified documents (two calls each)",
        )
        client = make_client(provider, model)

    run = run_propose(
        apply=args.apply,
        client=client,
        include_concepts=not args.documents_only,
        include_documents=not args.concepts_only,
        all_concepts=args.all_concepts,
        limit=args.limit,
    )

    print(_format_report(run, provider=provider, model=model))
    if not args.apply:
        print("\nDry run — no LLM calls, nothing written. Pass --apply to propose.")
    else:
        print("\nProposals written as origin='proposed' — accept or delete them in the taxonomy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
