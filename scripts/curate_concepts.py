"""Curate the auto-seeded concept vocabulary — prune non-concepts + merge near-duplicates.

The pruning counterpart to ``seed_concepts --promote-all``: a broad promoted vocabulary carries
extraction noise (DOI/date/license fragments, single-char tokens, author names, sentence
fragments). This runs three cheapest-first stages over ``doc_assistant.concept_curation`` —
deterministic artifact filter, optional local-LLM classification, optional embedding near-dup
merge — and rewrites the curated ``concepts`` / ``concept_aliases`` tables. Dry-run by default;
mutates only on ``--apply``. Re-run ``build_concept_skeleton --apply`` afterwards to regenerate the
derived skeleton over the cleaned vocabulary.

The LLM stage is provider-isolated exactly like Node B: local Ollama by default
(``CONCEPT_SKELETON_LLM_PROVIDER``), routed through ``assert_provider_intent``.

Usage::

    python -m scripts.curate_concepts                        # dry-run: artifact filter only
    python -m scripts.curate_concepts --llm                  # + Ollama noise classification
    python -m scripts.curate_concepts --llm --dedup          # + near-duplicate merge (preview)
    python -m scripts.curate_concepts --llm --dedup --apply  # execute the plan
"""

from __future__ import annotations

import argparse
import sys

from doc_assistant import config
from doc_assistant.concept_curation import (
    CurationPlan,
    apply_merges,
    classify_noise,
    dedup_pairs,
    doc_counts,
    is_artifact,
    load_concepts,
    plan_merges,
    remove_concepts,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _build_plan(args: argparse.Namespace) -> CurationPlan:
    concepts = load_concepts()
    plan = CurationPlan()
    plan.artifacts = [(cid, label) for cid, label in concepts if is_artifact(label)]
    artifact_ids = {cid for cid, _ in plan.artifacts}
    survivors = [(cid, label) for cid, label in concepts if cid not in artifact_ids]

    if args.llm:
        from doc_assistant.llm import assert_provider_intent, make_client

        provider = args.provider or config.CONCEPT_SKELETON_LLM_PROVIDER
        model = args.model or config.CONCEPT_SKELETON_LLM_MODEL
        assert_provider_intent(
            provider, operation="concept curation (LLM classify)", apply=args.apply, model=model
        )
        client = make_client(provider, model)
        print(f"LLM classifying {len(survivors)} survivor(s) with {provider}:{model} ...")
        plan.llm_noise = classify_noise(survivors, client)
        plan.n_calls = -1  # sentinel: classification ran (exact count logged)

    noise_ids = {cid for cid, _ in plan.llm_noise}
    remaining = [(cid, label) for cid, label in survivors if cid not in noise_ids]

    if args.dedup:
        print(f"Embedding {len(remaining)} concept(s) for near-duplicate merge ...")
        pairs = dedup_pairs(remaining, threshold=args.threshold, model=args.embed_model)
        label_by_id = {cid: label for cid, label in remaining}
        plan.merges = plan_merges(pairs, doc_counts(), label_by_id)

    return plan


def _report(plan: CurationPlan, total: int) -> None:
    kept = total - len(plan.remove_ids) - len(plan.merges)
    print("\n" + "=" * 72)
    print(f"Vocabulary curation plan  ({total} concepts)")
    print(f"  Artifact-filtered   : {len(plan.artifacts)}")
    print(f"  LLM-flagged noise   : {len(plan.llm_noise)}")
    print(f"  Near-dup merges     : {len(plan.merges)}")
    print(f"  -> concepts kept    : {kept}")
    print("=" * 72)
    for title, items in (("Artifacts", plan.artifacts), ("LLM noise", plan.llm_noise)):
        if items:
            sample = ", ".join(label for _, label in items[:20])
            print(f"\n{title} ({len(items)}): {sample}{' ...' if len(items) > 20 else ''}")
    if plan.merges:
        print(f"\nMerges ({len(plan.merges)}):")
        for m in plan.merges[:20]:
            print(f"  '{m.drop_label}' -> '{m.keep_label}'")
        if len(plan.merges) > 20:
            print(f"  ... (+{len(plan.merges) - 20} more)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Execute the plan (default: dry-run)")
    parser.add_argument("--llm", action="store_true", help="Add the Ollama classification stage")
    parser.add_argument("--dedup", action="store_true", help="Add the near-duplicate merge stage")
    parser.add_argument("--provider", default=None, help="LLM provider (default ollama)")
    parser.add_argument("--model", default=None, help="LLM model (default llama3.1:8b)")
    parser.add_argument("--embed-model", default=None, help="Embedding model for --dedup")
    parser.add_argument("--threshold", type=float, default=0.9, help="Cosine threshold (dedup)")
    args = parser.parse_args()

    from doc_assistant.logging_config import configure_logging

    configure_logging(json=config.LOG_JSON, level=config.LOG_LEVEL)

    total = len(load_concepts())
    plan = _build_plan(args)
    _report(plan, total)

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to curate.")
        return 0

    removed = remove_concepts(plan.remove_ids)
    merged = apply_merges(plan.merges)
    print(f"\nApplied: removed {removed} concept(s), merged {merged}.")
    print("Next: `build_concept_skeleton --apply` to rebuild the skeleton over the clean vocab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
