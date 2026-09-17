"""Curate the auto-seeded concept vocabulary — demote non-concepts + merge near-duplicates.

The curation counterpart to ``seed_concepts --promote-all``: a broad promoted vocabulary carries
extraction noise (DOI/date/license fragments, single-char tokens, author names, sentence
fragments). This runs three cheapest-first stages over
``doc_assistant.knowledge.concept_curation`` —
deterministic artifact filter, optional local-LLM classification, optional embedding near-dup
merge. Artifact + noise verdicts **demote** the concept out of the graph vocabulary
(``graph_include=False``, ADR-018 — the row + its keyword family survive, so a misclassified
specialist term is recoverable); only near-dup merges fold + drop a row. Dry-run by default;
mutates only on ``--apply``. Re-run ``build_concept_skeleton --apply`` afterwards to regenerate the
derived skeleton over the cleaned vocabulary.

The LLM stage is provider-isolated exactly like Node B: local Ollama by default
(``CONCEPT_SKELETON_LLM_PROVIDER``), routed through ``assert_provider_intent``.

Usage::

    python -m scripts.curate_concepts                        # dry-run: artifact filter only
    python -m scripts.curate_concepts --llm                  # + Ollama noise classification
    python -m scripts.curate_concepts --llm --dedup          # + near-duplicate merge (preview)
    python -m scripts.curate_concepts --llm --dedup --apply  # execute the plan
    python -m scripts.curate_concepts --merges               # list recorded merges
    python -m scripts.curate_concepts --undo-merge ID        # preview splitting one back
    python -m scripts.curate_concepts --undo-merge ID --apply

A merge moves the dropped concept's surface forms, taxonomy placements and gap triage to the
survivor before deleting it, and records itself so ``--undo-merge`` can split the two again
(ROADMAP 53). The dedup threshold and embedder default to the same values as
``suggest_concepts --near``, which previews exactly these pairs.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from doc_assistant import config
from doc_assistant.knowledge.concept_curation import (
    CurationPlan,
    apply_plan,
    classify_noise,
    dedup_pairs,
    doc_counts,
    is_artifact,
    load_concepts,
    plan_merges,
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
    in_graph = total - len(plan.demote_ids) - len(plan.merges)
    print("\n" + "=" * 72)
    print(f"Vocabulary curation plan  ({total} concepts)")
    print(f"  Artifact-filtered   : {len(plan.artifacts)}")
    print(f"  LLM-flagged noise   : {len(plan.llm_noise)}")
    print(f"  -> demoted from graph: {len(plan.demote_ids)}  (rows + keyword families kept)")
    print(f"  Near-dup merges     : {len(plan.merges)}")
    if plan.merges:
        # Pairs chain: a merge group can absorb concepts no single pair would (baseline
        # concept_merge_cosine_2026-09-17 — SPECTER2 at 0.85 made one group of 354).
        sizes = Counter(m.keep_label for m in plan.merges)
        keep, absorbed = sizes.most_common(1)[0]
        print(f"  -> largest group     : {absorbed + 1} concepts into '{keep}'")
    print(f"  -> graph vocabulary  : {in_graph}")
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


def _list_merges() -> int:
    from doc_assistant.knowledge.concept_merge import list_merges

    merges = list_merges()
    if not merges:
        print("No merges recorded.")
        return 0
    for m in merges:
        state = f"undone {m.undone_at:%Y-%m-%d}" if m.undone_at else "active"
        print(
            f"{m.id}  {m.merged_at:%Y-%m-%d}  '{m.drop_label}' -> '{m.keep_label}'  "
            f"({m.n_aliases_added} alias(es), {m.n_placements} placement(s); {state})"
        )
    return 0


def _undo(merge_id: str, *, apply: bool) -> int:
    from doc_assistant.knowledge.concept_merge import MergeUndoError, list_merges, undo_merge

    match = next((m for m in list_merges() if m.id == merge_id), None)
    if match is None:
        print(f"No merge is recorded with id {merge_id}.")
        return 1
    print(f"Split '{match.drop_label}' back out of '{match.keep_label}'.")
    if not apply:
        print("Dry run — nothing written. Re-run with --apply to undo the merge.")
        return 0
    try:
        undo_merge(merge_id)
    except MergeUndoError as e:
        print(f"Not undone: {e}")
        return 1
    print("Undone. Next: `build_concept_skeleton --apply` to rebuild the skeleton.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Execute the plan (default: dry-run)")
    parser.add_argument("--merges", action="store_true", help="List recorded merges and exit")
    parser.add_argument(
        "--undo-merge", default=None, metavar="ID", help="Split a recorded merge back (--apply)"
    )
    parser.add_argument("--llm", action="store_true", help="Add the Ollama classification stage")
    parser.add_argument("--dedup", action="store_true", help="Add the near-duplicate merge stage")
    parser.add_argument("--provider", default=None, help="LLM provider (default ollama)")
    parser.add_argument("--model", default=None, help="LLM model (default llama3.1:8b)")
    # One definition of "the same concept": the preview (`suggest_concepts --near`) has the same
    # defaults, so what it shows is what --dedup merges (ROADMAP 53).
    parser.add_argument(
        "--embed-model", default=config.CONCEPT_MERGE_MODEL, help="Embedding model for --dedup"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.CONCEPT_MERGE_COSINE,
        help="Cosine threshold (--dedup)",
    )
    args = parser.parse_args()

    from doc_assistant.logging_config import configure_logging

    configure_logging(json=config.LOG_JSON, level=config.LOG_LEVEL)

    if args.merges:
        return _list_merges()
    if args.undo_merge:
        return _undo(args.undo_merge, apply=args.apply)

    total = len(load_concepts())
    plan = _build_plan(args)
    _report(plan, total)

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to curate.")
        return 0

    demoted, outcome = apply_plan(plan)
    print(f"\nApplied: demoted {demoted} concept(s) from the graph, merged {outcome.n_merged}.")
    for skipped, reason in outcome.skipped:
        print(f"  not merged: '{skipped.drop_label}' -> '{skipped.keep_label}' — {reason}")
    if outcome.merged:
        print("Each merge is recorded: `--merges` lists them, `--undo-merge ID` splits one back.")
    print(
        "Demote keeps the row + its keyword family (ADR-018) — reversible via set_graph_include."
    )
    print("Next: `build_concept_skeleton --apply` to rebuild the skeleton over the clean vocab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
