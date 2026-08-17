"""Run the eval harness against the doc_assistant RAG pipeline.

Loads cases from ``tests/eval/cases.yaml``, wires the adapter +
scorer mix, runs the suite, persists to ``data/eval.duckdb``, and
prints the summary table.

Default scorer mix is the **free** subset (no API calls):
``contains_all`` + ``citation_overlap``. Pass ``--with-embedding`` or
``--with-llm-judge`` to opt into the paid scorers.

**A free scorer mix is not a free run.** Every case generates an answer, and the generator is
whatever ``.env`` resolves — which ships ``LLM_PROVIDER=anthropic``, so a bare invocation bills
the API (KI-4 wearing the harness's clothes: 5 trials over the private 35 cost ~839 K input
tokens on 2026-08-15 while nobody had asked for a paid scorer). ``--provider`` / ``--model``
name the generator explicitly, and a paid one now trips the same cost banner every enrichment
runner uses.

Usage::

    python -m scripts.run_eval                       # free scorers only
    python -m scripts.run_eval --with-embedding      # + embedding similarity
    python -m scripts.run_eval --with-llm-judge      # + LLM judge (costs $$$)
    python -m scripts.run_eval --cases custom.yaml   # custom case file
    python -m scripts.run_eval --provider ollama --model llama3.1:8b   # local, $0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from doc_assistant import sparse_index
from doc_assistant.config import (
    ANTHROPIC_API_KEY,
    JUDGE_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    PROJECT_ROOT,
)
from doc_assistant.embeddings import get_active_model_name
from doc_assistant.eval import (
    CitationOverlapScorer,
    ContainsAllScorer,
    EmbeddingSimilarityScorer,
    LLMJudgeScorer,
    Runner,
    Scorer,
    Store,
    load_cases_yaml,
)
from doc_assistant.eval.adapters import embedding_callable, rag_pipeline_adapter
from doc_assistant.eval.report import format_aggregate, format_flaky_cases, format_run_summary
from doc_assistant.eval.run_settings import run_defining_settings

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_CASES = PROJECT_ROOT / "tests" / "eval" / "cases.yaml"
DEFAULT_DB = PROJECT_ROOT / "data" / "eval.duckdb"


def resolve_generator(provider: str | None, model: str | None) -> tuple[str, str]:
    """The generator this run will use, from the flags plus the configured defaults.

    Either flag alone falls back to its config constant, which is the idiom every enrichment
    runner uses. The one combination that is refused is ``--provider`` *without* ``--model`` when
    the provider is being changed: the inherited model name belongs to the provider being left
    behind (``.env``'s ``anthropic`` pairs with ``claude-haiku-4-5-…``), so honouring it would
    hand Ollama a model it has never heard of and fail once per case, deep into a run.

    Raises ``ValueError`` with the command that would have been meant.
    """
    resolved_provider = provider or LLM_PROVIDER
    if model:
        return resolved_provider, model
    if resolved_provider != LLM_PROVIDER:
        raise ValueError(
            f"--provider {resolved_provider} changes the generator away from the configured "
            f"default ({LLM_PROVIDER}), but the configured model ({LLM_MODEL}) belongs to "
            f"{LLM_PROVIDER}. Name the model too, e.g. "
            f"--provider {resolved_provider} --model <model>."
        )
    return resolved_provider, LLM_MODEL


def index_composition(doc_hashes: set[str] | None) -> dict[str, Any]:
    """The corpus the run retrieved over, as run-record keys (RG-021).

    ``None`` in — the pipeline could not tell what the index held — records ``None`` for both
    keys rather than omitting them: a run that does not know its corpus should say so where a
    reader looks, not look like a run from before the keys existed. An empty corpus records
    ``0`` and the digest of the empty set, because that is a real composition.
    """
    if doc_hashes is None:
        return {"index_doc_count": None, "index_doc_digest": None}
    return {
        "index_doc_count": len(doc_hashes),
        "index_doc_digest": sparse_index.doc_set_digest(doc_hashes),
    }


def _build_scorers(
    pipeline: object,
    *,
    with_embedding: bool,
    with_llm_judge: bool,
) -> list[Scorer]:
    scorers: list[Scorer] = [ContainsAllScorer(), CitationOverlapScorer()]
    if with_embedding:
        scorers.append(EmbeddingSimilarityScorer(embedding_callable(pipeline)))  # type: ignore[arg-type]
    if with_llm_judge:
        if JUDGE_PROVIDER.lower() == "anthropic" and not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "--with-llm-judge with JUDGE_PROVIDER=anthropic requires "
                "ANTHROPIC_API_KEY in the env (or set JUDGE_PROVIDER=ollama)"
            )
        from doc_assistant.llm import get_judge_client

        scorers.append(LLMJudgeScorer(get_judge_client()))
    return scorers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=str, default=str(DEFAULT_CASES), help="Path to cases YAML")
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB), help="DuckDB path for persistence"
    )
    parser.add_argument(
        "--with-embedding",
        action="store_true",
        help="Add embedding-similarity scorer (uses the active pipeline embedder)",
    )
    parser.add_argument(
        "--with-llm-judge",
        action="store_true",
        help="Add LLM-as-judge scorer (Anthropic API — costs money)",
    )
    parser.add_argument(
        "--note", type=str, default=None, help="Optional note recorded on the run row"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help=(
            "Provider for ANSWER GENERATION (anthropic | ollama). Default: config LLM_PROVIDER, "
            "which .env ships as anthropic — so a bare run bills the API even with free scorers "
            "(KI-4). Changing it requires --model as well."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model for answer generation. Default: config LLM_MODEL (paired with --provider).",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=None,
        help=(
            "Override the ensemble weight on the BM25 (sparse) arm; the vector arm "
            "takes the complement (1 - w). Default: config BM25_WEIGHT (0.4). LOCKED "
            "retrieval setting — sweep with scripts/sweep_bm25_weight.py and change the "
            "default only on an eval win (rigor-gate)."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the eval N times and aggregate mean ± std per scorer. "
            "Each trial is a separate run in DuckDB; the CLI also prints the "
            "aggregate. Use this to measure variance from the answer-generation "
            "LLM (which runs at default temperature). Default: 1."
        ),
    )
    args = parser.parse_args()
    if args.repeat < 1:
        print("--repeat must be >= 1")
        return 1

    cases = load_cases_yaml(args.cases)
    if not cases:
        print(f"No cases found in {args.cases}")
        return 1

    try:
        provider, model = resolve_generator(args.provider, args.model)
    except ValueError as e:
        print(str(e))
        return 1

    print(f"Loaded {len(cases)} cases from {args.cases}")
    print(f"Generator: {provider} / {model}")

    # Before the pipeline loads, not after: the guard's whole value is a Ctrl-C window that opens
    # *before* anything is spent, and loading costs a minute of model weights either way. The
    # judge is named in the scope when it is on, because that is a second, independent spend.
    from doc_assistant.llm import assert_provider_intent

    judge_note = f"; LLM judge on {JUDGE_PROVIDER}" if args.with_llm_judge else ""
    assert_provider_intent(
        provider,
        operation="run_eval (answer generation)",
        model=model,
        scope=(
            f"{len(cases)} cases x {args.repeat} trial(s), one generated answer each{judge_note}"
        ),
    )

    print("Loading RAG pipeline (this can take a minute)...")
    from doc_assistant.pipeline import RAGPipeline

    pipeline = RAGPipeline(bm25_weight=args.bm25_weight)
    if (provider, model) != (pipeline.provider, pipeline.model):
        # ADR-011's live switch, reused: it rebuilds only the thin chat-model wrapper, so the
        # embedder, vector store, keyword index and reranker the run measures are untouched.
        pipeline.set_chat_model(provider, model)
    print(f"BM25 ensemble weight: {pipeline.bm25_weight} (vector {1.0 - pipeline.bm25_weight})")
    sut = rag_pipeline_adapter(pipeline)
    scorers = _build_scorers(
        pipeline,
        with_embedding=args.with_embedding,
        with_llm_judge=args.with_llm_judge,
    )
    scorer_names = ", ".join(s.name for s in scorers)
    print(f"Scorers: {scorer_names}")

    runner = Runner(scorers)

    def _progress(i: int, total: int, case: object) -> None:
        print(f"  [{i + 1:>2}/{total}] {getattr(case, 'id', '?')}")

    # Read once, before the trials: re-reading per trial would record whichever corpus happened
    # to be live at each persist, and a corpus that moves mid-run is a fact about the whole run.
    composition = index_composition(pipeline.indexed_doc_hashes)
    if composition["index_doc_digest"] is None:
        print("Index composition: UNKNOWN (keyword index unavailable - vector-only retrieval)")
    else:
        print(
            f"Index composition: {composition['index_doc_count']} documents, "
            f"digest {composition['index_doc_digest']}"
        )

    # settings_provider: every run records the chunk/retrieval settings that produced it, so a
    # sweep's own output can contradict its note. Without it, KI-41's six identical configs were
    # indistinguishable in the record for two months.
    with Store(args.db, settings_provider=run_defining_settings) as store:
        run_ids: list[str] = []
        for trial in range(args.repeat):
            if args.repeat > 1:
                print(f"\n=== Trial {trial + 1}/{args.repeat} ===")
            print(f"Running {len(cases)} cases...")
            results = runner.run(cases, sut, progress=_progress)  # type: ignore[arg-type]

            trial_note = args.note
            if args.repeat > 1:
                tag = f"[trial {trial + 1}/{args.repeat}]"
                trial_note = f"{args.note} {tag}" if args.note else tag

            run_id = store.persist_run(
                results,
                system_name=f"doc_assistant/{get_active_model_name()}",
                config={
                    "embedding_model": get_active_model_name(),
                    "n_cases": len(cases),
                    "scorers": [s.name for s in scorers],
                    "trial_index": trial,
                    "n_trials": args.repeat,
                    "bm25_weight": pipeline.bm25_weight,
                    # The generator that actually ran. `run_defining_settings` reads the config
                    # constants, which `--provider` deliberately never touches (set_chat_model
                    # assigns no module global) — so without these two keys an overridden run
                    # would record the default it overrode. Explicit keys win over the snapshot.
                    "llm_provider": pipeline.provider,
                    "llm_model": pipeline.model,
                    **composition,
                },
                note=trial_note,
            )
            run_ids.append(run_id)
            print()
            print(format_run_summary(store, run_id))
            print(f"Trial run id: {run_id}")

        print()
        if args.repeat > 1:
            print(
                format_aggregate(
                    store,
                    run_ids,
                    label=f"Aggregate ({get_active_model_name()}, n={args.repeat})",
                )
            )
            print()
            print(format_flaky_cases(store.flaky_cases(run_ids)))
            print()
        print(f"DuckDB: {Path(args.db).resolve()}")
        print(f"Run ids ({len(run_ids)}): {', '.join(rid[:8] for rid in run_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
