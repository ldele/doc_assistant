"""Run the eval harness against the doc_assistant RAG pipeline.

Loads cases from ``tests/eval/cases.yaml``, wires the adapter +
scorer mix, runs the suite, persists to ``data/eval.duckdb``, and
prints the summary table.

Default scorer mix is the **free** subset (no API calls):
``contains_all`` + ``citation_overlap``. Pass ``--with-embedding`` or
``--with-llm-judge`` to opt into the paid scorers.

Usage::

    python -m scripts.run_eval                       # free scorers only
    python -m scripts.run_eval --with-embedding      # + embedding similarity
    python -m scripts.run_eval --with-llm-judge      # + LLM judge (costs $$$)
    python -m scripts.run_eval --cases custom.yaml   # custom case file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doc_assistant.config import ANTHROPIC_API_KEY, JUDGE_PROVIDER, PROJECT_ROOT
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

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_CASES = PROJECT_ROOT / "tests" / "eval" / "cases.yaml"
DEFAULT_DB = PROJECT_ROOT / "data" / "eval.duckdb"


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

    print(f"Loaded {len(cases)} cases from {args.cases}")
    print("Loading RAG pipeline (this can take a minute)...")
    from doc_assistant.pipeline import RAGPipeline

    pipeline = RAGPipeline()
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

    with Store(args.db) as store:
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
