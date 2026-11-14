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
    python -m scripts.run_eval --provider ollama --model llama3.1:8b   # one column
    python -m scripts.run_eval --panel tests/eval/panel.example.yaml   # several models
    python -m scripts.run_eval --adapter controller ...   # through ChatController (own data home)
    python -m scripts.run_eval ... --paired-baseline <id>,<id>,<id>   # interval vs earlier trials

Before anything runs, the run states per column how many model turns it will ask for and who
bills them, and refuses a column that bills (or whose billing is not known) unless
``--allow-billed`` is given. The spec (``docs/specs/eval-harness-chat.md``) fixes how a run
measures; which models it uses and what it may cost are the run's settings, recorded with it.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc_assistant import sparse_index
from doc_assistant.config import (
    ANTHROPIC_API_KEY,
    DATA_PATH,
    JUDGE_MODEL,
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
from doc_assistant.eval.adapters import (
    controller_adapter,
    embedding_callable,
    rag_pipeline_adapter,
)
from doc_assistant.eval.cases import EvalCase
from doc_assistant.eval.paired import format_paired
from doc_assistant.eval.panel import Column, load_panel, turns_per_trial
from doc_assistant.eval.report import (
    compare_runs,
    format_aggregate,
    format_comparability,
    format_flaky_cases,
    format_run_summary,
    paired_runs,
)
from doc_assistant.eval.run_settings import run_defining_settings
from doc_assistant.eval.scorers import chat_scorers
from doc_assistant.eval.store import RunPrefixError

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
    scorers: list[Scorer] = [ContainsAllScorer(), CitationOverlapScorer(), *chat_scorers()]
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


@dataclass(frozen=True)
class Planned:
    """One column's share of the run, as stated before anything is asked of a model."""

    column: Column
    turns: int  # upper bound on generation turns
    reviewer: tuple[str, str] | None  # the reviewer instrument a controller turn may call
    billed: list[str]  # "provider/model" pairs that bill or whose billing is not known


def _bills(provider: str, model: str) -> bool:
    """Billed, or not known (ADR-060's guard rule), for one provider and model."""
    from doc_assistant import providers
    from doc_assistant.llm import ollama_probe

    if provider.strip().lower() == "ollama":
        ollama_probe()  # records which models a signed-in Ollama forwards to ollama.com
    return providers.describe(provider, model).billing.guard_trips


def plan_columns(
    columns: list[Column], cases: list[EvalCase], *, repeat: int, judge: tuple[str, str] | None
) -> list[Planned]:
    """Each app column's turn count and billing, before any model is called."""
    from doc_assistant.llm import resolve_reviewer

    per_trial = turns_per_trial([len(c.prior_turns) for c in cases])
    out: list[Planned] = []
    for col in columns:
        if not col.runs_here:
            continue
        reviewer = (
            resolve_reviewer(col.provider, col.model) if col.adapter == "controller" else None
        )
        pairs = [(col.provider, col.model)]
        if reviewer is not None:
            pairs.append(reviewer)
        if judge is not None:
            pairs.append(judge)
        billed = [f"{p}/{m}" for p, m in dict.fromkeys(pairs) if _bills(p, m)]
        out.append(Planned(col, per_trial * repeat, reviewer, billed))
    return out


def format_plan(plan: list[Planned], *, repeat: int, judge: tuple[str, str] | None) -> str:
    """The pre-run statement: what each column will ask of which model, and who bills."""
    lines = [
        "Before the run — per column (upper bounds; a turn the app answers without a model "
        "still counts):",
        "| column | adapter | provider / model | turns | reviewer (<= 1 per turn) | judge "
        "| bills |",
        "|---|---|---|---:|---|---|---|",
    ]
    for p in plan:
        rev = f"{p.reviewer[0]}/{p.reviewer[1]}" if p.reviewer else "-"
        jud = f"{judge[0]}/{judge[1]} x {p.turns}" if judge else "-"
        lines.append(
            f"| {p.column.name} | {p.column.adapter} | {p.column.provider}/{p.column.model} | "
            f"{p.turns} | {rev} | {jud} | {', '.join(p.billed) or 'nobody'} |"
        )
    return "\n".join(lines)


def controller_data_home_error(data_path: Path, project_root: Path) -> str | None:
    """Why the controller adapter may not run on this data home, or ``None`` when it may.

    The controller records every turn in the library it runs on, so it needs a data home of its
    own: named explicitly with ``DOC_DATA_DIR``, and not the checkout's ``data/`` (which the
    installed app opens in place on the dev machine)."""
    if not os.getenv("DOC_DATA_DIR"):
        return (
            "--adapter controller writes an answer record per turn into the library it runs on. "
            "Set DOC_DATA_DIR to a data home of its own (with the eval corpus ingested)."
        )
    if data_path.resolve() == (project_root / "data").resolve():
        return (
            "--adapter controller refuses the checkout's data/ folder: "
            "point DOC_DATA_DIR elsewhere."
        )
    return None


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
        "--baseline",
        type=str,
        default=None,
        metavar="RUN_ID",
        help=(
            "After the run, report whether it may be compared to this earlier run "
            "(an id prefix is enough). Closes the RG-021 loop: a run over a corpus, "
            "generator or geometry the baseline never saw is a different experiment, "
            "and this says so instead of leaving two means side by side."
        ),
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
        "--panel",
        type=str,
        default=None,
        help=(
            "A panel file (YAML) listing the columns to run: provider, model, adapter. "
            "Replaces --provider/--model/--adapter. See tests/eval/panel.example.yaml."
        ),
    )
    parser.add_argument(
        "--adapter",
        choices=("pipeline", "controller"),
        default="pipeline",
        help=(
            "pipeline: RAGPipeline directly (old runs were measured so). controller: the "
            "headless ChatController the app uses (router, history, framing; replays "
            "conversation turns; writes answer records, so it needs its own DOC_DATA_DIR)."
        ),
    )
    parser.add_argument(
        "--allow-billed",
        action="store_true",
        help="Run columns whose provider bills, or whose billing is not known. Off by default.",
    )
    parser.add_argument(
        "--paired-baseline",
        type=str,
        default=None,
        metavar="RUN_IDS",
        help=(
            "Comma-separated run ids (prefixes are enough) of the baseline's trials. After the "
            "run, prints the paired bootstrap interval per family and scorer against them. One "
            "app column only: an interval is drawn within a column, never across models."
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

    if args.panel:
        if args.provider or args.model or args.adapter != "pipeline":
            print("--panel replaces --provider, --model and --adapter; give one or the other.")
            return 1
        try:
            columns = load_panel(args.panel)
        except (OSError, ValueError) as e:
            print(f"--panel: {e}")
            return 1
    else:
        try:
            provider, model = resolve_generator(args.provider, args.model)
        except ValueError as e:
            print(str(e))
            return 1
        columns = [Column("default", provider, model, adapter=args.adapter)]

    app_columns = [c for c in columns if c.runs_here]
    for c in columns:
        if not c.runs_here:
            print(f"Column {c.name}: kind {c.kind}, filled outside the app; listed, not run here.")
    if not app_columns:
        print("No app column to run.")
        return 1
    if args.paired_baseline and len(app_columns) > 1:
        print("--paired-baseline needs one app column: an interval is never drawn across models.")
        return 1
    if any(c.adapter == "controller" for c in app_columns):
        err = controller_data_home_error(DATA_PATH, PROJECT_ROOT)
        if err:
            print(err)
            return 1

    print(f"Loaded {len(cases)} cases from {args.cases}")
    print(f"Data home: {DATA_PATH}")
    judge = (JUDGE_PROVIDER, JUDGE_MODEL) if args.with_llm_judge else None
    plan = plan_columns(app_columns, cases, repeat=args.repeat, judge=judge)
    print(format_plan(plan, repeat=args.repeat, judge=judge))
    billed = sorted({b for p in plan for b in p.billed})
    if billed and not args.allow_billed:
        print(
            f"\nRefused: {', '.join(billed)} bills, or its billing is not known. "
            "Pass --allow-billed to run it anyway."
        )
        return 1
    if judge is not None:
        print("llm_judge: optional, and every value it gives is marked not calibrated (spec, H5).")

    # Before the pipeline loads, not after: the guard's Ctrl-C window opens before anything is
    # spent, and loading costs a minute of model weights either way.
    from doc_assistant.llm import assert_provider_intent

    for p in plan:
        if p.billed:
            assert_provider_intent(
                p.column.provider,
                operation=f"run_eval column {p.column.name}",
                model=p.column.model,
                scope=f"up to {p.turns} turns",
            )

    print("Loading RAG pipeline (this can take a minute)...")
    from doc_assistant.pipeline import RAGPipeline

    pipeline = RAGPipeline(bm25_weight=args.bm25_weight)
    print(f"BM25 ensemble weight: {pipeline.bm25_weight} (vector {1.0 - pipeline.bm25_weight})")
    scorers = _build_scorers(
        pipeline,
        with_embedding=args.with_embedding,
        with_llm_judge=args.with_llm_judge,
    )
    print(f"Scorers: {', '.join(s.name for s in scorers)}")
    runner = Runner(scorers)
    controller = None

    def _progress(i: int, total: int, case: object) -> None:
        print(f"  [{i + 1:>2}/{total}] {getattr(case, 'id', '?')}")

    # Read once, before the trials: a corpus that moves mid-run is a fact about the whole run.
    composition = index_composition(pipeline.indexed_doc_hashes)
    if composition["index_doc_digest"] is None:
        print("Index composition: UNKNOWN (keyword index unavailable - vector-only retrieval)")
    else:
        print(
            f"Index composition: {composition['index_doc_count']} documents, "
            f"digest {composition['index_doc_digest']}"
        )

    # settings_provider: every run records the chunk/retrieval settings that produced it.
    reviewers = {p.column.name: p.reviewer for p in plan}
    with Store(args.db, settings_provider=run_defining_settings) as store:
        all_run_ids: list[str] = []
        for col in app_columns:
            # ADR-011's live switch: rebuilds only the thin chat-model wrapper; the embedder,
            # stores and reranker the run measures are untouched. Never persisted.
            if (col.provider, col.model) != (pipeline.provider, pipeline.model):
                pipeline.set_chat_model(col.provider, col.model)
            takes_case = col.adapter == "controller"
            if takes_case:
                if controller is None:
                    from doc_assistant.chat_controller import ChatController

                    controller = ChatController(rag=pipeline)
                sut: Any = controller_adapter(controller)
            else:
                sut = rag_pipeline_adapter(pipeline)
            print(f"\n##### Column {col.name}: {col.provider} / {col.model} via {col.adapter}")
            run_ids: list[str] = []
            for trial in range(args.repeat):
                if args.repeat > 1:
                    print(f"\n=== Trial {trial + 1}/{args.repeat} ===")
                print(f"Running {len(cases)} cases...")
                results = runner.run(cases, sut, progress=_progress, takes_case=takes_case)

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
                        # The generator that actually ran, which `run_defining_settings` cannot
                        # see (set_chat_model assigns no module global). Explicit keys win.
                        "llm_provider": pipeline.provider,
                        "llm_model": pipeline.model,
                        "column": col.name,
                        "adapter": col.adapter,
                        # The reviewer a controller turn may call is part of what it measured.
                        "reviewer": "/".join(reviewers[col.name] or ()) or None,
                        "judge": f"{judge[0]}/{judge[1]}" if judge else None,
                        "judge_calibrated": False if judge else None,
                        "data_home": str(DATA_PATH),
                        **composition,
                    },
                    note=trial_note,
                )
                run_ids.append(run_id)
                print()
                print(format_run_summary(store, run_id))
                print(f"Trial run id: {run_id}")
            all_run_ids.extend(run_ids)

            if args.repeat > 1:
                print()
                print(
                    format_aggregate(
                        store, run_ids, label=f"Aggregate ({col.name}, n={args.repeat})"
                    )
                )
                print()
                print(format_flaky_cases(store.flaky_cases(run_ids)))

            if args.paired_baseline:
                print()
                try:
                    base_ids = [
                        store.resolve_run_id(x.strip())
                        for x in args.paired_baseline.split(",")
                        if x.strip()
                    ]
                except RunPrefixError as e:
                    print(f"--paired-baseline: {e} The run itself is unaffected.")
                else:
                    print(
                        format_comparability(
                            compare_runs(store, base_ids[0], run_ids[0]),
                            run_a_label=f"baseline `{base_ids[0][:8]}`",
                            run_b_label=f"this run `{run_ids[0][:8]}`",
                        )
                    )
                    print()
                    print(
                        format_paired(
                            paired_runs(store, base_ids, run_ids),
                            label_a=f"baseline ({len(base_ids)} trial(s))",
                            label_b=f"{col.name} ({len(run_ids)} trial(s))",
                        )
                    )

        if args.baseline:
            print()
            try:
                baseline_id = store.resolve_run_id(args.baseline)
            except RunPrefixError as e:
                # Not fatal: the run happened and its numbers are already persisted.
                print(f"--baseline: {e} The run itself is unaffected.")
            else:
                # The first trial of the first column stands for the group.
                print(
                    format_comparability(
                        compare_runs(store, baseline_id, all_run_ids[0]),
                        run_a_label=f"baseline `{baseline_id[:8]}`",
                        run_b_label=f"this run `{all_run_ids[0][:8]}`",
                    )
                )

        print()
        print(f"DuckDB: {Path(args.db).resolve()}")
        print(f"Run ids ({len(all_run_ids)}): {', '.join(rid[:8] for rid in all_run_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
