"""Compare two eval runs — and say first whether they may be compared at all.

The store answers "what did these two runs score". This answers the question that has to come
first: **are these the same experiment?** On 2026-08-15 a five-trial run read as a 6% pipeline
improvement and was a model swap (RG-029); the evidence was in the same table, but nothing
computed it. Now something does.

Usage::

    python -m scripts.compare_runs --list                # what is in the store, and what it pins
    python -m scripts.compare_runs 57960670 d9181d5b     # compare two runs (id prefixes are fine)
    python -m scripts.compare_runs A B --per-case        # + the per-case deltas
    python -m scripts.compare_runs A B --varying child_chunk_size   # a sweep pair
    python -m scripts.compare_runs A --against tests/eval/baselines/x.md   # vs a commit
    python -m scripts.compare_runs A B --db other.duckdb

Read-only: it opens the DuckDB store, prints markdown, and writes nothing. Verdicts are printed
**before** the numbers, and the numbers are printed either way — informing beats blocking, and a
suppressed table only sends the reader back to the raw store where there is no verdict at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from doc_assistant.config import PROJECT_ROOT
from doc_assistant.eval.baseline_doc import baseline_settings, parse_provenance, table_drift
from doc_assistant.eval.comparability import Status, compare
from doc_assistant.eval.report import (
    compare_runs,
    diff_runs,
    format_comparability,
    format_diff,
    format_run_summary,
)
from doc_assistant.eval.store import RunPrefixError, Store

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DB = PROJECT_ROOT / "data" / "eval.duckdb"

#: Exit codes. A caller wiring this into a check wants the verdict without parsing prose, and
#: `unknown` is deliberately not `0`: "nobody recorded it" must not read as "all clear".
EXIT_BY_STATUS = {Status.COMPARABLE: 0, Status.UNKNOWN: 3, Status.NOT_COMPARABLE: 4}

#: A declared-varying setting that did not actually differ. Its own code because it is a
#: different failure from either of the above: the runs ARE comparable, and the experiment
#: between them is void (KI-41).
EXIT_VARIED_NOTHING = 5


def format_inventory(store: Store, limit: int) -> str:
    """What is in the store, and which runs pin enough of themselves to be compared.

    The two right-hand columns are the point. A run with no generator and no corpus recorded can
    still be *read*, but it can never be shown to have measured the same thing as another run —
    so the inventory says so up front rather than letting a reader discover it per comparison.
    """
    rows = store.conn.execute(
        "SELECT id, started_at, n_cases, config_json, note FROM runs "
        "ORDER BY started_at DESC LIMIT ?",
        [limit],
    ).fetchall()
    if not rows:
        return "_No runs in this store._"
    lines = [
        f"## {len(rows)} most recent run(s)",
        "",
        "| Run | Started | Cases | Generator | Index | Note |",
        "|---|---|---:|---|---|---|",
    ]
    for run_id, started, n_cases, config_json, note in rows:
        config = json.loads(config_json) if config_json else {}
        generator = (
            f"{config['llm_provider']}/{config['llm_model']}"
            if "llm_provider" in config and "llm_model" in config
            else "not recorded"
        )
        index = (
            f"{config['index_doc_count']} docs"
            if config.get("index_doc_count") is not None
            else "not recorded"
        )
        clean_note = (note or "").replace("|", "/")
        short_note = clean_note[:44] + ("..." if len(clean_note) > 44 else "")
        lines.append(
            f"| `{str(run_id)[:8]}` | {str(started)[:16]} | {n_cases} | "
            f"{generator} | {index} | {short_note} |"
        )
    unpinned = sum(1 for r in rows if "llm_model" not in json.loads(r[3] or "{}"))
    if unpinned:
        lines += [
            "",
            f"_{unpinned} of these {len(rows)} runs do not record their generator. Those runs "
            "cannot be compared to anything — not because the comparison is hard, but because "
            "nothing says what they measured (RG-029). They are readable, not comparable._",
        ]
    return "\n".join(lines)


def _against_baseline(store: Store, run_prefix: str, path: Path, varying: Sequence[str]) -> int:
    """Check one stored run against a **committed baseline document**.

    The gap this fills: `data/eval.duckdb` is gitignored, so the runs behind a committed baseline
    do not exist in a fresh clone — the numbers travel and the evidence does not. A baseline
    emitted by `scripts/emit_baseline.py` carries its settings in a provenance block, so the
    document itself is enough to say whether a new run measured the same thing.

    A baseline with no block (every one written before 2026-08-18) yields an empty settings dict,
    which the comparison reports as *unknown across the board* — correct, and more useful than it
    sounds: it names exactly which facts that document never recorded.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"Cannot read {path}: {e}")
        return 1
    try:
        run_id = store.resolve_run_id(run_prefix)
    except RunPrefixError as e:
        print(f"{e} Try --list.")
        return 1

    provenance = parse_provenance(text)
    settings = baseline_settings(text)
    drift = table_drift(text)
    if drift:
        # Printed before the verdict: if the document argues with itself, the verdict below is
        # about only half of it, and the reader needs to know which half they have been quoting.
        print(f"> **{path.name} disagrees with its own provenance block.**")
        for problem in drift:
            print(f"> - {problem}")
        print(
            "> The table is what a person quotes; the block is what this check reads. Re-emit the"
            " document, or correct it by hand and say why.\n"
        )
    if not provenance:
        print(
            f"> **{path.name} carries no provenance block.** It predates "
            "`scripts/emit_baseline.py`, so what produced its numbers lives only in its prose. "
            "Every setting below therefore reads as unrecorded — that is a statement about the "
            "document, not about this run.\n"
        )

    scorers = sorted({r["scorer_name"] for r in store.case_scores(run_id)})
    comparison = compare(settings, store.run_config(run_id), scorers, varying=varying)
    print(
        format_comparability(
            comparison,
            run_a_label=f"baseline `{path.name}`",
            run_b_label=f"run `{run_id[:8]}`",
        )
    )
    print()
    print(format_run_summary(store, run_id))
    if comparison.ineffective_variation:
        return EXIT_VARIED_NOTHING
    return EXIT_BY_STATUS[comparison.status]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", nargs="?", help="First run id (a prefix is enough)")
    parser.add_argument("run_b", nargs="?", help="Second run id (a prefix is enough)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="DuckDB store path")
    parser.add_argument(
        "--list",
        dest="list_runs",
        type=int,
        nargs="?",
        const=20,
        default=None,
        metavar="N",
        help="List the N most recent runs (default 20) and what each one pins about itself",
    )
    parser.add_argument(
        "--against",
        type=str,
        default=None,
        metavar="BASELINE.md",
        help=(
            "Compare ONE run against a committed baseline document instead of another run. "
            "Reads the provenance block scripts/emit_baseline.py writes; a baseline without "
            "one (every document older than 2026-08-18) reports unknown rather than agreement."
        ),
    )
    parser.add_argument(
        "--varying",
        nargs="+",
        default=(),
        metavar="KEY",
        help=(
            "Setting(s) this pair is MEANT to differ in (a sweep's independent variable, e.g. "
            "--varying child_chunk_size). Those differences stop blocking, everything else still "
            "does -- and a declared key that did NOT move is reported loudly, because that is "
            "KI-41: a sweep that compared one configuration with itself."
        ),
    )
    parser.add_argument(
        "--per-case",
        action="store_true",
        help="Also print the per-case, per-scorer deltas (B minus A)",
    )
    args = parser.parse_args()

    with Store(args.db) as store:
        if args.list_runs is not None:
            print(format_inventory(store, args.list_runs))
            return 0

        if args.against is not None:
            if not args.run_a:
                parser.print_usage()
                print("\n--against needs one run id: the run being checked.")
                return 1
            return _against_baseline(store, args.run_a, Path(args.against), args.varying)

        if not args.run_a or not args.run_b:
            parser.print_usage()
            print("\nGive two run ids to compare, or --list to see what is in the store.")
            return 1

        try:
            run_a = store.resolve_run_id(args.run_a)
            run_b = store.resolve_run_id(args.run_b)
        except RunPrefixError as e:
            print(f"{e} Try --list.")
            return 1

        comparison = compare_runs(store, run_a, run_b, varying=args.varying)
        label_a, label_b = f"`{run_a[:8]}`", f"`{run_b[:8]}`"
        # Verdict first: it is the sentence that decides whether the tables below mean anything.
        print(format_comparability(comparison, run_a_label=label_a, run_b_label=label_b))
        print()
        print(format_run_summary(store, run_a))
        print()
        print(format_run_summary(store, run_b))
        if args.per_case:
            print()
            print(
                format_diff(
                    diff_runs(store, run_a, run_b), run_a_label=label_a, run_b_label=label_b
                )
            )
        if comparison.ineffective_variation:
            # Worse than incomparable: the numbers look valid and answer nothing.
            return EXIT_VARIED_NOTHING
        return EXIT_BY_STATUS[comparison.status]


if __name__ == "__main__":
    raise SystemExit(main())
