"""Write a baseline document from the run record, so the committed file carries its own evidence.

`tests/eval/baselines/` is the project's reference record, and its setup sections have always been
typed by hand. That is where an error hides longest: the Haiku-vs-llama split across the 2026-08-08
arms lived only in prose, and `data/eval.duckdb` — the thing that could have contradicted it — is
gitignored, so a fresh clone has the conclusions and none of the evidence.

This emits the mechanical half from the stored runs: settings, corpus composition, generator, the
aggregate table, and a machine-readable provenance block that `compare_runs --against` can check a
later run against. The judgement half stays a TODO for a person, because the caveats are what make
a baseline worth keeping.

Usage::

    python -m scripts.emit_baseline 57960670 e9d6e5ab --title "Sparse arm, private 35"
    python -m scripts.emit_baseline 57960670 --out tests/eval/baselines/my_result_2026-08-18.md
    python -m scripts.emit_baseline --note "chunk-sweep | parent=2000/200 child=400/50"

**It refuses to emit from runs that are not one experiment.** Trials of a baseline must differ in
nothing; if they do, the document would average two experiments and read as one. That check is the
comparability layer applied to the emitter's own inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doc_assistant.config import PROJECT_ROOT
from doc_assistant.eval.baseline_doc import render_baseline
from doc_assistant.eval.comparability import Status
from doc_assistant.eval.report import compare_runs, format_comparability
from doc_assistant.eval.store import RunPrefixError, Store

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DB = PROJECT_ROOT / "data" / "eval.duckdb"


def agreed_settings(store: Store, run_ids: list[str]) -> dict[str, object]:
    """The settings every run agrees on. A key any run records differently is dropped.

    Dropping rather than picking one: a document that printed the first trial's value for a
    setting the trials disagreed about would state something no reader could act on. A dropped key
    then renders as "not recorded", which is the honest reading — this document cannot vouch for
    it. The pairwise comparability check below normally rejects such a set before it gets here;
    this is the backstop for `--force`.
    """
    configs = [store.run_config(run_id) for run_id in run_ids]
    if not configs:
        return {}
    shared: dict[str, object] = {}
    for key, value in configs[0].items():
        if all(key in cfg and cfg[key] == value for cfg in configs[1:]):
            shared[key] = value
    # trial_index differs by design across trials and would always drop out; n_trials is the
    # document's own `run_ids` length. Neither describes what was measured.
    return {k: v for k, v in shared.items() if k not in {"trial_index", "n_trials"}}


def resolve_runs(store: Store, ids: list[str], note: str | None) -> list[str]:
    """Run ids from prefixes, or every run carrying an exact note (how a sweep tags its arms)."""
    if note is not None:
        rows = store.conn.execute(
            "SELECT id FROM runs WHERE note = ? ORDER BY started_at", [note]
        ).fetchall()
        if not rows:
            raise RunPrefixError(f"No run carries the note {note!r}.")
        return [str(r[0]) for r in rows]
    return [store.resolve_run_id(prefix) for prefix in ids]


def check_one_experiment(store: Store, run_ids: list[str]) -> list[str]:
    """Problems that make these runs not one experiment. Empty means safe to emit.

    Every run is compared against the first rather than pairwise-all: comparability here is
    equality of recorded settings, so agreement with a common reference is agreement with each
    other, at N-1 comparisons instead of N².
    """
    problems: list[str] = []
    for other in run_ids[1:]:
        comparison = compare_runs(store, run_ids[0], other)
        if comparison.status is Status.NOT_COMPARABLE:
            problems.append(
                format_comparability(
                    comparison,
                    run_a_label=f"`{run_ids[0][:8]}`",
                    run_b_label=f"`{other[:8]}`",
                )
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", help="Run ids (prefixes are fine) — the trials")
    parser.add_argument(
        "--note", type=str, default=None, help="Instead of ids: every run with this exact note"
    )
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="DuckDB store path")
    parser.add_argument("--title", type=str, default="Eval baseline", help="Document title")
    parser.add_argument(
        "--date", type=str, default=None, metavar="YYYY-MM-DD", help="Date in the heading"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write here instead of stdout. Refuses to overwrite without --force.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out, and emit even when the runs are not one experiment",
    )
    args = parser.parse_args()

    if not args.run_ids and args.note is None:
        parser.print_usage()
        print("\nGive one or more run ids, or --note to select a sweep arm's runs.")
        return 1
    with Store(args.db) as store:
        try:
            run_ids = resolve_runs(store, args.run_ids, args.note)
        except RunPrefixError as e:
            print(str(e))
            return 1

        problems = check_one_experiment(store, run_ids)
        if problems and not args.force:
            print(f"REFUSED: the {len(run_ids)} runs are not one experiment.\n")
            for problem in problems:
                print(problem)
                print()
            print(
                "A baseline averages its trials, so emitting this would present two experiments\n"
                "as one number. Select the runs that belong together, or pass --force if you\n"
                "know why they differ and intend to say so in the document."
            )
            return 4

        date = args.date
        if date is None:
            row = store.conn.execute(
                "SELECT MAX(started_at) FROM runs WHERE id = ANY(?)", [run_ids]
            ).fetchone()
            date = str(row[0])[:10] if row and row[0] else "undated"

        settings = agreed_settings(store, run_ids)
        scores = store.aggregate_runs(run_ids)
        notes = []
        if problems:
            notes.append(
                "**--force was used: these runs are NOT one experiment.** Say here what differed "
                "and why the average is still meaningful, or split the document."
            )
        document = render_baseline(
            title=args.title,
            date=date,
            settings=settings,
            run_ids=run_ids,
            scores=scores,
            notes=notes,
        )

    if args.out is None:
        print(document)
        return 0

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"REFUSED: {out} exists. Pass --force to overwrite, or choose another path.")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    # encoding= is not optional on this box: file I/O defaults to the ANSI codepage on Windows
    # and this document carries non-ASCII (.claude/CONTEXT.md section 9).
    out.write_text(document, encoding="utf-8")
    print(f"Wrote {out} ({len(run_ids)} run(s)).")
    print("Now fill the TODO — the numbers are emitted, the judgement is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
