"""Eval reporting helpers (generic).

Views:

* ``format_run_summary`` — markdown table of mean score per scorer.
* ``diff_runs`` — pairwise per-case delta between two runs against the
  same eval set.
* ``compare_runs`` / ``format_comparability`` — whether that delta means
  anything at all (see :mod:`doc_assistant.eval.comparability`). A diff
  answers "how much did the number move"; a comparison answers "is this
  the same experiment", and printing the first without the second is how
  a model swap came to read as a 6% pipeline win (RG-029).

Pure formatting + arithmetic; the store does all DB access.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from doc_assistant.eval.comparability import (
    Comparison,
    Difference,
    Stage,
    State,
    Status,
    compare,
)
from doc_assistant.eval.store import Store


@dataclass
class RunDiffRow:
    """One row of a two-run diff: case + per-scorer delta."""

    case_id: str
    scorer_name: str
    value_a: float
    value_b: float

    @property
    def delta(self) -> float:
        return self.value_b - self.value_a


def format_run_summary(store: Store, run_id: str) -> str:
    """Markdown summary: per-scorer mean + scored/skipped counts.

    A scorer that couldn't grade any case (e.g., every case missing
    ``expected_answer``) shows ``mean = -`` and ``n_scored = 0`` so
    the reader can tell "scored zero" apart from "didn't run".
    """
    stats = store.scorer_stats(run_id)
    if not stats:
        return f"Run `{run_id[:8]}` has no scores."
    lines = [
        f"## Run `{run_id[:8]}` summary",
        "",
        "| Scorer | Mean | n_scored | n_skipped |",
        "|---|---:|---:|---:|",
    ]
    for scorer_name, s in sorted(stats.items()):
        mean = s["mean"]
        mean_cell = f"{mean:.3f}" if isinstance(mean, float) else "-"
        lines.append(f"| {scorer_name} | {mean_cell} | {s['n_scored']} | {s['n_skipped']} |")
    return "\n".join(lines)


def diff_runs(store: Store, run_a_id: str, run_b_id: str) -> list[RunDiffRow]:
    """Return per-case, per-scorer deltas (B minus A) for cases present in both runs."""
    a = {(r["case_id"], r["scorer_name"]): r["value"] for r in store.case_scores(run_a_id)}
    b = {(r["case_id"], r["scorer_name"]): r["value"] for r in store.case_scores(run_b_id)}
    shared = sorted(set(a.keys()) & set(b.keys()))
    return [
        RunDiffRow(
            case_id=cid,
            scorer_name=scorer,
            value_a=a[(cid, scorer)],
            value_b=b[(cid, scorer)],
        )
        for cid, scorer in shared
    ]


def format_aggregate(store: Store, run_ids: list[str], *, label: str = "Aggregate") -> str:
    """Markdown table of mean + two std columns per scorer across N runs.

    ``trial_mean_std`` is what you want for measurement reliability —
    how different is the mean if you rerun the whole eval?
    ``score_std`` is the per-(case, trial) spread; dominated by
    cross-case variance, less useful for comparison across runs.
    """
    stats = store.aggregate_runs(run_ids)
    if not stats:
        return f"{label}: no scores in the {len(run_ids)} run(s)."
    lines = [
        f"## {label} over {len(run_ids)} run(s)",
        "",
        "| Scorer | Mean | Trial-mean std | Per-score std | n_scored | n_skipped |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scorer_name, s in sorted(stats.items()):
        mean = s["mean"]
        tms = s.get("trial_mean_std")
        sds = s.get("score_std")
        mean_cell = f"{mean:.3f}" if isinstance(mean, float) else "-"
        tms_cell = f"{tms:.3f}" if isinstance(tms, float) else "-"
        sds_cell = f"{sds:.3f}" if isinstance(sds, float) else "-"
        lines.append(
            f"| {scorer_name} | {mean_cell} | {tms_cell} | {sds_cell} | "
            f"{s['n_scored']} | {s['n_skipped']} |"
        )
    lines.append("")
    lines.append(
        "_Trial-mean std answers 'how different would the mean be on a rerun'. "
        "Per-score std is dominated by per-case spread and is less useful for "
        "cross-run comparison._"
    )
    return "\n".join(lines)


def format_flaky_cases(rows: list[dict[str, Any]]) -> str:
    """Markdown list of cases that failed intermittently across trials."""
    if not rows:
        return "_No intermittent failures across trials._"
    lines = [
        f"## {len(rows)} flaky (case, scorer) pair(s)",
        "",
        "_Cases that scored in some trials and were skipped in others — "
        "usually an API timeout or judge parse failure on edge-case prompts._",
        "",
        "| Scorer | Case | Scored | Skipped |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scorer_name']} | {r['case_id']} | {r['n_scored']} | {r['n_skipped']} |"
        )
    return "\n".join(lines)


def compare_runs(
    store: Store, run_a_id: str, run_b_id: str, *, varying: Sequence[str] = ()
) -> Comparison:
    """Comparability of two stored runs, judged on what they recorded — and on their case sets.

    The case set is passed in as an extra difference rather than left to ``n_cases``, because a
    count is not an identity: two runs of 35 cases can be two different sets of 35, and the
    private set has been re-authored in place more than once. When both runs recorded their
    per-case rows the check is exact; a run with no rows at all (an empty run) leaves it unknown,
    which is the honest answer rather than a vacuous match.
    """
    cases_a, cases_b = store.case_ids(run_a_id), store.case_ids(run_b_id)
    if not cases_a or not cases_b:
        which = "A" if not cases_a else "B"
        case_diff = Difference(
            "case_set",
            Stage.CASES,
            State.UNKNOWN,
            detail=f"run {which} recorded no per-case rows",
        )
    elif set(cases_a) == set(cases_b):
        case_diff = Difference(
            "case_set", Stage.CASES, State.SAME, value_a=f"{len(cases_a)} cases"
        )
    else:
        only_a = sorted(set(cases_a) - set(cases_b))
        only_b = sorted(set(cases_b) - set(cases_a))
        case_diff = Difference(
            "case_set",
            Stage.CASES,
            State.DIFFERENT,
            value_a=f"{len(cases_a)} cases, {len(only_a)} not in B",
            value_b=f"{len(cases_b)} cases, {len(only_b)} not in A",
        )

    scorers = sorted(
        {r["scorer_name"] for r in store.case_scores(run_a_id)}
        & {r["scorer_name"] for r in store.case_scores(run_b_id)}
    )
    return compare(
        store.run_config(run_a_id),
        store.run_config(run_b_id),
        scorers,
        extra_differences=(case_diff,),
        varying=varying,
    )


_STATUS_MARK = {
    Status.COMPARABLE: "ok",
    Status.UNKNOWN: "UNKNOWN",
    Status.NOT_COMPARABLE: "NOT COMPARABLE",
}


def format_comparability(
    comparison: Comparison, *, run_a_label: str = "A", run_b_label: str = "B"
) -> str:
    """Markdown verdict: per scorer, then the evidence it was judged on.

    Ordered verdict-first because that is the sentence a reader needs before they look at any
    number — and unrecorded settings get their own section rather than being folded in with the
    equal ones, since "we checked and it matched" and "nobody wrote it down" are the two claims
    this whole layer exists to keep apart.
    """
    lines = [
        f"## Comparability: {run_a_label} vs {run_b_label}",
        "",
        f"**Overall: {_STATUS_MARK[comparison.status]}** "
        f"(the worst verdict across {len(comparison.verdicts)} shared scorer(s))",
        "",
    ]
    if comparison.ineffective_variation:
        # Above the verdict table on purpose: a void experiment is worse news than an
        # incomparable one, because its numbers look valid and answer a question nobody asked.
        keys = ", ".join(d.key for d in comparison.ineffective_variation)
        lines += [
            f"> **The declared variable did not change: {keys}.** These two runs were meant to "
            "differ there and do not, so whatever they show is one configuration compared with "
            "itself — the KI-41 shape. Check that the setting reached the code (a sweep driving "
            "its grid through the environment can be silently overwritten by `.env`).",
            "",
        ]
    if comparison.verdicts:
        lines += [
            "| Scorer | Verdict | Why |",
            "|---|---|---|",
        ]
        for v in comparison.verdicts:
            lines.append(f"| {v.scorer_name} | {_STATUS_MARK[v.status]} | {v.reason} |")
    else:
        lines.append("_The two runs share no scorer, so there is nothing to compare._")

    differing = comparison.by_state(State.DIFFERENT)
    intended = [d for d in differing if d.key in comparison.varying]
    unintended = [d for d in differing if d.key not in comparison.varying]
    if intended:
        lines += ["", f"**Varied on purpose ({len(intended)})**", ""]
        lines += [f"- {d.stage.value} · {d.describe()}" for d in intended]
    if unintended:
        heading = "Differing settings NOT declared" if comparison.varying else "Differing settings"
        lines += ["", f"**{heading} ({len(unintended)})**", ""]
        lines += [f"- {d.stage.value} · {d.describe()}" for d in unintended]

    unknown = comparison.by_state(State.UNKNOWN)
    if unknown:
        lines += [
            "",
            f"**Not recorded ({len(unknown)}) — cannot be ruled out**",
            "",
        ]
        lines += [f"- {d.stage.value} · {d.describe()}" for d in unknown]
        lines += [
            "",
            "_An unrecorded setting is never assumed to have matched. Runs from before a key "
            "existed cannot be back-filled: an inference would be indistinguishable from a "
            "recording, which is the defect the keys were added to prevent (RG-029)._",
        ]
    return "\n".join(lines)


def format_diff(rows: list[RunDiffRow], *, run_a_label: str = "A", run_b_label: str = "B") -> str:
    """Markdown table of a diff. Sort: largest absolute delta first."""
    if not rows:
        return "No overlapping (case, scorer) pairs between the two runs."

    lines = [
        f"## Diff: {run_b_label} - {run_a_label}",
        "",
        f"| Case | Scorer | {run_a_label} | {run_b_label} | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    sorted_rows = sorted(rows, key=lambda r: -abs(r.delta))
    for r in sorted_rows:
        sign = "+" if r.delta >= 0 else ""
        lines.append(
            f"| {r.case_id} | {r.scorer_name} | {r.value_a:.3f} | "
            f"{r.value_b:.3f} | {sign}{r.delta:.3f} |"
        )
    return "\n".join(lines)
