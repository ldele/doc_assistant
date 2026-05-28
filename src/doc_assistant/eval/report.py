"""Eval reporting helpers (generic).

Two views:

* ``format_run_summary`` — markdown table of mean score per scorer.
* ``diff_runs`` — pairwise per-case delta between two runs against the
  same eval set.

Pure formatting + arithmetic; the store does all DB access.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        lines.append(
            f"| {scorer_name} | {mean_cell} | {s['n_scored']} | {s['n_skipped']} |"
        )
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
