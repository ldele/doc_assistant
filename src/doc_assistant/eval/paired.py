"""Paired comparison of two groups of runs, per family and per scorer (generic).

Spec ``docs/specs/eval-harness-chat.md`` § The statistic. A group is the trials of one
configuration (``--repeat N`` writes N runs). The unit is the **case**: its score in a group is
its mean over that group's trials, and the comparison is the per-case difference B - A, with a
paired bootstrap interval of its mean. Binary scorers (a hold or a break) are reported as counts,
with the cases that changed state, never as an interval.

Comparisons are made within one column (one provider, model and adapter). Two models are not
paired runs of one system, so this module draws no interval across them; the caller decides
which groups it is given, and the comparability check (``compare_runs``) says whether they are
the same experiment.

Standard library only: the bootstrap is ``random.Random(seed)``, and the seed is part of the
result so the interval can be reproduced.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

#: Resamples of cases per interval. 10,000 keeps a 95% percentile interval stable to about the
#: third decimal on the n of a family (8 to 40 cases).
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261113


@dataclass(frozen=True)
class ScoreDiff:
    """One scorer on one family: the mean per-case difference B - A and its interval."""

    family: str
    scorer_name: str
    n: int  # cases scored in both groups
    mean_a: float
    mean_b: float
    diff: float
    low: float
    high: float

    @property
    def verdict(self) -> str:
        """The wording the spec fixes: never "better" or "worse" alone."""
        if self.n == 0:
            return "no cases in both groups"
        if self.low > 0:
            return "higher at this n"
        if self.high < 0:
            return "a measured drop"
        return "no difference at this n"


@dataclass(frozen=True)
class HoldCount:
    """One binary scorer on one family: how many cases held in each group, and which changed.

    A case holds in a group when it held in every trial of that group: a hold that comes and
    goes between trials is not one a gate can rely on."""

    family: str
    scorer_name: str
    n: int
    held_a: int
    held_b: int
    broke: list[str] = field(default_factory=list)  # held in A, not in B
    fixed: list[str] = field(default_factory=list)  # held in B, not in A


@dataclass(frozen=True)
class PairedComparison:
    diffs: list[ScoreDiff]
    holds: list[HoldCount]
    seed: int
    resamples: int

    @property
    def new_breaks(self) -> list[tuple[str, str, str]]:
        """``(family, scorer, case)`` for every case that held in A and not in B."""
        return [(h.family, h.scorer_name, c) for h in self.holds for c in h.broke]


def _case_means(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """``{(family, scorer): {case_id: mean over trials}}`` over the scored rows only."""
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in rows:
        if not r.get("scoreable", True):
            continue
        key = (str(r["family"]), str(r["scorer_name"]))
        acc.setdefault(key, {}).setdefault(str(r["case_id"]), []).append(float(r["value"]))
    return {k: {c: sum(v) / len(v) for c, v in cases.items()} for k, cases in acc.items()}


def _all_held(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, bool]]:
    """``{(family, scorer): {case_id: held in every trial}}`` over the scored rows only."""
    acc: dict[tuple[str, str], dict[str, bool]] = {}
    for r in rows:
        if not r.get("scoreable", True):
            continue
        key = (str(r["family"]), str(r["scorer_name"]))
        cases = acc.setdefault(key, {})
        held = float(r["value"]) >= 1.0
        cases[str(r["case_id"])] = cases.get(str(r["case_id"]), True) and held
    return acc


def bootstrap_interval(
    diffs: list[float], *, resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> tuple[float, float]:
    """95% percentile interval of the mean of ``diffs``, resampling cases with replacement.

    One case gives a zero-width interval at its own value: there is nothing to resample, and
    the wording then rests on that single difference, which is what n=1 can say."""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)  # nosec B311 -- seeded resampling, not security
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(0.975 * (resamples - 1))]
    return (lo, hi)


def compare_groups(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    binary: frozenset[str],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedComparison:
    """Compare group A (the baseline) with group B, per family and scorer.

    ``rows_*`` are :meth:`Store.case_score_rows` rows of each group's trials. Only cases scored
    in both groups are paired; a case scored in one group only is left out of that scorer's n.
    """
    means_a, means_b = _case_means(rows_a), _case_means(rows_b)
    held_a, held_b = _all_held(rows_a), _all_held(rows_b)
    diffs: list[ScoreDiff] = []
    holds: list[HoldCount] = []
    for key in sorted(set(means_a) | set(means_b)):
        family, scorer = key
        if scorer in binary:
            a, b = held_a.get(key, {}), held_b.get(key, {})
            common = sorted(set(a) & set(b))
            holds.append(
                HoldCount(
                    family=family,
                    scorer_name=scorer,
                    n=len(common),
                    held_a=sum(1 for c in common if a[c]),
                    held_b=sum(1 for c in common if b[c]),
                    broke=[c for c in common if a[c] and not b[c]],
                    fixed=[c for c in common if b[c] and not a[c]],
                )
            )
            continue
        ma, mb = means_a.get(key, {}), means_b.get(key, {})
        common = sorted(set(ma) & set(mb))
        per_case = [mb[c] - ma[c] for c in common]
        low, high = bootstrap_interval(per_case, resamples=resamples, seed=seed)
        n = len(common)
        diffs.append(
            ScoreDiff(
                family=family,
                scorer_name=scorer,
                n=n,
                mean_a=sum(ma[c] for c in common) / n if n else 0.0,
                mean_b=sum(mb[c] for c in common) / n if n else 0.0,
                diff=sum(per_case) / n if n else 0.0,
                low=low,
                high=high,
            )
        )
    return PairedComparison(diffs=diffs, holds=holds, seed=seed, resamples=resamples)


def format_paired(cmp: PairedComparison, *, label_a: str = "A", label_b: str = "B") -> str:
    """Markdown: one table of intervals and one of holds, each grouped by family."""
    lines = [
        f"### Paired comparison — {label_b} vs {label_a} (baseline)",
        "",
        f"Per-case difference B - A, mean over trials; paired bootstrap 95% interval "
        f"({cmp.resamples} resamples, seed {cmp.seed}).",
        "",
    ]
    if cmp.diffs:
        lines += [
            "| family | scorer | n | A | B | B - A | 95% interval | reads as |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
        for d in cmp.diffs:
            lines.append(
                f"| {d.family} | {d.scorer_name} | {d.n} | {d.mean_a:.3f} | {d.mean_b:.3f} | "
                f"{d.diff:+.3f} | [{d.low:+.3f}, {d.high:+.3f}] | {d.verdict} |"
            )
        lines.append("")
    if cmp.holds:
        lines += [
            "| family | scorer | n | held A | held B | broke (held in A only) | fixed |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for h in cmp.holds:
            lines.append(
                f"| {h.family} | {h.scorer_name} | {h.n} | {h.held_a} | {h.held_b} | "
                f"{', '.join(h.broke) or '—'} | {', '.join(h.fixed) or '—'} |"
            )
        lines.append("")
    if not cmp.diffs and not cmp.holds:
        lines.append("_No scored cases in both groups._")
    return "\n".join(lines)
