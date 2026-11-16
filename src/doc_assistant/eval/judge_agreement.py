"""How far a judge agrees with a person's grades, and the gate a judge must pass (generic).

Spec ``docs/specs/eval-harness-chat.md`` (H5) and ``tests/eval/TESTING.md`` § the local judge's
agreement gate. The unit is the **answer**: a person's grade and a judge's grade on the same
answer, each the mean of its rubric dimensions (and a judge's also its mean over repeats). Three
numbers, all on the answers both graded:

- **decision agreement** (binding): over every pair of answers the person grades at least
  ``min_gap`` apart, the share the judge orders the same way. A judge tie counts as a miss: a
  judge that cannot tell two answers apart has not made the person's decision.
- **rank correlation**: Spearman's rho on the grades, with a bootstrap 95% interval over answers.
- **per-answer deviation**: the mean absolute difference on the 1-5 scale, and its sign (bias).

Model-neutral: any judge's grades go through the same functions. Standard library only, seeded
like ``paired.py`` so an interval can be reproduced.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from doc_assistant.eval.paired import DEFAULT_RESAMPLES, DEFAULT_SEED

#: Proposed in TESTING.md before any judge was measured; kept after the first distribution
#: (H5, 2026-11-16), which gave no reason to loosen them. See ``tests/eval/judge_check/``.
DECISION_AGREEMENT_MIN = 0.9
MEAN_ABS_DEVIATION_MAX = 0.5


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman's rho, or None when either side is constant (no order to correlate)."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def decision_agreement(
    person: Sequence[float], judge: Sequence[float], *, min_gap: float = 1.0
) -> tuple[int, int]:
    """(pairs the judge orders as the person does, pairs the person separates by >= min_gap)."""
    agreed = total = 0
    for i in range(len(person)):
        for j in range(i + 1, len(person)):
            gap = person[i] - person[j]
            if abs(gap) >= min_gap:
                total += 1
                if (judge[i] - judge[j]) * gap > 0:
                    agreed += 1
    return agreed, total


def bootstrap_interval(
    a: Sequence[float],
    b: Sequence[float],
    stat: Callable[[Sequence[float], Sequence[float]], float | None],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float] | None:
    """95% percentile interval of ``stat`` over answers resampled with replacement.

    Resamples where the statistic is undefined (a constant side) are dropped; None when fewer
    than a hundredth of them are left, which says the statistic has no stable interval here."""
    rng = random.Random(seed)  # nosec B311 -- seeded resampling, not security
    idx = list(range(len(a)))
    vals: list[float] = []
    for _ in range(resamples):
        pick = [rng.choice(idx) for _ in idx]
        v = stat([a[i] for i in pick], [b[i] for i in pick])
        if v is not None:
            vals.append(v)
    if len(vals) < max(1, resamples // 100):
        return None
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


@dataclass(frozen=True)
class Agreement:
    """One judge against one person on the answers both graded."""

    n: int
    decision_agreed: int
    decision_pairs: int
    spearman: float | None
    spearman_interval: tuple[float, float] | None
    mean_abs_deviation: float
    bias: float  # mean judge - person; above zero, the judge grades higher

    @property
    def decision_rate(self) -> float | None:
        return self.decision_agreed / self.decision_pairs if self.decision_pairs else None

    def gate(self) -> tuple[bool, list[str]]:
        """Whether the judge passes the agreement gate, and why not."""
        why: list[str] = []
        rate = self.decision_rate
        if rate is None:
            why.append("the person's grades separate no pair of answers")
        elif rate < DECISION_AGREEMENT_MIN:
            why.append(f"decision agreement {rate:.2f} < {DECISION_AGREEMENT_MIN}")
        if self.mean_abs_deviation > MEAN_ABS_DEVIATION_MAX:
            why.append(
                f"mean absolute deviation {self.mean_abs_deviation:.2f} > {MEAN_ABS_DEVIATION_MAX}"
            )
        return (not why, why)


def agreement(
    person: Mapping[str, float],
    judge: Mapping[str, float],
    *,
    min_gap: float = 1.0,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Agreement:
    """Agreement on the answers both sides graded (keys present in both), in key order."""
    items = sorted(set(person) & set(judge))
    p = [float(person[i]) for i in items]
    j = [float(judge[i]) for i in items]
    agreed, pairs = decision_agreement(p, j, min_gap=min_gap)
    n = len(items)
    return Agreement(
        n=n,
        decision_agreed=agreed,
        decision_pairs=pairs,
        spearman=spearman(p, j),
        spearman_interval=bootstrap_interval(p, j, spearman, resamples=resamples, seed=seed),
        mean_abs_deviation=sum(abs(x - y) for x, y in zip(p, j, strict=True)) / n if n else 0.0,
        bias=sum(y - x for x, y in zip(p, j, strict=True)) / n if n else 0.0,
    )
