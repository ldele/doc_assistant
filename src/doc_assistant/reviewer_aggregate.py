"""Reviewer aggregation & self-improvement loop (Phase 6 / Integrity Chunk 2c).

Reads the per-answer reviewer verdicts (Chunk 2b, ``answer_reviews``) and mines
them for *systematic* failure modes — but designed around the hazard that makes
this hard: the reviewer is a **biased sampler** (it runs only on already-flagged
answers) *and* an LLM with its own tilts, so a recurring ``failure_tag`` is
ambiguous by construction.

Two guards make a count meaningful:

* **Minimum-N gate.** A tag is not reported as actionable until it clears
  ``MIN_FAILURE_TAG_COUNT`` occurrences across ``MIN_FAILURE_TAG_DOCS`` distinct
  answer records. Below the gate the report reads "insufficient evidence", and
  every count is shown against its denominator — never bare.
* **Eval-anchored bias-vs-fault.** A recurring tag is split into *reviewer bias*
  vs *real system fault* by anchoring against the **verified** golden set: if the
  reviewer also assigns the tag to known-good golden answers, it is bias (fix the
  rubric); if it assigns it in production but not on the golden set, it is a real
  fault (fix retrieval/chunking/prompting). Without the anchor the split is
  unfalsifiable and is reported as "unanchored" — never as bias-vs-fault.

Architecture: read-only aggregation over the existing sidecar tables. No mutation
of the chunk store (Enrichment-Layer Pattern). This module is **instrumentation,
not action** — it surfaces the recurring fault; a human decides the fix (same
discipline as the no-auto-retry rule). The pure aggregation / classification /
formatting functions take plain rows so they unit-test without a DB; only
``load_review_tags`` touches SQLite.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select

from doc_assistant.config import MIN_FAILURE_TAG_COUNT, MIN_FAILURE_TAG_DOCS
from doc_assistant.db.models import AnswerRecord, AnswerReview
from doc_assistant.db.session import session_scope

# If the reviewer assigns a tag to at least this fraction of **known-good golden**
# answers, the tag is treated as a reviewer-bias signal rather than a real fault.
DEFAULT_BIAS_RATE = 0.2

# The "no dominant fault" tag is never itself an actionable finding.
NEUTRAL_TAG = "none"


@dataclass
class ReviewTagRow:
    """One reviewer verdict, reduced to what aggregation needs."""

    failure_tag: str
    answer_record_id: str
    prompt_version: str | None = None


@dataclass
class FailureTagStat:
    """Per-tag aggregate: raw count + the distinct-answer denominator for the gate."""

    tag: str
    count: int
    distinct_answers: int


@dataclass
class TagVerdict:
    """An above-gate tag, adjudicated against the golden anchor (or unanchored)."""

    tag: str
    prod_count: int
    prod_total: int
    golden_rate: float | None
    verdict: str  # "reviewer_bias" | "system_fault" | "unanchored"


# ============================================================
# Pure aggregation + gate + classification
# ============================================================


def aggregate_tags(rows: list[ReviewTagRow]) -> tuple[list[FailureTagStat], int]:
    """Count reviews per ``failure_tag``. Returns ``(stats, total_reviews)``.

    ``total_reviews`` is every row (the denominator for a tag's share). Each
    stat carries both the raw count and the number of *distinct* answer records
    that tag appeared on — the latter is what the min-N gate uses so a single
    re-reviewed answer can't manufacture a pattern. Sorted count-desc.
    """
    counts: dict[str, int] = defaultdict(int)
    answers: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        tag = row.failure_tag or NEUTRAL_TAG
        counts[tag] += 1
        answers[tag].add(row.answer_record_id)
    stats = [FailureTagStat(tag, counts[tag], len(answers[tag])) for tag in counts]
    stats.sort(key=lambda s: (-s.count, s.tag))
    return stats, len(rows)


def is_actionable(
    stat: FailureTagStat,
    *,
    min_count: int = MIN_FAILURE_TAG_COUNT,
    min_docs: int = MIN_FAILURE_TAG_DOCS,
) -> bool:
    """True if a tag clears the minimum-N gate (and isn't the neutral tag)."""
    return (
        stat.tag != NEUTRAL_TAG and stat.count >= min_count and stat.distinct_answers >= min_docs
    )


def golden_tag_rates(rows: list[ReviewTagRow]) -> dict[str, float]:
    """Per-tag rate over a golden-set reviewer pass: ``count(tag) / n_golden``.

    The anchor: a high rate on **known-good** answers means the reviewer
    over-assigns that tag (bias), not that the golden answers are bad.
    """
    if not rows:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.failure_tag or NEUTRAL_TAG] += 1
    n = len(rows)
    return {tag: c / n for tag, c in counts.items()}


def classify_bias_vs_fault(
    stats: list[FailureTagStat],
    prod_total: int,
    golden_rates: dict[str, float] | None,
    *,
    min_count: int = MIN_FAILURE_TAG_COUNT,
    min_docs: int = MIN_FAILURE_TAG_DOCS,
    bias_rate: float = DEFAULT_BIAS_RATE,
) -> list[TagVerdict]:
    """Adjudicate each above-gate tag as reviewer-bias vs system-fault.

    ``golden_rates`` is the reviewer's per-tag rate on the verified golden set
    (``golden_tag_rates``). ``None`` (no anchor run) → every verdict is
    ``"unanchored"`` — the roadmap forbids calling it bias-vs-fault without the
    anchor. A tag the reviewer also assigns to ≥ ``bias_rate`` of known-good
    golden answers is ``reviewer_bias``; otherwise ``system_fault``.
    """
    verdicts: list[TagVerdict] = []
    for stat in stats:
        if not is_actionable(stat, min_count=min_count, min_docs=min_docs):
            continue
        if golden_rates is None:
            verdict, gr = "unanchored", None
        else:
            gr = golden_rates.get(stat.tag, 0.0)
            verdict = "reviewer_bias" if gr >= bias_rate else "system_fault"
        verdicts.append(TagVerdict(stat.tag, stat.count, prod_total, gr, verdict))
    return verdicts


# ============================================================
# Markdown formatting (pure)
# ============================================================


def format_tag_report(
    stats: list[FailureTagStat],
    total: int,
    *,
    min_count: int = MIN_FAILURE_TAG_COUNT,
    min_docs: int = MIN_FAILURE_TAG_DOCS,
) -> str:
    """Markdown table of tag counts with denominators + the min-N gate verdict."""
    if total == 0:
        return "_No reviewer verdicts recorded yet — nothing to aggregate._"
    lines = [
        f"## Reviewer failure tags over {total} review(s)",
        "",
        f"_Gate: a tag is **actionable** only at ≥ {min_count} occurrences across "
        f"≥ {min_docs} distinct answers. Counts are shown against the {total}-review "
        "denominator, never bare._",
        "",
        "| failure_tag | count | / total | distinct answers | actionable |",
        "|---|---:|---:|---:|:---:|",
    ]
    for s in stats:
        share = f"{s.count}/{total}"
        actionable = "✅" if is_actionable(s, min_count=min_count, min_docs=min_docs) else "—"
        lines.append(f"| {s.tag} | {s.count} | {share} | {s.distinct_answers} | {actionable} |")
    if not any(is_actionable(s, min_count=min_count, min_docs=min_docs) for s in stats):
        lines += ["", "_No tag clears the gate → **insufficient evidence** to act._"]
    return "\n".join(lines)


def format_bias_vs_fault(verdicts: list[TagVerdict], *, golden_n: int | None) -> str:
    """Markdown table of the eval-anchored bias-vs-fault adjudication."""
    if not verdicts:
        return (
            "## Bias-vs-fault\n\n_No tag cleared the minimum-N gate — "
            "insufficient evidence to adjudicate._"
        )
    anchored = golden_n is not None
    header = (
        f"## Bias-vs-fault (anchored on {golden_n} golden review(s))"
        if anchored
        else "## Bias-vs-fault — ⚠ UNANCHORED"
    )
    lines = [header, ""]
    if not anchored:
        lines += [
            "_Run with `--anchor` to reviewer-score the verified golden set; "
            "without it, bias vs fault is unfalsifiable and is not asserted._",
            "",
        ]
    lines += [
        "| failure_tag | production | golden rate | verdict |",
        "|---|---:|---:|---|",
    ]
    label = {
        "reviewer_bias": "reviewer bias (fix rubric/prompt)",
        "system_fault": "system fault (fix retrieval/chunking/prompt)",
        "unanchored": "unanchored — cannot adjudicate",
    }
    for v in verdicts:
        gr = "—" if v.golden_rate is None else f"{v.golden_rate:.2f}"
        verdict_label = label.get(v.verdict, v.verdict)
        lines.append(f"| {v.tag} | {v.prod_count}/{v.prod_total} | {gr} | {verdict_label} |")
    return "\n".join(lines)


def format_by_prompt_version(rows: list[ReviewTagRow]) -> str:
    """Compact per-``prompt_version`` breakdown: reviews + dominant non-neutral tag."""
    by_pv: dict[str, list[ReviewTagRow]] = defaultdict(list)
    for row in rows:
        by_pv[row.prompt_version or "unknown"].append(row)
    if len(by_pv) <= 1:
        return ""  # nothing to compare across versions
    lines = [
        "## By prompt_version",
        "",
        "| prompt_version | reviews | dominant fault |",
        "|---|---:|---|",
    ]
    for pv in sorted(by_pv):
        stats, total = aggregate_tags(by_pv[pv])
        faults = [s for s in stats if s.tag != NEUTRAL_TAG]
        dominant = f"{faults[0].tag} ({faults[0].count}/{total})" if faults else "none"
        lines.append(f"| {pv} | {total} | {dominant} |")
    return "\n".join(lines)


# ============================================================
# DB read (impure)
# ============================================================


def load_review_tags() -> list[ReviewTagRow]:
    """Load successful reviewer verdicts as ``ReviewTagRow`` (failed reviews excluded).

    Joins ``answer_reviews`` to ``answer_records`` for the ``prompt_version``
    slice. A review with an ``error`` set never produced a valid tag, so it is
    dropped here rather than counted as ``none``.
    """
    with session_scope() as session:
        rows = session.execute(
            select(
                AnswerReview.failure_tag,
                AnswerReview.answer_record_id,
                AnswerRecord.prompt_version,
            )
            .join(AnswerRecord, AnswerReview.answer_record_id == AnswerRecord.id)
            .where(AnswerReview.error.is_(None))
        ).all()
    return [
        ReviewTagRow(
            failure_tag=r[0] or NEUTRAL_TAG,
            answer_record_id=str(r[1]),
            prompt_version=r[2],
        )
        for r in rows
    ]
