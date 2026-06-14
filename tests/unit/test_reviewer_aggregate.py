"""Tests for reviewer aggregation & bias-vs-fault (Phase 6 / Integrity Chunk 2c).

The aggregation, min-N gate, and bias-vs-fault classification are pure and
tested directly; ``load_review_tags`` is covered against a temp SQLite seeded
with answer_records + answer_reviews.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.reviewer_aggregate import (
    FailureTagStat,
    ReviewTagRow,
    aggregate_tags,
    classify_bias_vs_fault,
    format_bias_vs_fault,
    format_by_prompt_version,
    format_tag_report,
    golden_tag_rates,
    is_actionable,
    load_review_tags,
)


def _rows(*pairs: tuple[str, str]) -> list[ReviewTagRow]:
    """Build rows from (tag, answer_id) pairs."""
    return [ReviewTagRow(failure_tag=t, answer_record_id=a) for t, a in pairs]


# ============================================================
# aggregate_tags
# ============================================================


def test_aggregate_counts_and_distinct_answers():
    rows = _rows(
        ("overclaim", "a1"),
        ("overclaim", "a1"),  # same answer re-reviewed
        ("overclaim", "a2"),
        ("none", "a3"),
    )
    stats, total = aggregate_tags(rows)
    assert total == 4
    by_tag = {s.tag: s for s in stats}
    assert by_tag["overclaim"].count == 3
    assert by_tag["overclaim"].distinct_answers == 2  # a1 counted once
    assert by_tag["none"].count == 1
    # sorted count-desc
    assert stats[0].tag == "overclaim"


def test_aggregate_empty():
    stats, total = aggregate_tags([])
    assert stats == []
    assert total == 0


# ============================================================
# is_actionable — the min-N gate
# ============================================================


def test_gate_requires_both_count_and_docs():
    s = FailureTagStat("overclaim", count=10, distinct_answers=5)
    assert is_actionable(s, min_count=10, min_docs=5)
    assert not is_actionable(s, min_count=11, min_docs=5)  # below count
    assert not is_actionable(s, min_count=10, min_docs=6)  # below docs


def test_gate_never_fires_for_neutral_tag():
    s = FailureTagStat("none", count=999, distinct_answers=999)
    assert not is_actionable(s, min_count=1, min_docs=1)


# ============================================================
# bias-vs-fault classification
# ============================================================


def test_unanchored_when_no_golden_rates():
    stats = [FailureTagStat("overclaim", 10, 5)]
    verdicts = classify_bias_vs_fault(stats, 20, None, min_count=10, min_docs=5)
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "unanchored"
    assert verdicts[0].golden_rate is None


def test_reviewer_bias_when_golden_also_flags():
    stats = [FailureTagStat("no_hedge", 10, 5)]
    # reviewer tags 40% of known-good golden answers with no_hedge → bias
    verdicts = classify_bias_vs_fault(
        stats, 20, {"no_hedge": 0.4}, min_count=10, min_docs=5, bias_rate=0.2
    )
    assert verdicts[0].verdict == "reviewer_bias"


def test_system_fault_when_golden_is_clean():
    stats = [FailureTagStat("missing_citation", 10, 5)]
    # reviewer almost never flags golden with this tag → real fault
    verdicts = classify_bias_vs_fault(
        stats, 20, {"missing_citation": 0.0}, min_count=10, min_docs=5, bias_rate=0.2
    )
    assert verdicts[0].verdict == "system_fault"


def test_classify_skips_below_gate_tags():
    stats = [FailureTagStat("overclaim", 3, 2)]  # below default gate
    verdicts = classify_bias_vs_fault(stats, 5, {"overclaim": 0.5}, min_count=10, min_docs=5)
    assert verdicts == []


# ============================================================
# golden_tag_rates
# ============================================================


def test_golden_tag_rates():
    rows = _rows(("overclaim", "g1"), ("none", "g2"), ("none", "g3"), ("overclaim", "g4"))
    rates = golden_tag_rates(rows)
    assert rates["overclaim"] == 0.5
    assert rates["none"] == 0.5


def test_golden_tag_rates_empty():
    assert golden_tag_rates([]) == {}


# ============================================================
# Markdown formatting
# ============================================================


def test_format_tag_report_empty():
    assert "nothing to aggregate" in format_tag_report([], 0)


def test_format_tag_report_marks_actionable_and_denominator():
    stats = [FailureTagStat("overclaim", 10, 5), FailureTagStat("no_hedge", 2, 1)]
    out = format_tag_report(stats, 30, min_count=10, min_docs=5)
    assert "10/30" in out  # count against denominator
    assert "✅" in out  # overclaim clears the gate
    assert "—" in out  # no_hedge does not


def test_format_tag_report_insufficient_evidence():
    stats = [FailureTagStat("overclaim", 2, 1)]
    out = format_tag_report(stats, 5, min_count=10, min_docs=5)
    assert "insufficient evidence" in out.lower()


def test_format_bias_vs_fault_unanchored_warns():
    out = format_bias_vs_fault([], golden_n=None)
    assert "insufficient evidence" in out.lower()


def test_format_bias_vs_fault_renders_anchored_verdicts():
    stats = [FailureTagStat("missing_citation", 10, 5), FailureTagStat("no_hedge", 12, 6)]
    verdicts = classify_bias_vs_fault(
        stats,
        30,
        {"missing_citation": 0.0, "no_hedge": 0.5},
        min_count=10,
        min_docs=5,
    )
    out = format_bias_vs_fault(verdicts, golden_n=10)
    assert "anchored on 10" in out
    assert "system fault" in out  # missing_citation: clean on golden
    assert "reviewer bias" in out  # no_hedge: also flagged on golden
    assert "0.50" in out  # golden rate rendered


def test_format_by_prompt_version_single_is_empty():
    rows = [ReviewTagRow("overclaim", "a1", "v1"), ReviewTagRow("none", "a2", "v1")]
    assert format_by_prompt_version(rows) == ""


def test_format_by_prompt_version_multi():
    rows = [
        ReviewTagRow("overclaim", "a1", "v1"),
        ReviewTagRow("overclaim", "a2", "v1"),
        ReviewTagRow("none", "a3", "v2"),
    ]
    out = format_by_prompt_version(rows)
    assert "v1" in out and "v2" in out
    assert "overclaim" in out


# ============================================================
# load_review_tags (temp DB)
# ============================================================


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield tmp_path
    engine.dispose()


def test_load_review_tags_joins_prompt_version_and_excludes_errors(temp_db: Path):
    from doc_assistant.provenance import record_answer
    from doc_assistant.reviewer import ReviewResult, persist_review

    a1 = record_answer(query="q1", answer="a", retrieved_chunks=[], prompt_version="v1")
    a2 = record_answer(query="q2", answer="a", retrieved_chunks=[], prompt_version="v2")
    persist_review(a1, ReviewResult(failure_tag="overclaim"), reviewer_kind="llm_haiku")
    persist_review(a2, ReviewResult(failure_tag="no_hedge"), reviewer_kind="llm_haiku")
    # A failed review must be excluded (never produced a valid tag).
    persist_review(a2, ReviewResult(error="timeout"), reviewer_kind="llm_haiku")

    rows = load_review_tags()
    assert len(rows) == 2  # the error review is dropped
    tags = {r.failure_tag for r in rows}
    assert tags == {"overclaim", "no_hedge"}
    pvs = {r.prompt_version for r in rows}
    assert pvs == {"v1", "v2"}
