"""The judge-agreement gate (spec eval-harness-chat.md, H5; TESTING.md, the local judge)."""

from __future__ import annotations

from doc_assistant.eval.judge_agreement import (
    agreement,
    decision_agreement,
    spearman,
)

PERSON = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}


def test_a_judge_that_grades_like_the_person_passes() -> None:
    res = agreement(PERSON, dict(PERSON), resamples=200)
    assert res.n == 5
    assert res.decision_rate == 1.0
    assert res.spearman == 1.0
    assert res.mean_abs_deviation == 0.0
    assert res.gate() == (True, [])


def test_a_judge_that_reverses_the_person_fails_on_decisions() -> None:
    judge = {k: 6.0 - v for k, v in PERSON.items()}
    res = agreement(PERSON, judge, resamples=200)
    assert res.spearman == -1.0
    assert res.decision_rate == 0.0
    ok, why = res.gate()
    assert not ok
    assert any("decision agreement" in w for w in why)


def test_a_judge_that_gives_every_answer_the_same_grade_has_no_correlation() -> None:
    judge = dict.fromkeys(PERSON, 5.0)
    res = agreement(PERSON, judge, resamples=200)
    assert res.spearman is None
    assert res.spearman_interval is None
    assert res.decision_rate == 0.0  # a judge tie is not the person's decision
    assert res.bias == 2.0


def test_right_order_but_far_off_fails_on_deviation() -> None:
    judge = {k: min(5.0, v + 0.9) for k, v in PERSON.items()}
    judge["e"] = 5.0
    judge["d"] = 4.95
    ok, why = agreement(PERSON, judge, resamples=200).gate()
    assert not ok
    assert any("deviation" in w for w in why)


def test_only_answers_both_graded_count() -> None:
    res = agreement({**PERSON, "x": 3.0}, {**PERSON, "y": 1.0}, resamples=200)
    assert res.n == 5


def test_pairs_closer_than_the_gap_are_not_decisions() -> None:
    assert decision_agreement([1.0, 1.5, 3.0], [1.0, 2.0, 3.0], min_gap=1.0) == (2, 2)


def test_ties_share_a_rank() -> None:
    assert spearman([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) is not None


def test_the_interval_is_reproducible_from_its_seed() -> None:
    judge = {"a": 2.0, "b": 1.0, "c": 3.0, "d": 5.0, "e": 4.0}
    one = agreement(PERSON, judge, resamples=500, seed=7).spearman_interval
    two = agreement(PERSON, judge, resamples=500, seed=7).spearman_interval
    assert one == two
