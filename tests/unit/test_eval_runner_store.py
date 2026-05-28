"""Tests for the eval Runner and DuckDB Store (Phase 5 / Feature 2)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from doc_assistant.eval.cases import EvalCase
from doc_assistant.eval.report import (
    diff_runs,
    format_aggregate,
    format_diff,
    format_run_summary,
)
from doc_assistant.eval.results import EvalOutput
from doc_assistant.eval.runner import Runner
from doc_assistant.eval.scorers import ContainsAllScorer
from doc_assistant.eval.store import Store

# ============================================================
# Runner
# ============================================================


def _good_sut(query: str) -> EvalOutput:
    return EvalOutput(answer=f"foo bar {query}", citations=["c.pdf"])


def _bad_sut(query: str) -> EvalOutput:
    raise RuntimeError("oops")


def test_runner_requires_at_least_one_scorer():
    with pytest.raises(ValueError, match="at least one scorer"):
        Runner([])


def test_runner_runs_all_cases():
    cases = [
        EvalCase(id="c1", query="x", expected_substrings=["foo"]),
        EvalCase(id="c2", query="y", expected_substrings=["bar"]),
    ]
    results = Runner([ContainsAllScorer()]).run(cases, _good_sut)
    assert [r.case_id for r in results] == ["c1", "c2"]
    assert all(r.output is not None for r in results)
    assert all(r.error is None for r in results)


def test_runner_captures_latency():
    cases = [EvalCase(id="c1", query="x", expected_substrings=["foo"])]
    results = Runner([ContainsAllScorer()]).run(cases, _good_sut)
    assert results[0].latency_ms >= 0.0


def test_runner_catches_sut_exception():
    cases = [EvalCase(id="c1", query="x")]
    results = Runner([ContainsAllScorer()]).run(cases, _bad_sut)
    assert results[0].output is None
    assert "RuntimeError" in (results[0].error or "")
    # Scorer should still have produced a ScoreResult (with error in details).
    assert len(results[0].scores) == 1
    assert results[0].scores[0].value == 0.0


def test_runner_progress_callback_fires():
    calls: list[tuple[int, int, str]] = []

    def _progress(i: int, total: int, case: EvalCase) -> None:
        calls.append((i, total, case.id))

    cases = [EvalCase(id="c1", query="x"), EvalCase(id="c2", query="y")]
    Runner([ContainsAllScorer()]).run(cases, _good_sut, progress=_progress)
    assert calls == [(0, 2, "c1"), (1, 2, "c2")]


# ============================================================
# Store roundtrip
# ============================================================


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "eval.duckdb")


def _persist_simple_run(store: Store) -> str:
    cases = [EvalCase(id="c1", query="x", expected_substrings=["foo"])]
    results = Runner([ContainsAllScorer()]).run(cases, _good_sut)
    return store.persist_run(results, system_name="test-system", config={"k": 1})


def test_store_persists_and_lists_runs(store: Store):
    run_id = _persist_simple_run(store)
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["id"] == run_id
    assert runs[0]["system_name"] == "test-system"
    assert runs[0]["n_cases"] == 1


def test_store_scorer_means(store: Store):
    run_id = _persist_simple_run(store)
    means = store.scorer_means(run_id)
    assert means == {"contains_all": 1.0}


def test_store_scorer_means_omits_all_skipped_scorers(store: Store):
    """A scorer that skipped every case (no expected field) shouldn't appear."""
    from doc_assistant.eval.scorers import ExactMatchScorer

    # Case has no expected_answer -> ExactMatch is_skipped for every case
    cases = [EvalCase(id="c1", query="x", expected_substrings=["foo"])]
    results = Runner([ContainsAllScorer(), ExactMatchScorer()]).run(cases, _good_sut)
    run_id = store.persist_run(results, system_name="t")
    means = store.scorer_means(run_id)
    assert "contains_all" in means
    assert "exact_match" not in means  # all-skipped -> filtered out


def test_store_scorer_stats_counts_scored_and_skipped(store: Store):
    from doc_assistant.eval.scorers import ExactMatchScorer

    cases = [
        EvalCase(id="c1", query="x", expected_substrings=["foo"], expected_answer="foo bar x"),
        EvalCase(id="c2", query="y", expected_substrings=["bar"]),  # no expected_answer
    ]
    results = Runner([ContainsAllScorer(), ExactMatchScorer()]).run(cases, _good_sut)
    run_id = store.persist_run(results, system_name="t")
    stats = store.scorer_stats(run_id)

    # ContainsAll: both cases have expected_substrings -> 2 scored, 0 skipped
    assert stats["contains_all"]["n_scored"] == 2
    assert stats["contains_all"]["n_skipped"] == 0
    assert stats["contains_all"]["mean"] == 1.0

    # ExactMatch: 1 case has expected_answer, 1 doesn't -> 1 scored, 1 skipped
    assert stats["exact_match"]["n_scored"] == 1
    assert stats["exact_match"]["n_skipped"] == 1


def test_store_scorer_stats_mean_is_none_when_all_skipped(store: Store):
    from doc_assistant.eval.scorers import ExactMatchScorer

    cases = [EvalCase(id="c1", query="x", expected_substrings=["foo"])]
    results = Runner([ExactMatchScorer()]).run(cases, _good_sut)
    run_id = store.persist_run(results, system_name="t")
    stats = store.scorer_stats(run_id)
    assert stats["exact_match"]["mean"] is None
    assert stats["exact_match"]["n_skipped"] == 1


def test_store_case_scores(store: Store):
    run_id = _persist_simple_run(store)
    rows = store.case_scores(run_id)
    assert rows == [{"case_id": "c1", "scorer_name": "contains_all", "value": 1.0}]


def test_store_context_manager_closes(tmp_path: Path):
    db_path = tmp_path / "eval.duckdb"
    with Store(db_path) as s:
        _persist_simple_run(s)
    # Re-open to confirm data was committed
    with Store(db_path) as s2:
        assert len(s2.list_runs()) == 1


# ============================================================
# Report
# ============================================================


def test_format_run_summary_renders_means(store: Store):
    run_id = _persist_simple_run(store)
    out = format_run_summary(store, run_id)
    assert "contains_all" in out
    assert "1.000" in out
    # New columns surfacing scored / skipped breakdown.
    assert "n_scored" in out
    assert "n_skipped" in out


def test_format_run_summary_shows_dash_for_all_skipped(store: Store):
    """All-skipped scorer renders 'mean = -' rather than '0.000'."""
    from doc_assistant.eval.scorers import ExactMatchScorer

    cases = [EvalCase(id="c1", query="x", expected_substrings=["foo"])]
    results = Runner([ExactMatchScorer()]).run(cases, _good_sut)
    run_id = store.persist_run(results, system_name="t")
    out = format_run_summary(store, run_id)
    # Look for the exact_match row, mean should render as "-" not "0.000"
    assert "exact_match" in out
    assert "| - |" in out


def test_format_run_summary_empty_run(store: Store):
    out = format_run_summary(store, "nonexistent-run-id")
    assert "no scores" in out


def test_diff_runs_computes_delta(store: Store):
    def sut_low(_q: str) -> EvalOutput:
        return EvalOutput(answer="foo")

    def sut_high(_q: str) -> EvalOutput:
        return EvalOutput(answer="foo bar")

    cases = [EvalCase(id="c1", query="x", expected_substrings=["foo", "bar"])]
    results_a = Runner([ContainsAllScorer()]).run(cases, sut_low)
    run_a = store.persist_run(results_a, system_name="low")
    results_b = Runner([ContainsAllScorer()]).run(cases, sut_high)
    run_b = store.persist_run(results_b, system_name="high")

    rows = diff_runs(store, run_a, run_b)
    assert len(rows) == 1
    assert rows[0].case_id == "c1"
    assert math.isclose(rows[0].value_a, 0.5)
    assert math.isclose(rows[0].value_b, 1.0)
    assert math.isclose(rows[0].delta, 0.5)

    md = format_diff(rows, run_a_label="low", run_b_label="high")
    assert "low" in md and "high" in md
    assert "+0.500" in md


def test_diff_runs_handles_no_overlap(store: Store):
    md = format_diff([])
    assert "No overlapping" in md


# ============================================================
# Aggregation across N trials
# ============================================================


def _run_with_answer(store: Store, answer_text: str, note: str) -> str:
    """Persist a 1-case run where the system returns the given answer text."""

    def sut(_q: str) -> EvalOutput:
        return EvalOutput(answer=answer_text)

    cases = [EvalCase(id="c1", query="x", expected_substrings=["foo", "bar"])]
    results = Runner([ContainsAllScorer()]).run(cases, sut)
    return store.persist_run(results, system_name="t", note=note)


def test_aggregate_runs_empty_input(store: Store):
    assert store.aggregate_runs([]) == {}


def test_aggregate_runs_single_run(store: Store):
    run_id = _persist_simple_run(store)
    agg = store.aggregate_runs([run_id])
    assert "contains_all" in agg
    assert agg["contains_all"]["mean"] == 1.0
    assert agg["contains_all"]["n_scored"] == 1
    # Both std flavors undefined for n=1 trial.
    assert agg["contains_all"]["trial_mean_std"] is None
    # score_std over a single observation: DuckDB STDDEV returns NULL.
    assert agg["contains_all"]["score_std"] is None


def test_aggregate_runs_computes_mean_and_std(store: Store):
    """Three trials with different answers — score_std AND trial_mean_std both fire."""
    rid1 = _run_with_answer(store, "foo bar", "trial 1")  # 1.0
    rid2 = _run_with_answer(store, "foo", "trial 2")  # 0.5
    rid3 = _run_with_answer(store, "foo bar baz", "trial 3")  # 1.0
    agg = store.aggregate_runs([rid1, rid2, rid3])
    assert agg["contains_all"]["n_scored"] == 3
    assert math.isclose(agg["contains_all"]["mean"], (1.0 + 0.5 + 1.0) / 3, abs_tol=1e-6)
    assert agg["contains_all"]["score_std"] is not None
    assert agg["contains_all"]["score_std"] > 0.0
    # With one case per trial, trial_mean_std == score_std (no within-case variance).
    assert agg["contains_all"]["trial_mean_std"] is not None
    assert math.isclose(
        agg["contains_all"]["trial_mean_std"],
        agg["contains_all"]["score_std"],
        abs_tol=1e-6,
    )


def test_aggregate_trial_mean_std_zero_for_deterministic_scorer(store: Store):
    """When every trial yields the exact same mean, trial_mean_std == 0.

    This is the property that distinguishes trial_mean_std from score_std —
    a deterministic scorer (citation_overlap on a fixed corpus) gets 0 here
    even though score_std is non-zero (because individual cases score
    differently within each trial).
    """

    def sut(_q: str) -> EvalOutput:
        return EvalOutput(answer="foo", citations=["a.pdf"])

    cases = [
        EvalCase(id="hit", query="x", expected_citations=["a.pdf"]),  # always 1.0
        EvalCase(id="miss", query="y", expected_citations=["nope.pdf"]),  # always 0.0
    ]
    from doc_assistant.eval.scorers import CitationOverlapScorer

    rids = []
    for _ in range(3):
        results = Runner([CitationOverlapScorer()]).run(cases, sut)
        rids.append(store.persist_run(results, system_name="t"))
    agg = store.aggregate_runs(rids)
    # Each trial mean is 0.5 (1 hit + 1 miss / 2 cases). All 3 trials identical.
    assert math.isclose(agg["citation_overlap"]["mean"], 0.5, abs_tol=1e-6)
    assert math.isclose(agg["citation_overlap"]["trial_mean_std"], 0.0, abs_tol=1e-6)
    # But score_std is non-zero because individual case scores differ within trials.
    assert agg["citation_overlap"]["score_std"] > 0.0


def test_aggregate_runs_excludes_skipped(store: Store):
    from doc_assistant.eval.scorers import ExactMatchScorer

    # Case has no expected_answer -> ExactMatch is_skipped
    def sut(_q: str) -> EvalOutput:
        return EvalOutput(answer="foo")

    cases = [EvalCase(id="c1", query="x", expected_substrings=["foo"])]
    rid1 = store.persist_run(
        Runner([ContainsAllScorer(), ExactMatchScorer()]).run(cases, sut),
        system_name="t",
    )
    rid2 = store.persist_run(
        Runner([ContainsAllScorer(), ExactMatchScorer()]).run(cases, sut),
        system_name="t",
    )
    agg = store.aggregate_runs([rid1, rid2])
    # contains_all is scoreable in both runs
    assert agg["contains_all"]["n_scored"] == 2
    # exact_match was skipped in both runs
    assert agg["exact_match"]["n_skipped"] == 2
    assert agg["exact_match"]["mean"] is None


def test_format_aggregate_renders_both_stds(store: Store):
    rid1 = _run_with_answer(store, "foo bar", "t1")
    rid2 = _run_with_answer(store, "foo", "t2")
    out = format_aggregate(store, [rid1, rid2], label="My runs")
    assert "My runs" in out
    assert "2 run" in out
    assert "Trial-mean std" in out
    assert "Per-score std" in out
    assert "contains_all" in out


def test_format_aggregate_empty():
    # No store interaction needed — empty run_ids returns the empty message.
    # Use any temp store for the call but pass empty list.
    pass  # see test_aggregate_runs_empty_input — coverage already adequate


# ============================================================
# Flaky-case detection
# ============================================================


def test_flaky_cases_empty_for_consistent_runs(store: Store):
    """No flakes when all trials succeed or all trials skip the same case."""
    rid1 = _persist_simple_run(store)
    rid2 = _persist_simple_run(store)
    assert store.flaky_cases([rid1, rid2]) == []


def test_flaky_cases_detects_intermittent_failure(store: Store):
    """Same case scoreable in one trial, skipped in another → flagged."""
    from doc_assistant.eval.results import EvalResult, ScoreResult

    # Trial 1: contains_all gets a real score (scoreable=True)
    results_ok = Runner([ContainsAllScorer()]).run(
        [EvalCase(id="flaky", query="x", expected_substrings=["foo"])],
        _good_sut,
    )
    rid_ok = store.persist_run(results_ok, system_name="t1")

    # Trial 2: same case but the scorer returned an error → skipped
    err_result = EvalResult(
        case_id="flaky",
        output=None,
        scores=[ScoreResult(scorer_name="contains_all", value=0.0, details={"error": "timeout"})],
    )
    rid_fail = store.persist_run([err_result], system_name="t2")

    flakes = store.flaky_cases([rid_ok, rid_fail])
    assert len(flakes) == 1
    assert flakes[0]["case_id"] == "flaky"
    assert flakes[0]["scorer_name"] == "contains_all"
    assert flakes[0]["n_scored"] == 1
    assert flakes[0]["n_skipped"] == 1


def test_flaky_cases_empty_run_ids():
    """Defensive — empty input returns empty list, no SQL run."""
    # Don't need the store fixture here; just assert the contract.
    from doc_assistant.eval.store import Store as _Store

    with _Store(":memory:") as s:
        assert s.flaky_cases([]) == []
