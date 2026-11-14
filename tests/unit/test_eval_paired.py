"""The paired comparison (spec ``docs/specs/eval-harness-chat.md`` § The statistic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from doc_assistant.eval.paired import (
    bootstrap_interval,
    compare_groups,
    format_paired,
)
from doc_assistant.eval.report import paired_runs
from doc_assistant.eval.results import EvalOutput, EvalResult, ScoreResult
from doc_assistant.eval.store import Store

BINARY = frozenset({"hold"})


def _row(run: str, case: str, scorer: str, value: float, family: str = "single", **kw: Any):
    return {
        "run_id": run,
        "case_id": case,
        "scorer_name": scorer,
        "value": value,
        "scoreable": kw.get("scoreable", True),
        "family": family,
    }


def test_interval_is_reproducible_from_its_seed() -> None:
    d = [0.1, -0.2, 0.3, 0.0, 0.25]
    assert bootstrap_interval(d, seed=1) == bootstrap_interval(d, seed=1)
    lo, hi = bootstrap_interval(d, seed=1)
    assert lo <= sum(d) / len(d) <= hi


def test_interval_edges() -> None:
    assert bootstrap_interval([]) == (0.0, 0.0)
    assert bootstrap_interval([0.4]) == (0.4, 0.4)


def test_a_clear_rise_reads_higher() -> None:
    a = [_row("a", f"c{i}", "cov", 0.2) for i in range(8)]
    b = [_row("b", f"c{i}", "cov", 0.8) for i in range(8)]
    (d,) = compare_groups(a, b, binary=BINARY).diffs
    assert d.n == 8 and d.diff == pytest.approx(0.6) and d.verdict == "higher at this n"


def test_a_clear_fall_reads_as_a_measured_drop() -> None:
    a = [_row("a", f"c{i}", "cov", 0.9) for i in range(8)]
    b = [_row("b", f"c{i}", "cov", 0.4) for i in range(8)]
    (d,) = compare_groups(a, b, binary=BINARY).diffs
    assert d.verdict == "a measured drop"


def test_mixed_differences_read_no_difference_at_this_n() -> None:
    vals = [0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.05, -0.05]
    a = [_row("a", f"c{i}", "cov", 0.5) for i in range(8)]
    b = [_row("b", f"c{i}", "cov", 0.5 + v) for i, v in enumerate(vals)]
    (d,) = compare_groups(a, b, binary=BINARY).diffs
    assert d.verdict == "no difference at this n"


def test_case_is_the_unit_trials_are_averaged_first() -> None:
    a = [_row("a1", "c", "cov", 0.0), _row("a2", "c", "cov", 1.0)]  # mean 0.5
    b = [_row("b1", "c", "cov", 0.5)]
    (d,) = compare_groups(a, b, binary=BINARY).diffs
    assert d.n == 1 and d.diff == pytest.approx(0.0)


def test_only_cases_scored_in_both_groups_are_paired() -> None:
    a = [_row("a", "c1", "cov", 0.5), _row("a", "c2", "cov", 0.5, scoreable=False)]
    b = [_row("b", "c1", "cov", 0.7), _row("b", "c2", "cov", 0.9), _row("b", "c3", "cov", 1)]
    (d,) = compare_groups(a, b, binary=BINARY).diffs
    assert d.n == 1


def test_families_are_never_pooled() -> None:
    a = [_row("a", "s", "cov", 0.5, "single"), _row("a", "h", "cov", 0.5, "hostile")]
    b = [_row("b", "s", "cov", 0.6, "single"), _row("b", "h", "cov", 0.4, "hostile")]
    fams = {d.family for d in compare_groups(a, b, binary=BINARY).diffs}
    assert fams == {"single", "hostile"}


def test_binary_scorers_are_counted_with_broke_and_fixed_cases() -> None:
    a = [
        _row("a1", "x", "hold", 1),
        _row("a2", "x", "hold", 1),
        _row("a1", "y", "hold", 0),
        _row("a1", "z", "hold", 1),
        _row("a2", "z", "hold", 0),  # flips across trials: not a hold
    ]
    b = [_row("b", "x", "hold", 0), _row("b", "y", "hold", 1), _row("b", "z", "hold", 1)]
    cmp = compare_groups(a, b, binary=BINARY)
    assert cmp.diffs == []
    (h,) = cmp.holds
    assert (h.n, h.held_a, h.held_b) == (3, 1, 2)
    assert h.broke == ["x"] and h.fixed == ["y", "z"]
    assert cmp.new_breaks == [("single", "hold", "x")]


def test_format_names_seed_and_wording() -> None:
    a = [_row("a", "c", "cov", 0.2), _row("a", "c", "hold", 1)]
    b = [_row("b", "c", "cov", 0.4), _row("b", "c", "hold", 0)]
    text = format_paired(compare_groups(a, b, binary=BINARY, seed=7))
    assert "seed 7" in text and "| single | cov |" in text and "| single | hold |" in text


def test_paired_runs_reads_the_store_with_families(tmp_path: Path) -> None:
    def _res(case: str, family: str, v: float) -> EvalResult:
        return EvalResult(
            case_id=case,
            output=EvalOutput(answer="x"),
            scores=[ScoreResult("citation_coverage", v), ScoreResult("route", 1.0)],
            family=family,
        )

    with Store(tmp_path / "e.duckdb") as store:
        a = store.persist_run(
            [_res("c1", "single", 0.2), _res("r1", "records", 0.5)], system_name="s"
        )
        b = store.persist_run(
            [_res("c1", "single", 0.6), _res("r1", "records", 0.5)], system_name="s"
        )
        cmp = paired_runs(store, [a], [b])
    fams = {(d.family, d.scorer_name) for d in cmp.diffs}
    assert fams == {("single", "citation_coverage"), ("records", "citation_coverage")}
    assert {h.scorer_name for h in cmp.holds} == {"route"}
