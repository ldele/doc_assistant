"""The store/report/CLI layer around comparability: resolving runs, case sets, and the report.

`test_eval_comparability.py` pins the judgement on two config dicts. This pins the parts that
touch stored runs — above all the **case-set** check, which exists because `n_cases` is a count
and not an identity: two runs of 35 cases can be two different sets of 35, and this project has
re-authored a case file in place more than once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.compare_runs import EXIT_BY_STATUS, _against_baseline, format_inventory

from doc_assistant.eval.comparability import State, Status
from doc_assistant.eval.report import compare_runs, format_comparability
from doc_assistant.eval.results import EvalOutput, EvalResult, ScoreResult
from doc_assistant.eval.store import RunPrefixError, Store

RECORDED = {
    "n_cases": 2,
    "index_doc_count": 96,
    "index_doc_digest": "a" * 64,
    "parent_chunk_size": 2000,
    "parent_chunk_overlap": 200,
    "child_chunk_size": 400,
    "child_chunk_overlap": 50,
    "baseline_chunk_size": 1000,
    "baseline_chunk_overlap": 200,
    "embedding_model": "bge-base",
    "use_parent_child": True,
    "use_multi_query": False,
    "top_k": 10,
    "candidate_k": 20,
    "bm25_weight": 0.4,
    "rerank_candidate_cap": 60,
    "llm_provider": "ollama",
    "llm_model": "llama3.1:8b",
    "judge_provider": "anthropic",
    "judge_model": "claude-haiku-4-5-20251001",
}


def _results(case_ids: list[str], *, value: float = 1.0) -> list[EvalResult]:
    return [
        EvalResult(
            case_id=cid,
            output=EvalOutput(answer="a", citations=["x.pdf"], raw={"query": "q"}),
            scores=[
                ScoreResult("citation_overlap", value),
                ScoreResult("contains_all", value),
            ],
        )
        for cid in case_ids
    ]


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "eval.duckdb")


class TestResolveRunId:
    def test_a_prefix_resolves_to_the_full_id(self, store: Store) -> None:
        run_id = store.persist_run(_results(["a"]), system_name="t", config=dict(RECORDED))

        assert store.resolve_run_id(run_id[:8]) == run_id
        assert store.resolve_run_id(run_id) == run_id

    def test_an_unmatched_prefix_raises(self, store: Store) -> None:
        store.persist_run(_results(["a"]), system_name="t")

        with pytest.raises(RunPrefixError, match="No run id starts with"):
            store.resolve_run_id("zzzzzzzz")

    def test_an_ambiguous_prefix_raises_rather_than_picking(self, store: Store) -> None:
        """Silently choosing one of two matches would answer about a run nobody named.

        Ids are written directly because real ones are UUIDs and will not share a prefix on
        demand; the same shortcut the store's backward-compatibility test uses.
        """
        for suffix in ("one", "two"):
            store.conn.execute(
                "INSERT INTO runs (id, started_at, finished_at, system_name, config_json, "
                "n_cases, note) VALUES (?, now(), now(), 't', '{}', 0, NULL)",
                [f"dup-run-{suffix}"],
            )

        with pytest.raises(RunPrefixError, match="runs start with"):
            store.resolve_run_id("dup-run")

    def test_an_empty_prefix_is_refused(self, store: Store) -> None:
        with pytest.raises(RunPrefixError, match="Empty run id"):
            store.resolve_run_id("   ")


class TestCaseSets:
    def test_the_same_questions_compare_clean(self, store: Store) -> None:
        a = store.persist_run(_results(["q1", "q2"]), system_name="t", config=dict(RECORDED))
        b = store.persist_run(_results(["q1", "q2"]), system_name="t", config=dict(RECORDED))

        comparison = compare_runs(store, a, b)

        assert comparison.status is Status.COMPARABLE
        case_set = next(d for d in comparison.differences if d.key == "case_set")
        assert case_set.state is State.SAME

    def test_different_questions_are_caught_even_when_the_count_matches(
        self, store: Store
    ) -> None:
        """The reason a count is not an identity. Both runs recorded ``n_cases: 2`` and every
        other setting is identical — only the questions differ, and nothing else would notice."""
        a = store.persist_run(_results(["q1", "q2"]), system_name="t", config=dict(RECORDED))
        b = store.persist_run(_results(["q3", "q4"]), system_name="t", config=dict(RECORDED))

        comparison = compare_runs(store, a, b)

        assert comparison.status is Status.NOT_COMPARABLE
        case_set = next(d for d in comparison.differences if d.key == "case_set")
        assert case_set.state is State.DIFFERENT
        assert "2 not in B" in str(case_set.value_a)

    def test_a_partial_overlap_still_counts_as_different(self, store: Store) -> None:
        a = store.persist_run(_results(["q1", "q2"]), system_name="t", config=dict(RECORDED))
        b = store.persist_run(_results(["q1", "q3"]), system_name="t", config=dict(RECORDED))

        assert compare_runs(store, a, b).status is Status.NOT_COMPARABLE

    def test_a_run_with_no_case_rows_is_unknown_not_a_vacuous_match(self, store: Store) -> None:
        """Two empty sets are equal, and calling that agreement would be a lie of arithmetic."""
        a = store.persist_run([], system_name="t", config=dict(RECORDED))
        b = store.persist_run(_results(["q1"]), system_name="t", config=dict(RECORDED))

        comparison = compare_runs(store, a, b)
        case_set = next(d for d in comparison.differences if d.key == "case_set")

        assert case_set.state is State.UNKNOWN
        assert comparison.status is Status.UNKNOWN

    def test_case_ids_are_sorted_and_complete(self, store: Store) -> None:
        run_id = store.persist_run(_results(["q2", "q1"]), system_name="t")

        assert store.case_ids(run_id) == ["q1", "q2"]
        assert store.case_ids("no-such-run") == []


class TestTheReport:
    def test_the_verdict_comes_before_the_evidence(self, store: Store) -> None:
        """A reader who stops after the first line must already have the answer."""
        a = store.persist_run(_results(["q1"]), system_name="t", config=dict(RECORDED))
        b = store.persist_run(
            _results(["q1"]), system_name="t", config={**RECORDED, "llm_model": "qwen2.5:3b"}
        )

        rendered = format_comparability(compare_runs(store, a, b))

        assert rendered.index("NOT COMPARABLE") < rendered.index("Differing settings")
        assert "llm_model" in rendered
        # The retrieval scorer survives the generator swap and the report says so.
        assert "| citation_overlap | ok |" in rendered

    def test_unrecorded_settings_get_their_own_section_with_the_no_backfill_note(
        self, store: Store
    ) -> None:
        a = store.persist_run(_results(["q1"]), system_name="t", config={"n_cases": 1})
        b = store.persist_run(_results(["q1"]), system_name="t", config={"n_cases": 1})

        rendered = format_comparability(compare_runs(store, a, b))

        assert "cannot be ruled out" in rendered
        assert "back-filled" in rendered


class TestTheInventory:
    def test_it_names_what_each_run_pins_and_flags_the_rest(self, store: Store) -> None:
        store.persist_run(_results(["q1"]), system_name="t", config=dict(RECORDED))
        store.persist_run(_results(["q1"]), system_name="t", config={"embedding_model": "bge"})

        rendered = format_inventory(store, limit=10)

        assert "ollama/llama3.1:8b" in rendered
        assert "96 docs" in rendered
        assert "not recorded" in rendered
        assert "1 of these 2 runs do not record their generator" in rendered

    def test_an_empty_store_says_so(self, store: Store) -> None:
        assert "No runs" in format_inventory(store, limit=10)

    def test_a_pipe_in_a_note_cannot_break_the_table(self, store: Store) -> None:
        """Notes are user text and the sweeps put pipes in them (``chunk-sweep | parent=...``)."""
        store.persist_run(_results(["q1"]), system_name="t", note="chunk-sweep | parent=2000")

        row = next(ln for ln in format_inventory(store, 10).splitlines() if "chunk-sweep" in ln)

        assert row.count("|") == 7  # 6 columns => 7 delimiters, none injected by the note


def test_unknown_does_not_exit_zero() -> None:
    """A caller wiring this into a check must not read "nobody recorded it" as "all clear"."""
    assert EXIT_BY_STATUS[Status.COMPARABLE] == 0
    assert EXIT_BY_STATUS[Status.UNKNOWN] != 0
    assert EXIT_BY_STATUS[Status.NOT_COMPARABLE] != 0


class TestAgainstACommittedBaseline:
    """`--against` checks a run against a baseline DOCUMENT, with no run store required.

    That is the gap: `data/eval.duckdb` is gitignored, so in a fresh clone the committed numbers
    exist and the runs behind them do not.
    """

    def _baseline(self, tmp_path: Path, settings: dict[str, object]) -> Path:
        from doc_assistant.eval.baseline_doc import render_baseline

        path = tmp_path / "baseline.md"
        path.write_text(
            render_baseline(
                title="t", date="2026-08-18", settings=settings, run_ids=["r"], scores={}
            ),
            encoding="utf-8",
        )
        return path

    def test_a_matching_run_exits_zero(self, store: Store, tmp_path: Path) -> None:
        run_id = store.persist_run(_results(["q1"]), system_name="t", config=dict(RECORDED))
        path = self._baseline(tmp_path, dict(RECORDED))

        assert _against_baseline(store, run_id[:8], path, ()) == 0

    def test_a_different_generator_exits_not_comparable(
        self, store: Store, tmp_path: Path
    ) -> None:
        run_id = store.persist_run(
            _results(["q1"]), system_name="t", config={**RECORDED, "llm_model": "qwen2.5:3b"}
        )
        path = self._baseline(tmp_path, dict(RECORDED))

        assert (
            _against_baseline(store, run_id[:8], path, ()) == EXIT_BY_STATUS[Status.NOT_COMPARABLE]
        )

    def test_a_baseline_without_provenance_is_unknown_not_ok(
        self, store: Store, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every hand-written baseline. The exit code must not read as agreement, and the output
        must blame the document rather than the run."""
        run_id = store.persist_run(_results(["q1"]), system_name="t", config=dict(RECORDED))
        path = tmp_path / "old.md"
        path.write_text("# An old baseline\n\nSetup: bge-base, 97 documents.\n", encoding="utf-8")

        code = _against_baseline(store, run_id[:8], path, ())

        assert code == EXIT_BY_STATUS[Status.UNKNOWN] != 0
        assert "carries no provenance block" in capsys.readouterr().out

    def test_a_missing_file_is_an_error_not_a_verdict(self, store: Store, tmp_path: Path) -> None:
        run_id = store.persist_run(_results(["q1"]), system_name="t", config=dict(RECORDED))

        assert _against_baseline(store, run_id[:8], tmp_path / "nope.md", ()) == 1
