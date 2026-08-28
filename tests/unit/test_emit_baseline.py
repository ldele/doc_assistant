"""The emitter's guards: one experiment in, one document out.

A baseline averages its trials. If the runs behind it are not the same experiment, the average
presents two experiments as one number — which is the failure the whole comparability layer exists
to name, so the emitter refuses rather than emitting a document that would then be quoted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.emit_baseline import agreed_settings, check_one_experiment, resolve_runs

from doc_assistant.eval.baseline_doc import baseline_settings
from doc_assistant.eval.results import EvalOutput, EvalResult, ScoreResult
from doc_assistant.eval.store import RunPrefixError, Store

BASE = {
    "n_cases": 1,
    "index_doc_count": 96,
    "index_doc_digest": "a" * 64,
    "llm_provider": "ollama",
    "llm_model": "llama3.1:8b",
    "embedding_model": "bge-base",
    "top_k": 10,
}


def _results() -> list[EvalResult]:
    return [
        EvalResult(
            case_id="q1",
            output=EvalOutput(answer="a", citations=["x.pdf"], raw={"query": "q"}),
            scores=[ScoreResult("citation_overlap", 1.0), ScoreResult("contains_all", 0.5)],
        )
    ]


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "eval.duckdb")


class TestOneExperiment:
    def test_identical_trials_raise_no_problem(self, store: Store) -> None:
        runs = [
            store.persist_run(_results(), system_name="t", config={**BASE, "trial_index": i})
            for i in range(3)
        ]

        assert check_one_experiment(store, runs) == []

    def test_a_generator_difference_is_reported(self, store: Store) -> None:
        """The 2026-08-15 shape: two runs that look like trials of one thing and are not."""
        a = store.persist_run(_results(), system_name="t", config=dict(BASE))
        b = store.persist_run(
            _results(), system_name="t", config={**BASE, "llm_model": "qwen2.5:3b"}
        )

        problems = check_one_experiment(store, [a, b])

        assert len(problems) == 1
        assert "llm_model" in problems[0]

    def test_only_unknown_settings_do_not_block_an_emit(self, store: Store) -> None:
        """Historical runs record almost nothing, and refusing to document them would leave the
        old results with no machine-readable record at all. The emitted document says what is
        missing instead — refusal is for runs known to differ."""
        runs = [
            store.persist_run(_results(), system_name="t", config={"embedding_model": "bge-base"})
            for _ in range(2)
        ]

        assert check_one_experiment(store, runs) == []


class TestAgreedSettings:
    def test_only_what_every_run_agrees_on_survives(self, store: Store) -> None:
        """A document must not print the first trial's value for something the trials disputed."""
        a = store.persist_run(_results(), system_name="t", config={**BASE, "top_k": 10})
        b = store.persist_run(_results(), system_name="t", config={**BASE, "top_k": 8})

        settings = agreed_settings(store, [a, b])

        assert "top_k" not in settings
        assert settings["llm_model"] == "llama3.1:8b"

    def test_a_key_missing_from_one_run_is_dropped(self, store: Store) -> None:
        a = store.persist_run(_results(), system_name="t", config=dict(BASE))
        b = store.persist_run(
            _results(), system_name="t", config={k: v for k, v in BASE.items() if k != "top_k"}
        )

        assert "top_k" not in agreed_settings(store, [a, b])

    def test_trial_bookkeeping_never_reaches_the_document(self, store: Store) -> None:
        """`trial_index` differs by design and `n_trials` is the document's own run count —
        neither describes what was measured."""
        runs = [
            store.persist_run(
                _results(), system_name="t", config={**BASE, "trial_index": i, "n_trials": 3}
            )
            for i in range(3)
        ]

        settings = agreed_settings(store, runs)

        assert "trial_index" not in settings
        assert "n_trials" not in settings

    def test_the_agreed_settings_are_what_the_document_carries(self, store: Store) -> None:
        """End to end: emit from two runs, read the settings back out of the markdown."""
        from doc_assistant.eval.baseline_doc import render_baseline

        runs = [
            store.persist_run(_results(), system_name="t", config=dict(BASE)) for _ in range(2)
        ]
        settings = agreed_settings(store, runs)

        document = render_baseline(
            title="t", date="2026-08-18", settings=settings, run_ids=runs, scores={}
        )

        assert baseline_settings(document) == settings


class TestResolveRuns:
    def test_runs_are_selected_by_exact_note(self, store: Store) -> None:
        """How a sweep tags its arms: every trial of one arm shares a note."""
        wanted = [
            store.persist_run(_results(), system_name="t", note="chunk-sweep | arm 2")
            for _ in range(2)
        ]
        store.persist_run(_results(), system_name="t", note="chunk-sweep | arm 3")

        assert sorted(resolve_runs(store, [], "chunk-sweep | arm 2")) == sorted(wanted)

    def test_an_unmatched_note_raises_rather_than_emitting_an_empty_document(
        self, store: Store
    ) -> None:
        with pytest.raises(RunPrefixError, match="No run carries the note"):
            resolve_runs(store, [], "no such note")

    def test_ids_resolve_by_prefix(self, store: Store) -> None:
        run_id = store.persist_run(_results(), system_name="t")

        assert resolve_runs(store, [run_id[:8]], None) == [run_id]
