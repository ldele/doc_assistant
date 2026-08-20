"""Guard tests for the comparability layer — when two eval runs may be read against each other.

The layer encodes a piece of reasoning this project has written by hand at least twice (RG-029,
and the generator caveat in `chunking_sweep_private_2026-08-08.md`): a scorer that reads the
*retrieved documents* survives a generator swap, and a scorer that reads the *answer* does not.
These tests pin that asymmetry, the three-state rule that keeps "unrecorded" from reading as
"matched", and the two ways the layer refuses to be quietly wrong — an unknown scorer is assumed
to depend on everything, and a run-defining setting nobody classified is reported rather than
ignored.
"""

from __future__ import annotations

import pytest

from doc_assistant.eval.comparability import (
    IGNORED_KEYS,
    SETTING_STAGE,
    Difference,
    Stage,
    State,
    Status,
    compare,
    compare_configs,
    scorer_verdict,
    unclassified_keys,
)
from doc_assistant.eval.report import format_comparability
from doc_assistant.eval.run_settings import run_defining_settings

#: A fully-recorded run: every key the layer knows about, so a comparison against a copy of it
#: has nothing left unknown. Values are shaped like the real ones but are not asserted on.
FULL: dict[str, object] = {
    "n_cases": 35,
    "index_doc_count": 96,
    "index_doc_digest": "2251" + "0" * 60,
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

ALL_SCORERS = ["citation_overlap", "contains_all", "embedding_similarity", "llm_judge"]


def _status(comparison: object, scorer: str) -> Status:
    verdicts = {v.scorer_name: v.status for v in comparison.verdicts}  # type: ignore[attr-defined]
    return verdicts[scorer]


class TestTheGeneratorAsymmetry:
    """The RG-029 case: same corpus, same retrieval, different LLM."""

    def test_a_generator_swap_spares_retrieval_scores_and_invalidates_answer_scores(self) -> None:
        """The finding that took a human two days, as a computation.

        `citation_overlap` is computed from the retrieved documents before generation, so it is
        the signal to trust when the generator moved — and `contains_all` scores the generated
        text, so its 0.822-vs-0.777 is a measurement of two different models.
        """
        haiku = {**FULL, "llm_provider": "anthropic", "llm_model": "claude-haiku-4-5-20251001"}

        comparison = compare(FULL, haiku, ALL_SCORERS)

        assert _status(comparison, "citation_overlap") is Status.COMPARABLE
        assert _status(comparison, "contains_all") is Status.NOT_COMPARABLE
        assert _status(comparison, "embedding_similarity") is Status.NOT_COMPARABLE
        assert _status(comparison, "llm_judge") is Status.NOT_COMPARABLE
        assert comparison.status is Status.NOT_COMPARABLE

    def test_the_blocking_reason_names_the_model(self) -> None:
        haiku = {**FULL, "llm_model": "claude-haiku-4-5-20251001"}

        verdict = next(v for v in compare(FULL, haiku, ["contains_all"]).verdicts if v.blocking)

        assert "llm_model" in verdict.reason
        assert "llama3.1:8b" in verdict.reason


class TestTheStagePrefix:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("index_doc_digest", "deadbeef"),
            ("top_k", 8),
            ("child_chunk_size", 256),
            ("embedding_model", "specter2"),
            ("n_cases", 10),
        ],
    )
    def test_a_change_at_or_above_retrieval_invalidates_every_scorer(
        self, key: str, value: object
    ) -> None:
        """Retrieval feeds generation, so nothing downstream survives a retrieval change —
        including the retrieval scorers themselves."""
        other = {**FULL, key: value}

        comparison = compare(FULL, other, ALL_SCORERS)

        assert _status(comparison, "citation_overlap") is Status.NOT_COMPARABLE
        assert _status(comparison, "contains_all") is Status.NOT_COMPARABLE

    def test_the_judge_model_touches_only_the_judge(self) -> None:
        """Two runs can share a generator and still have been graded by different judges — which
        moves `llm_judge` and nothing else."""
        other = {**FULL, "judge_model": "claude-sonnet-4-5"}

        comparison = compare(FULL, other, ALL_SCORERS)

        assert _status(comparison, "llm_judge") is Status.NOT_COMPARABLE
        assert _status(comparison, "contains_all") is Status.COMPARABLE
        assert _status(comparison, "citation_overlap") is Status.COMPARABLE

    def test_identical_configs_are_comparable_on_every_scorer(self) -> None:
        comparison = compare(FULL, dict(FULL), ALL_SCORERS)

        assert comparison.status is Status.COMPARABLE
        assert comparison.by_state(State.DIFFERENT) == ()
        assert comparison.by_state(State.UNKNOWN) == ()


class TestUnrecordedIsNotMatched:
    """The three-state rule. Of the 75 runs in the live store, none pins its generator."""

    def test_two_historical_runs_are_unknown_not_comparable(self) -> None:
        """The shape of every pre-2026-08-15 pair: same recorded keys, same values, and the
        settings that decide the verdict are simply absent."""
        historical = {"embedding_model": "bge-base", "n_cases": 35, "scorers": ["contains_all"]}

        comparison = compare(historical, dict(historical), ALL_SCORERS)

        assert comparison.status is Status.UNKNOWN
        assert _status(comparison, "contains_all") is Status.UNKNOWN

    def test_a_known_difference_outranks_an_unknown_one(self) -> None:
        """Knowing a run measured something else is a stronger statement than not knowing."""
        a = {"llm_model": "llama3.1:8b", "n_cases": 35}
        b = {"llm_model": "claude-haiku-4-5-20251001", "n_cases": 35}

        assert _status(compare(a, b, ["contains_all"]), "contains_all") is Status.NOT_COMPARABLE

    def test_a_key_recorded_by_only_one_side_says_which(self) -> None:
        recorded_by_b_only = compare_configs({"n_cases": 10}, {"n_cases": 10, "top_k": 10})
        top_k = next(d for d in recorded_by_b_only if d.key == "top_k")

        assert top_k.state is State.UNKNOWN
        assert "not recorded by A" in top_k.detail

    def test_absent_keys_are_reported_rather_than_skipped(self) -> None:
        """The silence is the finding. A comparison that omitted the rows nobody recorded would
        reproduce exactly the table those runs printed before anyone knew they were incomparable.
        """
        differences = compare_configs({}, {})

        assert {d.key for d in differences} == set(SETTING_STAGE)
        assert all(d.state is State.UNKNOWN for d in differences)


class TestRefusalToBeQuietlyWrong:
    @pytest.mark.parametrize("moved", ["llm_model", "judge_model", "top_k", "index_doc_digest"])
    def test_an_unclassified_scorer_is_assumed_to_depend_on_everything(self, moved: str) -> None:
        """A scorer nobody has classified must not be waved through as comparable.

        Including the scoring *instruments*: a classified scorer answers only for its own judge,
        but an unknown one has no declared instrument, and silence is not the same as "no judge".
        """
        other = {**FULL, moved: "something-else"}

        verdict = scorer_verdict("brand_new_scorer", compare_configs(FULL, other))

        assert verdict.status is Status.NOT_COMPARABLE

    def test_a_classified_scorer_ignores_an_instrument_that_is_not_its_own(self) -> None:
        """The other half of the same rule: `contains_all` is not graded by the judge, so a judge
        swap says nothing about it. Without this, every answer scorer would inherit the judge."""
        other = {**FULL, "judge_model": "claude-sonnet-4-5"}

        assert scorer_verdict("contains_all", compare_configs(FULL, other)).status is (
            Status.COMPARABLE
        )

    def test_every_run_defining_setting_is_classified(self) -> None:
        """The guard that keeps this module in step with the record.

        A key added to `run_defining_settings` but not to `SETTING_STAGE` would be excluded from
        every verdict in silence, and comparisons would read cleaner than the evidence supports.
        This fails the moment the two drift apart.
        """
        unclassified = unclassified_keys(run_defining_settings())

        assert unclassified == (), (
            f"run_defining_settings() emits {unclassified} which comparability.py neither "
            "places in a stage nor ignores — add them to SETTING_STAGE or IGNORED_KEYS."
        )

    def test_the_runner_bookkeeping_keys_are_ignored_not_classified(self) -> None:
        """`trial_index` differs by design between the trials of one experiment — treating it as a
        dimension would make every multi-trial run incomparable with itself."""
        a = {**FULL, "trial_index": 0, "n_trials": 5}
        b = {**FULL, "trial_index": 3, "n_trials": 5}

        assert compare(a, b, ALL_SCORERS).status is Status.COMPARABLE
        assert "trial_index" in IGNORED_KEYS

    def test_no_shared_scorer_is_unknown_not_comparable(self) -> None:
        """Two runs with nothing in common to compare have not been shown to agree on anything."""
        assert compare(FULL, dict(FULL), []).status is Status.UNKNOWN


class TestExtraDifferences:
    def test_a_caller_supplied_difference_participates_in_the_verdict(self) -> None:
        """How the CLI feeds in the real case-set identity, which `n_cases` cannot express: two
        runs of 35 cases can be two different sets of 35."""
        case_set = Difference(
            "case_set", Stage.CASES, State.DIFFERENT, value_a="35 cases", value_b="35 others"
        )

        comparison = compare(FULL, dict(FULL), ALL_SCORERS, extra_differences=(case_set,))

        assert comparison.status is Status.NOT_COMPARABLE
        assert _status(comparison, "citation_overlap") is Status.NOT_COMPARABLE


class TestDeclaredVariation:
    """`varying` turns the question from "are these the same run?" into the one a sweep asks:
    did exactly the intended thing change? Two opposite failures fall out of it."""

    def test_the_declared_variable_stops_blocking(self) -> None:
        """A chunking sweep exists to read two arms that differ in chunk size against each
        other. Without this the tool would object to the experiment itself."""
        arm = {**FULL, "child_chunk_size": 256, "child_chunk_overlap": 32}

        comparison = compare(
            FULL, arm, ALL_SCORERS, varying=("child_chunk_size", "child_chunk_overlap")
        )

        assert comparison.status is Status.COMPARABLE
        assert comparison.ineffective_variation == ()

    def test_something_else_moving_still_blocks(self) -> None:
        """The useful half: the sweep's real risk is that something besides the grid moved."""
        arm = {**FULL, "child_chunk_size": 256, "index_doc_digest": "b" * 64}

        comparison = compare(FULL, arm, ALL_SCORERS, varying=("child_chunk_size",))

        assert comparison.status is Status.NOT_COMPARABLE
        assert [d.key for d in comparison.verdicts[0].blocking] == ["index_doc_digest"]

    def test_a_variable_that_did_not_move_is_the_ki41_shape(self) -> None:
        """The 2026-06-06 sweep drove its grid through env vars `.env` silently overwrote, so six
        arms re-ingested one configuration and the result read as "no config beats the default".
        Comparable runs, void experiment — which is why it is a separate field from `status`."""
        comparison = compare(FULL, dict(FULL), ALL_SCORERS, varying=("child_chunk_size",))

        assert comparison.status is Status.COMPARABLE  # the runs really are the same
        assert [d.key for d in comparison.ineffective_variation] == ["child_chunk_size"]
        assert comparison.ineffective_variation[0].state is State.SAME

    def test_a_variable_neither_run_recorded_is_also_ineffective(self) -> None:
        """You cannot show you varied what nobody wrote down — the KI-41 record, exactly."""
        historical = {"embedding_model": "bge-base", "n_cases": 35}

        comparison = compare(
            historical, dict(historical), ALL_SCORERS, varying=("child_chunk_size",)
        )

        assert [d.key for d in comparison.ineffective_variation] == ["child_chunk_size"]
        assert comparison.ineffective_variation[0].state is State.UNKNOWN

    def test_the_report_leads_with_the_void_experiment(self) -> None:
        rendered = format_comparability(
            compare(FULL, dict(FULL), ALL_SCORERS, varying=("child_chunk_size",))
        )

        assert "did not change" in rendered
        assert rendered.index("did not change") < rendered.index("| Scorer |")
