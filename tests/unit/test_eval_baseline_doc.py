"""A baseline document must carry its own evidence — and admit when it does not.

The committed baselines are the project's reference results, and `data/eval.duckdb` is gitignored,
so in a fresh clone the numbers travel and the runs behind them do not. These tests pin the two
halves of the fix: a document emitted from the run record round-trips its settings, and a document
written before that existed parses to nothing rather than to false agreement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.eval.baseline_doc import (
    PROVENANCE_MARKER,
    baseline_settings,
    parse_provenance,
    render_baseline,
    table_drift,
)

SETTINGS = {
    "index_doc_count": 96,
    "index_doc_digest": "a" * 64,
    "n_cases": 35,
    "llm_provider": "ollama",
    "llm_model": "llama3.1:8b",
    "embedding_model": "bge-base",
    "child_chunk_size": 400,
}
SCORES = {
    "citation_overlap": {"mean": 0.936, "trial_mean_std": 0.0, "n_scored": 105, "n_skipped": 0},
    "contains_all": {"mean": 0.777, "trial_mean_std": 0.004, "n_scored": 105, "n_skipped": 0},
}


def _doc(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "title": "Sparse arm, private 35",
        "date": "2026-08-18",
        "settings": SETTINGS,
        "run_ids": ["run-a", "run-b", "run-c"],
        "scores": SCORES,
    }
    kwargs.update(overrides)
    return render_baseline(**kwargs)  # type: ignore[arg-type]


class TestRoundTrip:
    def test_the_settings_survive_render_and_parse(self) -> None:
        """The whole point: the document is the evidence, so what goes in must come back out."""
        assert baseline_settings(_doc()) == SETTINGS

    def test_the_block_carries_the_run_ids_and_trial_count(self) -> None:
        provenance = parse_provenance(_doc())

        assert provenance["run_ids"] == ["run-a", "run-b", "run-c"]
        assert provenance["n_trials"] == 3
        assert provenance["schema"] == 1

    def test_the_scores_are_recorded_too(self) -> None:
        """So a later reader can tell the document's own table from the run it is checking."""
        assert parse_provenance(_doc())["scores"]["contains_all"]["mean"] == 0.777


class TestTolerance:
    def test_a_document_without_a_block_parses_to_nothing(self) -> None:
        """Every baseline written before 2026-08-18. Reporting `{}` makes the comparison say
        *unknown*; raising would simply mean nobody ran the check."""
        assert parse_provenance("# A hand-written baseline\n\nSetup: bge-base, 97 docs.\n") == {}
        assert baseline_settings("# nothing here") == {}

    def test_a_malformed_block_parses_to_nothing_rather_than_half(self) -> None:
        """A half-read record is worse than an absent one, because it looks authoritative."""
        broken = PROVENANCE_MARKER + "\n```json\n{not valid json,,,}\n```\n"

        assert parse_provenance(broken) == {}

    def test_a_json_block_without_the_marker_is_not_provenance(self) -> None:
        """A baseline may quote JSON for other reasons — a config excerpt, an API payload."""
        text = '# Baseline\n\n```json\n{"settings": {"llm_model": "not-this"}}\n```\n'

        assert parse_provenance(text) == {}

    def test_a_block_that_is_not_an_object_is_rejected(self) -> None:
        assert parse_provenance(PROVENANCE_MARKER + "\n```json\n[1, 2, 3]\n```\n") == {}

    def test_every_committed_baseline_parses_without_raising(self) -> None:
        """Run against the real folder, because tolerance that is only tested on fixtures is a
        claim about fixtures. These are ~30 hand-written documents of varied shape."""
        folder = Path(__file__).resolve().parents[2] / "tests" / "eval" / "baselines"
        documents = sorted(folder.glob("*.md"))

        assert documents, "expected committed baselines to exist"
        for path in documents:
            assert isinstance(parse_provenance(path.read_text(encoding="utf-8")), dict)


class TestTheRenderedDocument:
    def test_it_marks_the_judgement_as_owed(self) -> None:
        """An emitted baseline is not a finished one. The caveats are what make it worth keeping,
        and no emitter can derive them."""
        assert "TODO" in _doc()

    def test_a_missing_setting_is_printed_as_not_recorded_not_dropped(self) -> None:
        """A baseline whose corpus is unknown has to say so on its face — the silence is exactly
        what made the 2026-06-06 sweep unauditable for two months."""
        rendered = _doc(settings={"n_cases": 10})

        assert "**not recorded**" in rendered
        assert "`index_doc_digest`" in rendered  # the row is present, not omitted

    def test_it_warns_when_the_corpus_or_generator_is_unpinned(self) -> None:
        rendered = _doc(settings={"n_cases": 10})

        assert "does not pin" in rendered
        assert "not as a reference to compare against" in rendered

    def test_a_fully_recorded_document_carries_no_such_warning(self) -> None:
        assert "does not pin" not in _doc()

    def test_the_results_table_lists_every_scorer(self) -> None:
        rendered = _doc()

        assert "`citation_overlap` | 0.936" in rendered
        assert "`contains_all` | 0.777" in rendered

    def test_notes_are_rendered_when_given(self) -> None:
        assert "a caveat worth keeping" in _doc(notes=["a caveat worth keeping"])

    @pytest.mark.parametrize("value", [None, 0, False])
    def test_falsy_setting_values_still_render(self, value: object) -> None:
        """`use_multi_query: False` and `index_doc_count: 0` are real recorded values; a truthiness
        test would print them as unrecorded and quietly misdescribe the run."""
        rendered = _doc(settings={"use_multi_query": value})

        assert "| `use_multi_query` |" in rendered
        assert "| `use_multi_query` | **not recorded** |" not in rendered


class TestTableDrift:
    """The table is what a person quotes; the block is what the checker reads. If they disagree,
    this layer has become a better-hidden version of the problem it exists to solve."""

    def test_a_freshly_emitted_document_does_not_drift(self) -> None:
        assert table_drift(_doc()) == []

    def test_a_hand_edited_number_is_caught(self) -> None:
        """Someone tidies the pretty table, or copies a document and edits it, while the block
        still carries the original."""
        tampered = _doc().replace("| 0.777 |", "| 0.850 |", 1)

        problems = table_drift(tampered)

        assert len(problems) == 1
        assert "contains_all" in problems[0]
        assert "0.850" in problems[0] and "0.777" in problems[0]

    def test_a_document_without_provenance_has_nothing_to_disagree_with(self) -> None:
        assert table_drift("# old\n\n| `contains_all` | 0.777 |\n") == []

    def test_no_committed_baseline_disagrees_with_its_own_block(self) -> None:
        """Runs over the real folder, and is **vacuous today on purpose**.

        None of the ~30 committed baselines carries a provenance block yet, so this asserts
        nothing right now — say so plainly rather than counting it as coverage. It exists for the
        first emitted baseline that someone later hand-edits: the table is what a person quotes
        and the block is what the checker reads, and the drift between them would otherwise be
        found by nobody.
        """
        folder = Path(__file__).resolve().parents[2] / "tests" / "eval" / "baselines"

        for path in sorted(folder.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            assert table_drift(text) == [], f"{path.name} disagrees with its provenance block"

    def test_a_scorer_the_block_does_not_name_is_ignored(self) -> None:
        """A document may show a derived or hand-computed row; only what the block claims is
        checked, because only that is what the block is answerable for."""
        extra = _doc() + "\n| `hand_computed` | 0.123 | - | - | - |\n"

        assert table_drift(extra) == []
