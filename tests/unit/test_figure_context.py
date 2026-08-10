"""Guard tests for figure→prose context (`find_figure_context`).

A figure chunk retrieves on its own text (caption + VLM description) but the model should
read it inside the passage that *uses* it — "as shown in Fig. 3, the activation collapses"
is what makes the figure mean something. Before 2026-08-09 a figure was `parent == child`:
self-contained, and citable with no surrounding argument.

Two rules, in order: **cited** (a passage refers to the figure) beats **placed** (the
passage carrying the caption — where the figure sits, which is weaker but honest).
"""

from __future__ import annotations

import pytest

from doc_assistant.ingest.figures import (
    figure_label,
    figure_parent_text,
    find_figure_context,
    reference_pattern,
)

CAPTION = "Figure 2.2. The original and revised oculomotor circuit."
CITING = "The loop was believed to involve the caudate. As shown in Fig. 2.2, GPi projects onward."
UNRELATED = "An introductory paragraph that mentions nothing numbered at all."


# ---- figure_label ----------------------------------------------------------


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("Figure 2.2. The revised circuit.", "2.2"),
        ("Fig. 3 Something happens", "3"),
        ("FIGURE 5.1: a plate", "5.1"),
        ("figure 12 — the tail", "12"),
        ("Figure 4b. A panel.", "4b"),
    ],
)
def test_label_is_read_from_the_caption(caption: str, expected: str) -> None:
    assert figure_label(caption) == expected


@pytest.mark.parametrize(
    "caption", [None, "", "A photograph of the apparatus", "Table 2. Results"]
)
def test_no_label_when_the_caption_does_not_carry_one(caption: str | None) -> None:
    # Without a label there is nothing to search the prose for; guessing would attach the
    # figure to an unrelated passage, which is worse than leaving it self-contained.
    assert figure_label(caption) is None


# ---- reference_pattern -----------------------------------------------------


@pytest.mark.parametrize(
    "prose",
    [
        "as shown in Fig. 2.2 the loop closes",
        "see Figure 2.2",
        "(Fig 2.2)",
        "compare Figures 2.1 and 2.2 for the revision",
        "in Figs. 2.1, 2.2 the circuits differ",
    ],
)
def test_pattern_matches_real_reference_forms(prose: str) -> None:
    assert reference_pattern("2.2").search(prose)


@pytest.mark.parametrize("prose", ["see Fig. 2.20 below", "Figure 2.21 shows", "value 2.2 mm"])
def test_pattern_does_not_match_a_longer_number_or_a_bare_value(prose: str) -> None:
    # Without the trailing guard, figure 2.2 would attach itself to figure 2.20's passage.
    assert reference_pattern("2.2").search(prose) is None


# ---- find_figure_context ---------------------------------------------------


def test_cited_passage_wins() -> None:
    parents = [UNRELATED, CITING, CAPTION]
    assert find_figure_context(CAPTION, parents) == (1, "cited")


def test_falls_back_to_where_the_figure_is_placed() -> None:
    # No prose refers to it, so the parent carrying the caption is the honest answer.
    parents = [UNRELATED, f"{CAPTION} Conventions as in 2.1."]
    assert find_figure_context(CAPTION, parents) == (1, "placed")


def test_the_caption_cannot_cite_itself() -> None:
    # The caption contains "Figure 2.2" too. If it were not stripped before searching,
    # every figure would match its own caption and "cited" would be unreachable.
    parents = [f"{CAPTION} Conventions as in Figure 2.1."]
    assert find_figure_context(CAPTION, parents) == (0, "placed")


def test_first_citing_parent_wins_when_several_refer_to_it() -> None:
    parents = [UNRELATED, CITING, "Later we return to Fig. 2.2 once more.", CAPTION]
    assert find_figure_context(CAPTION, parents) == (1, "cited")


def test_no_label_means_no_context() -> None:
    assert find_figure_context("A photograph of the rig", [CITING]) == (None, "none")


def test_label_present_but_never_mentioned() -> None:
    assert find_figure_context("Figure 9. Orphan.", [UNRELATED, CITING]) == (None, "none")


def test_whitespace_differences_do_not_defeat_caption_matching() -> None:
    # The caption stored on the Figure row comes from PDF text blocks; the markdown copy
    # has different line breaks. Matching on normalised whitespace is what makes the
    # "placed" fallback usable at all.
    wrapped = "Figure 2.2.  The original and\n   revised oculomotor circuit."
    assert find_figure_context(CAPTION, [UNRELATED, wrapped]) == (1, "placed")


def test_empty_parents_is_safe() -> None:
    assert find_figure_context(CAPTION, []) == (None, "none")


# ---- figure_parent_text ----------------------------------------------------


def test_figure_text_comes_first() -> None:
    # It is what matched the query. Burying it under the prose invites the model to answer
    # from the surrounding argument and cite the figure for it.
    out = figure_parent_text("FIG TEXT", "CONTEXT PROSE")
    assert out.startswith("FIG TEXT")
    assert "CONTEXT PROSE" in out


@pytest.mark.parametrize("context", [None, "", "   "])
def test_without_context_the_parent_is_just_the_figure(context: str | None) -> None:
    assert figure_parent_text("FIG TEXT", context) == "FIG TEXT"
