"""Unit tests for stage-0 candidate ranking (pure core of `concept_curation`).

The load-bearing tests here are the *guards against over-eager exclusion*. Ranking exists to
order a human's review, not to filter for them — the 2026-07-18 trap in
`docs/specs/feature-concept-graph.md` is what auto-exclusion produces on a multi-domain corpus.
"""

from __future__ import annotations

from doc_assistant.concept_curation import (
    harvest_name_bigrams,
    rank_candidates,
)

_NO_NAMES = frozenset[str]()


def _rank(counts: dict[str, int], **kw: object) -> list:
    return rank_candidates(
        counts,
        promoted=kw.get("promoted", set()),  # type: ignore[arg-type]
        in_graph=kw.get("in_graph", set()),  # type: ignore[arg-type]
        name_bigrams=kw.get("name_bigrams", _NO_NAMES),  # type: ignore[arg-type]
    )


# --- ordering ------------------------------------------------------------------------------


def test_ranked_by_document_reach_descending() -> None:
    """A cross-document graph wants cross-document vocabulary — reach leads."""
    ranked = _rank({"rare": 1, "everywhere": 9, "some": 3})
    assert [c.name for c in ranked] == ["everywhere", "some", "rare"]


def test_ties_break_on_name_for_stable_output() -> None:
    ranked = _rank({"beta": 2, "alpha": 2, "gamma": 2})
    assert [c.name for c in ranked] == ["alpha", "beta", "gamma"]


def test_every_candidate_is_returned_never_dropped() -> None:
    """Ranking orders; it does not filter. Callers decide what to hide."""
    ranked = _rank({"a term": 1, "2015": 1, "another term": 4})
    assert len(ranked) == 3


# --- the exclusion guards (why this is ranking, not filtering) -------------------------------


def test_single_token_candidate_is_never_author_like() -> None:
    """THE guard: `bert` appears in 4 authors strings and `colbert` in 1 on the real corpus,
    because `documents.authors` holds whole citations *including titles*. A substring rule would
    drop two of the most important concepts in an IR corpus."""
    names = harvest_name_bigrams(["Bert Colbert and Jane Smith"])
    ranked = _rank({"bert": 4, "colbert": 1}, name_bigrams=names)
    assert all(not c.author_like for c in ranked)


def test_multi_token_candidate_matching_a_name_is_flagged_not_removed() -> None:
    names = harvest_name_bigrams(["Ziyang Wang and Jane Smith"])
    ranked = _rank({"ziyang wang": 1}, name_bigrams=names)
    assert ranked[0].author_like is True
    assert len(ranked) == 1  # flagged, still present


def test_a_one_document_candidate_still_ranks() -> None:
    """`pddl` is a legitimate 1-document concept — reach is a signal, not a gate."""
    ranked = _rank({"pddl": 1})
    assert ranked[0].name == "pddl"
    assert ranked[0].artifact is False


# --- signals -------------------------------------------------------------------------------


def test_artifact_flag_tracks_is_artifact() -> None:
    ranked = {c.name: c for c in _rank({"18653 v1": 2, "mrr 10": 2, "dense retrieval": 2})}
    assert ranked["18653 v1"].artifact is True
    assert ranked["mrr 10"].artifact is True
    assert ranked["dense retrieval"].artifact is False


def test_promoted_and_in_graph_flags() -> None:
    ranked = {
        c.name: c
        for c in _rank(
            {"BM25": 5, "speckles": 2, "fresh": 1},
            promoted={"BM25", "speckles"},
            in_graph={"BM25"},
        )
    }
    assert (ranked["BM25"].promoted, ranked["BM25"].in_graph) == (True, True)
    assert (ranked["speckles"].promoted, ranked["speckles"].in_graph) == (True, False)
    assert (ranked["fresh"].promoted, ranked["fresh"].in_graph) == (False, False)


# --- name harvesting -------------------------------------------------------------------------


def test_harvest_emits_both_name_orders() -> None:
    """The field mixes "Given Surname" and "Surname, Given"."""
    assert harvest_name_bigrams(["Jane Smith"]) == frozenset({"jane smith", "smith jane"})


def test_harvest_ignores_lowercase_runs() -> None:
    """Only capitalised pairs, so a citation's lowercase prose cannot contribute a bigram."""
    assert harvest_name_bigrams(["effective passage search over dense retrieval"]) == frozenset()


def test_harvest_tolerates_empty_and_none_rows() -> None:
    assert harvest_name_bigrams(["", "Jane Smith"]) == frozenset({"jane smith", "smith jane"})


def test_harvest_reads_multiple_documents() -> None:
    names = harvest_name_bigrams(["Jane Smith", "Omar Khatab and Matei Zaharia"])
    assert "jane smith" in names
    assert "omar khatab" in names
    assert "matei zaharia" in names
