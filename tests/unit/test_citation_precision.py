"""A citation link must agree on the title before it is stored (KI-45).

The defect: `match_to_library`'s rule 2 matched on **first-author surname + publication year with
no title comparison at all**. On a 97-document corpus holding many same-year papers by common
surnames it fired constantly — **13 of 16 stored resolutions were false**, and one document had 9
of its 11 links pointing at two unrelated papers. In a research-integrity app that is the app
asserting the user owns a paper they do not.

The fix is one predicate — `resolution_is_credible` — used by every rule *and* by the read side,
so the three cannot drift apart again. Surname+year now only narrows the candidates; the title
decides.

Measured on the real library after the fix: 16 links → 41, the 12 false ones dropped, and every
surviving new link verified by hand.
"""

from __future__ import annotations

import pytest

from doc_assistant.ingest.citations import (
    FUZZY_TITLE_THRESHOLD,
    MIN_TITLE_WORD_COVERAGE,
    _title_similarity,
    _title_word_coverage,
    resolution_is_credible,
)


def credible(parsed: str | None, library: str | None, **kw) -> bool:
    return resolution_is_credible(
        parsed_title=parsed,
        parsed_doi=kw.get("parsed_doi"),
        library_title=library,
        library_doi=kw.get("library_doi"),
    )


# ============================================================
# The substitution the ratio cannot see
# ============================================================


def test_one_substituted_word_makes_it_a_different_paper():
    """The case that survived the first fix and had to be measured out.

    `SequenceMatcher` reads these as 0.91 — comfortably over the ratio threshold — because they
    differ in one word out of four. Comparing the sets of words is what notices.
    """
    a = "Bidirectional recurrent neural networks"
    b = "Relational recurrent neural networks"
    assert _title_similarity(a, b) >= FUZZY_TITLE_THRESHOLD, "the ratio alone would accept this"
    assert not credible(a, b), "but it is a different paper"


def test_the_other_measured_near_miss():
    a = "Recurrent neural network grammars"
    b = "RECURRENT NEURAL NETWORK REGULARIZATION"
    assert not credible(a, b)


def test_a_true_match_with_a_spelling_difference_survives():
    """`behavioural` vs `behavioral` must not be read as a substitution."""
    a = "Learnable latent embeddings for joint behavioural and neural analysis"
    b = "Learnable latent embeddings for joint behavioral and neural analysis"
    assert credible(a, b)


def test_a_hyphenation_difference_survives():
    a = "Multianimal pose estimation, identification and tracking with deeplabcut"
    b = "Multi-animal pose estimation, identification and tracking with DeepLabCut"
    assert credible(a, b)


def test_case_and_trailing_noise_survive():
    """The regex often leaves "In" or a venue fragment on the end of a reference title."""
    assert credible("Attention is all you need. In", "Attention Is All You Need")
    assert credible(
        "Deep residual learning for image recognition. In",
        "Deep Residual Learning for Image Recognition",
    )


# ============================================================
# The rules the fix must not break
# ============================================================


def test_an_exact_doi_decides_regardless_of_titles():
    """Two records of one paper can disagree on the title; a DOI cannot."""
    assert credible(
        "some mangled title",
        "an entirely different string",
        parsed_doi="10.1234/ABC",
        library_doi="10.1234/abc",
    )


def test_a_contained_title_is_still_accepted():
    """What a strict ratio misses: the regex prefixes a title with the author-list tail.

    Measured on the real corpus this recovered a true link scoring 0.78 and admitted none of the
    12 false ones.
    """
    ref = (
        "A., Lopes, G., et al. Real-time, low-latency closed-loop feedback using markerless "
        "posture tracking"
    )
    lib = "Real-time, low-latency closed-loop feedback using markerless posture tracking"
    assert credible(ref, lib)


def test_no_title_on_either_side_is_not_credible():
    """ "Unverifiable" is treated as "not credible" — the whole point of the fix."""
    assert not credible(None, "A real title")
    assert not credible("A real title", None)
    assert not credible(None, None)


def test_unrelated_papers_are_rejected():
    """The shape of the original bug: two same-year papers by the same surname."""
    assert not credible(
        "A review of graph neural networks and pretrained language models for knowledge graphs",
        "Cell class-specific long-range axonal projections of neurons in mouse whisker cortex",
    )


# ============================================================
# The coverage primitive
# ============================================================


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("alpha beta gamma", "alpha beta gamma", 1.0),
        ("alpha beta", "alpha beta gamma delta", 1.0),  # asymmetric: a prefix is fine
        ("alpha beta gamma delta", "zeta beta gamma delta", 0.75),  # one substitution
    ],
)
def test_coverage_is_the_shorter_titles_words_found_in_the_longer(a, b, expected):
    assert _title_word_coverage(a, b) == pytest.approx(expected)


def test_coverage_is_symmetric_in_its_arguments():
    """Which side is 'parsed' must not change the verdict."""
    a, b = "alpha beta", "alpha beta gamma delta"
    assert _title_word_coverage(a, b) == _title_word_coverage(b, a)


def test_an_empty_title_covers_nothing():
    assert _title_word_coverage("", "anything at all") == 0.0
    assert _title_word_coverage(None, "anything at all") == 0.0


def test_the_threshold_sits_in_the_measured_gap():
    """Not tuned to a target: every false link scored 0.75, every true one 0.88 or above."""
    assert 0.75 < MIN_TITLE_WORD_COVERAGE <= 0.88


# ============================================================
# The DOI rule matches literally (found while removing the per-reference table scans)
# ============================================================


def _candidate(doc_id: str, *, doi: str | None = None, title: str | None = None):
    from doc_assistant.ingest.citations import LibraryCandidate

    return LibraryCandidate(id=doc_id, title=title, authors=None, year=None, doi=doi)


def _parsed(*, doi: str | None = None, title: str | None = None):
    from doc_assistant.ingest.citations import ParsedCitation

    return ParsedCitation(
        raw_text="",
        doi=doi,
        title=title,
        authors=None,
        year=None,
        extraction_method="test",
        confidence=1.0,
    )


def test_a_doi_containing_an_underscore_matches_only_itself():
    """The DOI rule was `Document.doi.ilike(parsed.doi)`, and `ilike` reads `_` as a wildcard.

    DOIs legitimately contain underscores, so a reference to `10.1234/abc_def` could resolve to
    `10.1234/abcXdef` — a different paper, asserted as one the user owns. Same defect class as
    the eval store's run-id prefix, in a different table.
    """
    from doc_assistant.ingest.citations import match_to_library

    library = [_candidate("wrong-paper", doi="10.1234/abcXdef")]
    assert match_to_library(_parsed(doi="10.1234/abc_def"), candidates=library) is None

    library.append(_candidate("right-paper", doi="10.1234/abc_def"))
    assert match_to_library(_parsed(doi="10.1234/abc_def"), candidates=library) == "right-paper"


def test_a_doi_containing_a_percent_matches_only_itself():
    from doc_assistant.ingest.citations import match_to_library

    library = [_candidate("wrong-paper", doi="10.1234/abcANYTHINGdef")]
    assert match_to_library(_parsed(doi="10.1234/abc%def"), candidates=library) is None


def test_the_doi_comparison_is_still_case_insensitive():
    """`ilike` was case-insensitive and DOIs are case-insensitive by spec — that must survive."""
    from doc_assistant.ingest.citations import match_to_library

    library = [_candidate("paper", doi="10.1234/ABC")]
    assert match_to_library(_parsed(doi="10.1234/abc"), candidates=library) == "paper"


def test_a_preloaded_library_answers_exactly_as_a_self_read_one_would():
    """`candidates` is an optimisation; it must not be a second matcher."""
    from doc_assistant.ingest.citations import match_to_library

    library = [
        _candidate("a", title="Attention is all you need"),
        _candidate("b", title="Deep residual learning for image recognition"),
    ]
    hit = match_to_library(_parsed(title="Attention is all you need"), candidates=library)
    assert hit == "a"
    miss = match_to_library(
        _parsed(title="An entirely unrelated paper about mouse whisker cortex"),
        candidates=library,
    )
    assert miss is None
