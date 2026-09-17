"""Tests for the Tier-2a deterministic floor
(``doc_assistant.knowledge.gaps.detect_unsourced_claims``)
over a fixed toy claim set — no DB, no LLM.
"""

from __future__ import annotations

from doc_assistant.knowledge.gaps import ClaimForGap, detect_unsourced_claims
from doc_assistant.synthesis import MARKER_OK, MARKER_UNSUPPORTED

_CONCEPTS = [("rag", "RAG"), ("bm25", "BM25")]
_ALIASES: dict[str, list[str]] = {}


def test_unsourced_claim_aggregation():
    claims = [
        ClaimForGap(id="c1", text="RAG improves retrieval.", marker=MARKER_UNSUPPORTED),
        ClaimForGap(id="c2", text="RAG is popular.", marker=MARKER_UNSUPPORTED),
        ClaimForGap(id="c3", text="BM25 is sparse.", marker=MARKER_UNSUPPORTED),
    ]
    gaps = detect_unsourced_claims(claims, _CONCEPTS, _ALIASES)
    by_concept = {g.concept_id: g for g in gaps}
    assert set(by_concept) == {"rag", "bm25"}
    assert by_concept["rag"].evidence.fact_ids == ("c1", "c2")
    assert by_concept["bm25"].evidence.fact_ids == ("c3",)
    assert all(g.tier == "t2a" and g.determinism == "deterministic" for g in gaps)
    assert all(g.kind == "unsourced_claim" for g in gaps)
    assert all(g.rating is None for g in gaps)


def test_cited_claims_produce_no_gap():
    claims = [ClaimForGap(id="c1", text="RAG improves retrieval.", marker=MARKER_OK)]
    assert detect_unsourced_claims(claims, _CONCEPTS, _ALIASES) == []


def test_no_unsupported_claims_is_a_no_op():
    assert detect_unsourced_claims([], _CONCEPTS, _ALIASES) == []


def test_unsupported_claim_matching_no_concept_produces_nothing():
    claims = [ClaimForGap(id="c1", text="completely unrelated prose", marker=MARKER_UNSUPPORTED)]
    assert detect_unsourced_claims(claims, _CONCEPTS, _ALIASES) == []


def test_matches_via_alias():
    claims = [
        ClaimForGap(id="c1", text="Dense passage retrieval helps.", marker=MARKER_UNSUPPORTED)
    ]
    aliases = {"rag": ["dense passage retrieval"]}
    gaps = detect_unsourced_claims(claims, _CONCEPTS, aliases)
    assert [g.concept_id for g in gaps] == ["rag"]
    assert gaps[0].evidence.fact_ids == ("c1",)


def test_pieces_that_cannot_be_claims_are_not_counted_as_unsourced():
    """KL1: `split_sentences` cuts after every "." — so "RAG has two parts:\n\n1." becomes a piece,
    and so do headings and the model's Sources block. On the working library 306 of 1,239
    "unsupported" pieces were these. Uncited, but asserting nothing, so not an unsourced claim."""
    claims = [
        ClaimForGap(
            id="lead", text="RAG consists of two main components:\n\n1.", marker=MARKER_UNSUPPORTED
        ),
        ClaimForGap(id="head", text="## What RAG solves", marker=MARKER_UNSUPPORTED),
        ClaimForGap(id="src", text="**Sources:** rag_lewis_2020.pdf", marker=MARKER_UNSUPPORTED),
        ClaimForGap(id="real", text="RAG reduces hallucination.", marker=MARKER_UNSUPPORTED),
    ]
    (gap,) = detect_unsourced_claims(claims, _CONCEPTS, _ALIASES)
    assert gap.concept_id == "rag"
    assert gap.evidence.fact_ids == ("real",)
