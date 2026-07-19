"""Unit tests for the semantic concept layer (pure core; no model/DB)."""

from __future__ import annotations

from doc_assistant.knowledge.concept_semantics import (
    abstract_candidates,
    extract_abstract,
    nearest_pairs,
)

ABSTRACT_MD = """<!-- page:1 -->

## **Dense Passage Retrieval**

**Some Authors**

## **Abstract**

We show retrieval can use _dense_ representations via a dual-encoder, beating BM25.[1]

## **1 Introduction**

body text that should not be captured.
"""


def test_extract_abstract_between_heading_and_next_section() -> None:
    a = extract_abstract(ABSTRACT_MD)
    assert a is not None
    assert "dense representations" in a
    assert "dual-encoder" in a and "BM25" in a
    assert "[1]" not in a  # footnote marker stripped
    assert "Introduction" not in a  # bounded at the next heading
    assert "*" not in a and "_" not in a  # markdown emphasis stripped


def test_extract_abstract_none_when_absent() -> None:
    assert extract_abstract("# My Notes\n\nNo abstract heading here, just prose.") is None


def test_abstract_candidates_prefers_repeated_multiword_phrase() -> None:
    cands = abstract_candidates(
        "Dense Passage Retrieval",
        "dense passage retrieval beats bm25; dense passage retrieval uses embeddings.",
        top_k=5,
    )
    assert "dense passage retrieval" in cands  # repeated multi-word phrase ranks in
    assert len(cands) <= 5


def test_abstract_candidates_empty_for_non_paper() -> None:
    assert abstract_candidates(None, None, top_k=5) == []


def test_nearest_pairs_filters_by_threshold_and_orders() -> None:
    labels = ["a", "b", "c"]
    vectors = [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]]  # a≈b, c orthogonal
    pairs = nearest_pairs(labels, vectors, threshold=0.9)
    assert [(p.label_a, p.label_b) for p in pairs] == [("a", "b")]
    assert pairs[0].cosine > 0.9


def test_nearest_pairs_empty_when_all_far() -> None:
    labels = ["a", "b"]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    assert nearest_pairs(labels, vectors, threshold=0.5) == []
