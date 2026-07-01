"""Unit tests for the deterministic TF-IDF keyword extractor (pure core, no DB/LLM)."""

from __future__ import annotations

from doc_assistant.keywords import candidate_terms, tf_idf_keywords, tokenize


def test_tokenize_casefolds_and_keeps_tech_tokens() -> None:
    toks = tokenize("BM25 and the Cross-Encoder use GPT-4, SPECTER2.")
    assert "bm25" in toks
    assert "cross-encoder" in toks  # internal hyphen preserved
    assert "gpt-4" in toks
    assert "specter2" in toks
    assert toks == [t.lower() for t in toks]  # everything case-folded


def test_candidate_terms_rejects_stopword_boundaries_and_short_terms() -> None:
    tokens = ["dense", "passage", "retrieval", "the", "model"]  # "the"/"model" are stopwords
    terms = set(candidate_terms(tokens, ngram_max=2, min_chars=3))
    assert terms == {
        "dense",
        "passage",
        "retrieval",
        "dense passage",
        "passage retrieval",
    }
    # No phrase containing a stopword ("the"/"model") survives.
    assert not any("the" in t.split() or "model" in t.split() for t in terms)


def test_candidate_terms_drops_pure_numeric_and_too_short() -> None:
    tokens = ["ab", "2024", "retrieval"]
    terms = set(candidate_terms(tokens, ngram_max=1, min_chars=3))
    assert "ab" not in terms  # too short
    assert "2024" not in terms  # no alphabetic character
    assert "retrieval" in terms


def test_tf_idf_ranks_distinctive_above_ubiquitous() -> None:
    # "retrieval" is in every doc (ubiquitous → low idf); "colbert" only in d1 (distinctive).
    doc_terms = {
        "d1": ["colbert", "colbert", "retrieval", "bm25"],
        "d2": ["bm25", "retrieval", "ranking"],
        "d3": ["retrieval", "dense", "dense"],
    }
    ranked = tf_idf_keywords(doc_terms, top_k=10)
    d1_terms = [k.term for k in ranked["d1"]]
    assert d1_terms[0] == "colbert"  # distinctive + high tf ranks first
    assert d1_terms[-1] == "retrieval"  # corpus-ubiquitous ranks last
    # df is corpus-wide, not per-doc.
    by_term = {k.term: k for k in ranked["d1"]}
    assert by_term["retrieval"].df == 3
    assert by_term["colbert"].df == 1


def test_tf_idf_is_deterministic_and_respects_top_k() -> None:
    doc_terms = {"d1": ["alpha", "beta", "beta", "gamma", "delta"]}
    first = tf_idf_keywords(doc_terms, top_k=2)["d1"]
    second = tf_idf_keywords(doc_terms, top_k=2)["d1"]
    assert [k.term for k in first] == [k.term for k in second]  # byte-stable
    assert len(first) == 2  # top_k honoured


def test_tf_idf_tie_breaks_by_term_ascending() -> None:
    # Single doc → identical idf; equal tf → identical score → tie broken by term asc.
    doc_terms = {"d1": ["zeta", "alpha", "mu"]}
    ranked = [k.term for k in tf_idf_keywords(doc_terms, top_k=3)["d1"]]
    assert ranked == ["alpha", "mu", "zeta"]
