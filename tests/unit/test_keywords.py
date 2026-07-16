"""Unit tests for the deterministic TF-IDF keyword extractor (pure core, no DB/LLM)."""

from __future__ import annotations

from doc_assistant.keywords import (
    c_value_scores,
    candidate_terms,
    contrastive_keywords,
    corpus_band_keywords,
    tf_idf_keywords,
    tokenize,
    weirdness,
)


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


def test_candidate_terms_drops_venue_and_id_artifacts() -> None:
    # publisher / journal / repo / ID tokens are noise, and n-grams containing them go too
    tokens = ["7554", "elife", "connectome", "biorxiv", "pmid", "deeplabcut"]
    terms = set(candidate_terms(tokens, ngram_max=2, min_chars=3))
    assert "elife" not in terms
    assert "biorxiv" not in terms
    assert "7554 elife" not in terms  # the eLife DOI-registrant bigram
    assert "elife connectome" not in terms  # any n-gram touching a venue token
    # genuine domain terms survive
    assert "connectome" in terms
    assert "deeplabcut" in terms


def test_candidate_terms_keeps_domain_words_that_double_as_venues() -> None:
    # 'cell' / 'neuron' / 'nature' are journals but also real concepts — must NOT be filtered
    tokens = ["cell", "neuron", "nature", "membrane"]
    terms = set(candidate_terms(tokens, ngram_max=1, min_chars=3))
    assert {"cell", "neuron", "nature", "membrane"} <= terms


def test_candidate_terms_drops_repeated_token_ngrams() -> None:
    # "outflux outflux [outflux]" is an OCR artifact, not a keyphrase
    tokens = ["outflux", "outflux", "outflux", "retrieval"]
    terms = set(candidate_terms(tokens, ngram_max=3, min_chars=3))
    assert "outflux outflux" not in terms
    assert "outflux outflux outflux" not in terms
    assert "outflux" in terms  # the single token is still a legitimate candidate


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


def test_corpus_band_excludes_singletons_and_hubs() -> None:
    # "retrieval" is in all 3 docs (hub), "solo" in 1 (singleton), "bm25"/"dense" in 2 (band).
    doc_terms = {
        "d1": ["retrieval", "bm25", "solo"],
        "d2": ["retrieval", "bm25", "dense"],
        "d3": ["retrieval", "dense"],
    }
    picked = {k.term for k in corpus_band_keywords(doc_terms, min_df=2, max_df=2, top_k=10)}
    assert picked == {"bm25", "dense"}  # only the df==2 shared band survives
    # df=3 hub and df=1 singleton are both excluded — the two RG-001 failure modes.
    assert "retrieval" not in picked
    assert "solo" not in picked


def test_corpus_band_ranks_by_breadth_and_is_deterministic() -> None:
    doc_terms = {
        "d1": ["alpha", "alpha", "beta"],
        "d2": ["alpha", "beta"],
        "d3": ["alpha"],  # alpha df=3, beta df=2
    }
    picked = corpus_band_keywords(doc_terms, min_df=2, max_df=3, top_k=2)
    assert [k.term for k in picked] == ["alpha", "beta"]  # broader (df=3) ranks first
    again = corpus_band_keywords(doc_terms, min_df=2, max_df=3, top_k=2)
    assert [k.term for k in picked] == [k.term for k in again]  # byte-stable


# ---- R3: C-value nested discount + reference-corpus weirdness ---------------


def test_c_value_discounts_fully_nested_and_ranks_container_top() -> None:
    freqs = {
        "dense passage retrieval": 5,  # trigram, never nested
        "passage retrieval": 5,  # occurs only inside the trigram
        "passage": 5,  # only nested
        "dense": 5,  # only nested
        "retrieval": 8,  # 5 nested + 3 standalone
        "bm25": 4,  # standalone unigram, no nesting
    }
    c = c_value_scores(freqs)
    assert c["dense passage retrieval"] == 10.0  # log2(4) * (5 - 0)
    assert c["passage retrieval"] == 0.0  # fully nested → discounted
    assert c["passage"] == 0.0
    assert c["dense"] == 0.0
    assert c["retrieval"] > 0.0  # keeps its standalone substance
    assert c["dense passage retrieval"] > c["retrieval"]  # container outranks its unigram
    assert c["bm25"] == 4.0  # log2(2) * (4 - 0)


def test_weirdness_favors_domain_tokens_over_common_english() -> None:
    assert weirdness("bm25", ref_ceiling=8.0) == 8.0  # OOV technical token → the ceiling
    assert weirdness("the", ref_ceiling=8.0) < 1.0  # ubiquitous English word
    assert weirdness("retrieval", ref_ceiling=8.0) > weirdness("model", ref_ceiling=8.0)
    # A phrase is bounded by its most-common token (min over tokens).
    assert weirdness("neural bm25", ref_ceiling=8.0) == weirdness("neural", ref_ceiling=8.0)
    assert weirdness("", ref_ceiling=8.0) == 0.0


def test_contrastive_ranks_domain_over_common_and_drops_nested() -> None:
    doc_terms = {
        "d1": ["bm25", "bm25", "system", "dense passage retrieval", "passage retrieval"],
        "d2": ["bm25", "bm25", "system", "system", "dense passage retrieval", "passage retrieval"],
    }
    picked = contrastive_keywords(doc_terms, top_k=10, ref_ceiling=8.0, min_cvalue=0.0)
    terms = [k.term for k in picked]
    assert "passage retrieval" not in terms  # fully-nested fragment dropped (C-value gate)
    assert "dense passage retrieval" in terms
    assert "bm25" in terms
    assert terms.index("bm25") < terms.index("system")  # OOV domain token outranks common word


def test_contrastive_is_deterministic_and_respects_top_k() -> None:
    doc_terms = {"d1": ["bm25", "colbert", "specter2", "dense"], "d2": ["bm25", "colbert"]}
    a = contrastive_keywords(doc_terms, top_k=2, ref_ceiling=8.0, min_cvalue=0.0)
    b = contrastive_keywords(doc_terms, top_k=2, ref_ceiling=8.0, min_cvalue=0.0)
    assert [k.term for k in a] == [k.term for k in b]  # byte-stable
    assert len(a) == 2  # top_k honoured
