"""Unit tests for the deterministic TF-IDF keyword extractor (pure core, no DB/LLM)."""

from __future__ import annotations

from doc_assistant.knowledge.keywords import (
    ScoredKeyword,
    c_value_scores,
    candidate_terms,
    cited_authors,
    contrastive_keywords,
    corpus_band_keywords,
    is_citation_artifact,
    majority,
    split_pages,
    strip_page_furniture,
    strip_reference_section,
    strip_urls,
    suppress_nested,
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


# --------------------------------------------------------------------------- #
# D1 — page furniture (REVIEW 2026-08-12 §2b R2, PLAN_2026-08-11 §1.2)
# --------------------------------------------------------------------------- #


def _paged(*pages: str) -> str:
    """Cached-markdown shape: a `<!-- page:N -->` marker before each page block."""
    return "\n".join(f"<!-- page:{i + 1} -->\n{p}" for i, p in enumerate(pages))


def test_split_pages_returns_one_block_when_there_are_no_markers() -> None:
    # EPUB/HTML/MD have no page markers — "repeats across pages" is unanswerable, so nothing
    # may be guessed. One block is what makes strip_page_furniture a no-op there.
    assert split_pages("just text\nover two lines") == ["just text\nover two lines"]


def test_split_pages_splits_on_markers() -> None:
    assert len(split_pages(_paged("a", "b", "c"))) == 4  # preamble + 3 pages


def test_strip_page_furniture_removes_a_running_header() -> None:
    """The measured nihms-66884 defect: a PMC stamp on every page beat the paper's own subject."""
    header = "Exp Brain Res. Author manuscript; available in PMC 2008 September 26."
    text = _paged(
        f"{header}\nMotor commands drive the reach.",
        f"{header}\nSensory feedback corrects the trajectory.",
        f"{header}\nCursor control was measured.",
        f"{header}\nMotor commands again.",
    )
    out = strip_page_furniture(text)
    assert "Author manuscript" not in out
    assert "Motor commands drive the reach." in out
    assert "Sensory feedback corrects the trajectory." in out


def test_strip_page_furniture_is_digit_blind_so_page_numbers_are_one_footer() -> None:
    """ "Page 1 of 9" and "Page 2 of 9" are one running footer, not nine distinct lines."""
    text = _paged(
        "Page 1 of 4\nreal content one",
        "Page 2 of 4\nreal content two",
        "Page 3 of 4\nreal content three",
        "Page 4 of 4\nreal content four",
    )
    out = strip_page_furniture(text)
    assert "Page" not in out
    assert "real content one" in out


def test_strip_page_furniture_keeps_a_line_that_repeats_on_only_a_few_pages() -> None:
    """Below the threshold nothing is removed — a section heading reused twice is not furniture."""
    text = _paged(
        "Methods\nalpha",
        "Methods\nbeta",
        "Results\ngamma",
        "Discussion\ndelta",
        "Conclusion\nepsilon",
        "Appendix\nzeta",
    )
    out = strip_page_furniture(text)
    assert "Methods" in out


def test_strip_page_furniture_is_a_noop_without_page_markers() -> None:
    text = "same line\nsame line\nsame line\nsame line\nsame line"
    assert strip_page_furniture(text) == text


def test_strip_page_furniture_is_a_noop_below_the_page_floor() -> None:
    """Two pages cannot establish repetition; refusing to guess beats stripping real content."""
    text = _paged("shared\nalpha", "shared\nbeta")
    assert strip_page_furniture(text) == text


def test_strip_page_furniture_survives_an_empty_document() -> None:
    assert strip_page_furniture("") == ""


# --------------------------------------------------------------------------- #
# D2 — shingle suppression
# --------------------------------------------------------------------------- #


def _kw(term: str, score: float) -> ScoredKeyword:
    return ScoredKeyword(term=term, score=score, tf=1, df=1)


def test_suppress_nested_keeps_only_the_best_of_an_overlapping_family() -> None:
    """The measured transformer_vaswani_2017 defect: 9 of 15 slots on one figure artifact."""
    scored = [
        _kw("eos pad br", 9.0),
        _kw("eos pad", 8.0),
        _kw("eos", 7.0),
        _kw("pad br", 6.0),
        _kw("pad", 5.0),
        _kw("dot-product attention", 4.0),
        _kw("sub-layers", 3.0),
    ]
    out = [k.term for k in suppress_nested(scored, top_k=5)]
    assert out == ["eos pad br", "dot-product attention", "sub-layers"]


def test_suppress_nested_drops_a_superspan_of_an_accepted_term_too() -> None:
    """Both directions: taking `eos` first must still exclude `eos pad br`, or the fragments
    would simply pull their own containers in behind them."""
    scored = [_kw("eos", 9.0), _kw("eos pad br", 8.0), _kw("attention", 7.0)]
    assert [k.term for k in suppress_nested(scored, top_k=5)] == ["eos", "attention"]


def test_suppress_nested_keeps_terms_that_merely_share_a_word() -> None:
    """Overlap means *contiguous span*, not "shares a token" — these are distinct concepts."""
    scored = [_kw("attention heads", 3.0), _kw("multi-head attention", 2.0), _kw("attention", 1.0)]
    out = [k.term for k in suppress_nested(scored, top_k=5)]
    assert out == ["attention heads", "multi-head attention"]


def test_suppress_nested_respects_top_k_and_rank_order() -> None:
    scored = [_kw("alpha", 5.0), _kw("beta", 4.0), _kw("gamma", 3.0), _kw("delta", 2.0)]
    assert [k.term for k in suppress_nested(scored, top_k=2)] == ["alpha", "beta"]


def test_suppress_nested_on_an_empty_list_returns_empty() -> None:
    assert suppress_nested([], top_k=15) == []


def test_tf_idf_keywords_applies_the_suppression() -> None:
    """The wiring, not just the helper: the shipped per_doc path must not emit shingles."""
    doc_terms = {
        "d1": ["eos pad br"] * 9 + ["eos pad"] * 9 + ["eos"] * 9 + ["attention"] * 3,
        "d2": ["unrelated"] * 4,
    }
    terms = [k.term for k in tf_idf_keywords(doc_terms, top_k=5)["d1"]]
    assert "attention" in terms
    # At most one member of the eos/pad family survives.
    assert sum(1 for t in terms if "eos" in t or "pad" in t) == 1


# --------------------------------------------------------------------------- #
# D5 — the tokeniser stops truncating designators
# --------------------------------------------------------------------------- #


def test_tokenize_keeps_a_designator_whole() -> None:
    """The measured D5 defect: `16p11.2` became `16p11` and `C57BL/6` became `c57bl` —
    silently renaming a chromosomal locus and a mouse strain."""
    toks = tokenize("The 16p11.2 deletion in C57BL/6 mice")
    assert "16p11.2" in toks
    assert "c57bl/6" in toks
    assert "16p11" not in toks
    assert "c57bl" not in toks


def test_tokenize_still_splits_prose_abbreviations_and_urls() -> None:
    """A separator followed by a LETTER is prose, not a designator — the whole of the rule."""
    toks = tokenize("see e.g. the paper at arxiv.org/abs/1706 and i.e. this one")
    assert "e.g" not in toks
    assert "i.e" not in toks
    assert "arxiv.org" not in toks
    assert "arxiv" in toks  # split back apart, where VENUE_STOPWORDS can see it


def test_tokenize_keeps_the_existing_tech_token_behaviour() -> None:
    toks = tokenize("BM25 and the Cross-Encoder use GPT-4, SPECTER2.")
    assert {"bm25", "cross-encoder", "gpt-4", "specter2"} <= set(toks)


def test_tokenize_does_not_glue_a_sentence_across_a_full_stop() -> None:
    assert "model.the" not in tokenize("we train the model. the next section")


def test_candidate_terms_drops_a_token_whose_head_is_a_stopword() -> None:
    """`fig.2` survives tokenisation as one token now, so the plain stopword check misses it."""
    terms = set(
        candidate_terms(tokenize("fig.2 shows the 16p11.2 locus"), ngram_max=1, min_chars=3)
    )
    assert "fig.2" not in terms
    assert "16p11.2" in terms


# --------------------------------------------------------------------------- #
# D4 — citation artifacts, by shape and by structure (never by name)
# --------------------------------------------------------------------------- #


def test_is_citation_artifact_matches_year_suffixes_and_article_ids() -> None:
    assert is_citation_artifact("2014a")
    assert is_citation_artifact("2015b")
    assert is_citation_artifact("e04250")  # eLife-style article id
    # DOI registrant prefix — only reachable once D5 stopped splitting on `.`, which is how it
    # became visible at all (it surfaced as `10.18653` on the real corpus).
    assert is_citation_artifact("10.18653")
    assert is_citation_artifact("10.1038")


def test_is_citation_artifact_spares_real_specialist_vocabulary() -> None:
    """⚠ The regression guard for this whole feature. These have been called junk twice and are
    real terms in a multi-domain corpus — a shape rule must not touch any of them."""
    protected = (
        "cre",
        "dbs",
        "16p11",
        "c57bl",
        "cajal",
        "p53",
        "s100b",
        "e2f1",
        "bm25",
        "gpt-4",
        "16p11.2",
        "c57bl/6",
        "dlight1.1",
        "gpt 3.5",  # designators D5 now keeps whole
    )
    for term in protected:
        assert not is_citation_artifact(term), term


def test_candidate_terms_drops_citation_shapes() -> None:
    terms = set(
        candidate_terms(tokenize("shadmehr 2014a e04250 cerebellum"), ngram_max=1, min_chars=3)
    )
    assert "2014a" not in terms
    assert "e04250" not in terms
    assert "cerebellum" in terms
    assert "shadmehr" in terms  # a surname in prose is NOT filtered — that is the whole D4 rule


def test_strip_reference_section_cuts_the_bibliography() -> None:
    body = "Motor commands drive the reach. " * 40
    text = (
        f"{body}\n\n## References\n\nShadmehr R, Krakauer J. 2008. Exp Brain Res.\nMathis A. 2018."
    )
    out = strip_reference_section(text)
    assert "Shadmehr" not in out
    assert "Motor commands" in out


def test_strip_reference_section_accepts_the_common_heading_spellings() -> None:
    """⚠ The markdown-emphasis forms are the ones that matter. A first cut without them fired on
    only 25 of 97 real documents — `## **References**` is what PyMuPDF4LLM actually emits."""
    body = "content. " * 60
    for heading in (
        "References",
        "REFERENCES",
        "## Bibliography",
        "5. References",
        "Works Cited",
        "## **References**",
        "## **REFERENCES**",
        "_REFERENCES_",
        "## 10. References",
    ):
        text = f"{body}\n{heading}\nSmith J. 2020."
        assert "Smith" not in strip_reference_section(text), heading


def test_strip_reference_section_ignores_a_running_header_with_the_word() -> None:
    """`References  |  109` is a page footer, not the bibliography heading."""
    body = "content. " * 60
    text = f"{body}\nReferences  |  109\nreal body continues here."
    assert "real body continues here." in strip_reference_section(text)


def test_strip_reference_section_ignores_the_word_in_prose() -> None:
    """A whole-line heading only — otherwise a paper *about* citation practice loses its body."""
    text = "We surveyed references in the literature. " * 30
    assert strip_reference_section(text) == text


def test_strip_reference_section_ignores_a_heading_in_the_first_half() -> None:
    """Structural guard: a bibliography lives at the back. A match up front is something else,
    and cutting there would amputate the document."""
    text = "References\n" + ("real body content. " * 100)
    assert strip_reference_section(text) == text


def test_strip_reference_section_is_a_noop_without_a_bibliography() -> None:
    text = "a paper with no reference list at all. " * 20
    assert strip_reference_section(text) == text


def test_strip_reference_section_survives_an_empty_document() -> None:
    assert strip_reference_section("") == ""


# --------------------------------------------------------------------------- #
# Session 5c — the junk filters: each with what it drops and what it keeps
# --------------------------------------------------------------------------- #


def test_candidate_terms_drops_the_br_token_a_table_cell_leaves() -> None:
    tokens = tokenize("seen<br>unseen<br>accuracy")
    terms = set(candidate_terms(tokens, ngram_max=2, min_chars=3))
    assert not any("br" in t.split() for t in terms)  # `seen br`, `br unseen`
    assert {"seen", "unseen", "accuracy"} <= terms  # the cells' own words stay
    # `br` only as a whole token: words that merely start with it are untouched.
    assert "brain" in candidate_terms(["brain"], ngram_max=1, min_chars=3)


def test_candidate_terms_drops_a_token_with_no_letter() -> None:
    tokens = tokenize("Genetics 111, November 7. p 0.01 conv 256")
    terms = set(candidate_terms(tokens, ngram_max=2, min_chars=3))
    for junk in ("genetics 111", "november 7", "conv 256", "0.01"):
        assert junk not in terms
    assert {"genetics", "november", "conv"} <= terms  # the words beside the numbers stay


def test_candidate_terms_keeps_a_name_written_with_its_number_joined() -> None:
    tokens = tokenize("SAM2 and GPT-4 on 16p11.2 in C57BL/6 mice; layer 2/3")
    terms = set(candidate_terms(tokens, ngram_max=1, min_chars=3))
    assert {"sam2", "gpt-4", "16p11.2", "c57bl/6"} <= terms
    # The cost, stated: the same names written with a space lose the number-only token.
    spaced = set(candidate_terms(tokenize("SAM 2 and layer 5"), ngram_max=2, min_chars=3))
    assert "sam 2" not in spaced and "layer 5" not in spaced
    assert "layer" in spaced


def test_candidate_terms_drops_letter_spaced_text_and_initials() -> None:
    tokens = tokenize("N E U R O N S were drawn by T. N. Wiesel")
    terms = set(candidate_terms(tokens, ngram_max=3, min_chars=3))
    assert not any(sum(len(tok) == 1 for tok in t.split()) >= 2 for t in terms)
    assert "wiesel" in terms


def test_candidate_terms_keeps_a_phrase_with_one_single_letter() -> None:
    tokens = tokenize("T cell receptors bind vitamin D")
    terms = set(candidate_terms(tokens, ngram_max=2, min_chars=3))
    assert "t cell" in terms
    assert "vitamin d" in terms


def test_strip_urls_removes_addresses_dois_and_email_by_shape() -> None:
    text = (
        "See https://aclanthology.org/2024.acl-long.12 and www.example.com/a, "
        "aclanthology.org/2024.findings-acl.5 or doi:10.1038/nn.4502 and 10.7554/eLife.84566, "
        "write to barbaraf@icb.usp.br about cortex."
    )
    tokens = tokenize(strip_urls(text))
    for fragment in ("aclanthology", "org/2024", "example", "usp", "barbaraf", "nn.4502"):
        assert fragment not in tokens
    assert "cortex" in tokens and "write" in tokens


def test_strip_urls_leaves_prose_abbreviations_and_designators() -> None:
    text = "e.g. the 16p11.2 deletion, i.e. a CNV; Fig. 2 shows c57bl/6 mice."
    assert strip_urls(text) == text


def test_cited_authors_reads_et_al_on_a_majority_of_uses() -> None:
    tokens = tokenize(
        "Elias et al. (2020) mapped it. Later work (Elias et al., 2021) agrees; "
        "as Elias showed, the target holds. Gouveia et al. replicated it."
    )
    cited = cited_authors(tokens)
    assert "elias" in cited  # 2 of 3 uses are followed by "et al"
    assert "gouveia" in cited  # 1 of 1


def test_cited_authors_spares_a_name_the_document_is_about() -> None:
    # The subject of the text: named often, cited "et al." rarely. Below a majority, not cited.
    tokens = tokenize(
        "Cajal drew neurons. Cajal argued for the neuron doctrine. Cajal et al. is rare. "
        "Cajal stained with silver."
    )
    assert "cajal" not in cited_authors(tokens)
    assert cited_authors([]) == frozenset()


def test_candidate_terms_drops_the_documents_cited_authors() -> None:
    tokens = tokenize("Zouhar et al. measured retrieval. Zouhar et al. again. dense retrieval")
    terms = set(candidate_terms(tokens, ngram_max=2, min_chars=3, cited=cited_authors(tokens)))
    assert not any("zouhar" in t.split() for t in terms)
    assert {"dense retrieval", "retrieval"} <= terms
    # Without the document's evidence the same name stays: there is no list of names.
    assert "zouhar" in candidate_terms(tokens, ngram_max=1, min_chars=3)


def test_majority_is_shared_with_the_term_signals() -> None:
    from doc_assistant.knowledge import term_signals

    assert term_signals.majority is majority
    assert majority(1, 2) and not majority(1, 3) and not majority(0, 0)


# --------------------------------------------------------------------------- #
# Robustness contract: 0 documents, 1 document, a synthetic 10,000-document corpus
# --------------------------------------------------------------------------- #


def _terms_of(text: str) -> list[str]:
    tokens = tokenize(strip_urls(text))
    return candidate_terms(tokens, ngram_max=3, min_chars=3, cited=cited_authors(tokens))


def test_extractor_core_at_zero_documents() -> None:
    assert tf_idf_keywords({}, top_k=15) == {}
    assert _terms_of("") == []


def test_extractor_core_at_one_document() -> None:
    text = "grid cells encode space; grid cells fire"
    ranked = tf_idf_keywords({"d1": _terms_of(text)}, top_k=15)
    terms = [k.term for k in ranked["d1"]]
    assert terms and len(terms) <= 15
    assert all(k.score > 0 for k in ranked["d1"])


def test_extractor_core_at_ten_thousand_documents() -> None:
    # Synthetic vocabulary: each document has its own word, a word shared by a tenth of the
    # corpus, and the junk shapes 5c removes. Every bound below holds at any corpus size.
    n = 10_000
    doc_terms = {
        f"d{i}": _terms_of(
            f"topic{i} topic{i} shared{i % 10} table<br>cell {i} n e u Smith et al. Smith et al."
        )
        for i in range(n)
    }
    ranked = tf_idf_keywords(doc_terms, top_k=15)
    assert len(ranked) == n
    for doc_id, keywords in ranked.items():
        assert 0 < len(keywords) <= 15
        assert keywords[0].term == f"topic{doc_id[1:]}"  # its own word ranks first
        for k in keywords:
            tokens = k.term.split()
            assert "br" not in tokens and "smith" not in tokens
            assert all(any(ch.isalpha() for ch in tok) for tok in tokens)
            assert sum(len(tok) == 1 for tok in tokens) < 2
