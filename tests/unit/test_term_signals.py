"""Guards for term signals — what the prose says about a term (ADR-053 decision 4).

The contract: a long form is found beside its short form by Schwartz & Hearst's rule, copied
verbatim; case is identity; a signal is counted, offered and never applied. The sentences are
written for the tests in the shapes the working library uses (a citation after the short form,
curly quotes round the long one), not copied from it. Curly quotes are built by code point.
"""

from __future__ import annotations

import pytest

from doc_assistant.knowledge.term_signals import (
    abbreviation_pairs,
    is_short_form,
    load_term_signals,
    long_form_before,
    majority,
    offer_example,
    offer_marks,
    signal_payload,
    spelled_out,
    spells_initials,
    term_signals,
    where_it_occurs,
)

LQ, RQ = chr(0x201C), chr(0x201D)

DBS_A = (
    "For these patients, neuromodulation therapies such as deep brain stimulation (DBS) have "
    "been investigated in several centres."
)
DBS_B = (
    "We derived the method from lesion and deep brain stimulation (DBS) network mapping, which "
    "has given insight into circuits."
)
PDDL = (
    f"The agent synthesizes abstractions in the form of {LQ}planning domain definition "
    f"language{RQ} (PDDL; Ghallab et al. (1998)) operators with little guidance."
)


def _chunks(*texts: tuple[str, str]) -> list[tuple[str, str, str]]:
    """``(document, text)`` pairs as the reader's ``(chunk_key, document_id, text)``."""
    seen: dict[str, int] = {}
    out = []
    for document_id, text in texts:
        index = seen.get(document_id, 0)
        seen[document_id] = index + 1
        out.append((f"{document_id}:p{index}", document_id, text))
    return out


# ---------------------------------------------------------------- one sentence


def test_a_long_form_is_found_before_its_short_form_in_brackets() -> None:
    pairs = abbreviation_pairs(DBS_A)
    assert [(p.short, p.long, p.initials) for p in pairs] == [
        ("DBS", "deep brain stimulation", True)
    ]


def test_a_citation_after_the_short_form_and_quotes_round_the_long_one_are_left_out() -> None:
    pairs = abbreviation_pairs(PDDL)
    assert [(p.short, p.long) for p in pairs] == [("PDDL", "planning domain definition language")]


def test_the_long_form_is_copied_verbatim_from_the_sentence() -> None:
    # Odd spacing from the PDF, and a hyphenated long form: neither is tidied.
    odd = "We use the mean  Intersection-over-Union (mIoU) as the main metric in every table."
    (pair,) = abbreviation_pairs(odd)
    assert pair.long == "mean  Intersection-over-Union"
    assert pair.long in odd
    assert pair.initials


def test_the_short_form_may_come_first_with_the_long_form_in_the_brackets() -> None:
    (pair,) = abbreviation_pairs("We compare against DPR (dense passage retriever) on both sets.")
    assert (pair.short, pair.long) == ("DPR", "dense passage retriever")


def test_a_remark_in_brackets_is_not_a_long_form() -> None:
    remark = "We keep BM25 (a much stronger baseline than reported before) in all the tables."
    assert abbreviation_pairs(remark) == []


def test_a_cited_name_in_brackets_is_not_a_short_form() -> None:
    cited = "Large-scale parcellation gives functional connectomes (Sporns, 2011) of the cortex."
    assert not is_short_form("Sporns")
    assert abbreviation_pairs(cited) == []


def test_a_short_form_is_not_its_own_long_form() -> None:
    assert abbreviation_pairs("We report the BM25 score (BM25) for every query in the set.") == []


def test_brackets_that_hold_no_short_form_give_nothing() -> None:
    assert abbreviation_pairs("The effect was small (see below) and it faded within a week.") == []
    assert abbreviation_pairs("An unclosed bracket (DBS never ends in this sentence.") == []
    assert not is_short_form("x")
    assert not is_short_form("a far too long one")


def test_the_first_letter_must_start_a_word() -> None:
    # "d" is inside "radio": the long form has to begin at a word.
    assert long_form_before("DB", "radio band") is None
    assert long_form_before("DB", "the deep brain") == "deep brain"


def test_initials_are_evidence_and_the_published_rule_still_accepts_the_pair() -> None:
    loose = "Most of the recent deep learning approaches (DNNs) need far more data than this."
    tight = "Most of the recent deep neural networks (DNNs) need far more data than this one."
    (loose_pair,) = abbreviation_pairs(loose)
    (tight_pair,) = abbreviation_pairs(tight)
    assert (loose_pair.long, loose_pair.initials) == ("deep learning approaches", False)
    assert (tight_pair.long, tight_pair.initials) == ("deep neural networks", True)


def test_a_capital_inside_the_long_form_counts_as_an_initial() -> None:
    assert spells_initials("SBERT", "Sentence-BERT")
    assert spells_initials("NeuN", "anti-neuron-specific nuclear protein")
    assert not spells_initials("OBF", "orthogonal backfill")


# ---------------------------------------------------------------- the library


def test_pairs_are_counted_in_documents_not_in_sentences() -> None:
    from doc_assistant.knowledge.definitions import document_sentences

    chunks = _chunks(("a", f"{DBS_A} {DBS_A}"), ("b", DBS_B), ("c", PDDL))
    pairs = {p.short: p for p in spelled_out(document_sentences(chunks))}
    assert pairs["DBS"].documents == 2
    assert pairs["PDDL"].documents == 1
    assert pairs["DBS"].document_id == "a"
    assert pairs["DBS"].long in pairs["DBS"].sentence


def test_dbs_and_pddl_are_spelled_out_from_the_prose_and_offered_not_added() -> None:
    concepts = [("c-dbs", "dbs"), ("c-pddl", "pddl")]
    aliases: dict[str, list[str]] = {}
    signals = term_signals(concepts, aliases, _chunks(("a", DBS_A), ("b", DBS_B), ("c", PDDL)))

    (dbs,) = signals["c-dbs"].offers
    assert (dbs.form, dbs.kind, dbs.beside) == ("deep brain stimulation", "long", "dbs")
    assert (dbs.documents, dbs.initials, dbs.known) == (2, True, False)
    assert dbs.form in dbs.sentence
    assert dbs.beside_written == "DBS"  # the label is stored lower-case; the sentence is not

    (pddl,) = signals["c-pddl"].offers
    assert (pddl.form, pddl.known) == ("planning domain definition language", False)

    # Offered only: the inputs are what they were.
    assert concepts == [("c-dbs", "dbs"), ("c-pddl", "pddl")]
    assert aliases == {}


def test_a_form_the_concept_already_has_is_found_and_marked_known() -> None:
    signals = term_signals(
        [("c-dbs", "dbs")],
        {"c-dbs": ["Deep Brain Stimulation"]},
        _chunks(("a", DBS_A)),
    )
    (offer,) = signals["c-dbs"].offers
    assert offer.known


def test_a_label_that_is_the_long_form_is_offered_its_short_form() -> None:
    rag = (
        "Our system builds on retrieval-augmented generation (RAG) and adds a reranking step "
        "before the reader."
    )
    signals = term_signals([("c-rag", "retrieval-augmented generation")], {}, _chunks(("a", rag)))
    (offer,) = signals["c-rag"].offers
    assert (offer.form, offer.kind) == ("RAG", "short")
    assert offer.beside == "retrieval-augmented generation"
    assert offer.beside_written == "retrieval-augmented generation"


def test_a_short_form_written_in_another_case_is_another_word() -> None:
    cre = (
        "Binding at the cAMP response element (CRE) was reduced in the treated animals after "
        "one week."
    )
    chunks = _chunks(("a", cre))
    concepts = [("c-cre", "cre")]
    # The library writes the concept Cre: CRE is not it (ADR-053 decision 3).
    cased = term_signals(concepts, {}, chunks, written={("c-cre", "cre"): "Cre"})
    assert cased["c-cre"].offers == ()
    # With no written form the label matches in any case, as everywhere else.
    folded = term_signals(concepts, {}, chunks)
    assert [o.form for o in folded["c-cre"].offers] == ["cAMP response element"]


def test_the_word_beside_a_label_is_counted_and_function_words_are_not() -> None:
    text = (
        "We train a pose estimation model on mouse videos. The pose estimation step runs first. "
        "Animal pose estimation is hard in the dark. We also correct the pose, and it helps here."
    )
    signals = term_signals([("c-pose", "pose")], {}, _chunks(("a", text)))["c-pose"]
    assert signals.one_word
    assert (signals.mentions, signals.documents) == (4, 1)
    assert signals.after is not None
    assert (signals.after.word, signals.after.count) == ("estimation", 3)
    # "a pose", "the pose": articles are not neighbours; "animal pose" is.
    assert signals.before is not None
    assert (signals.before.word, signals.before.count) == ("animal", 1)


def test_a_label_followed_by_et_al_is_counted_as_cited() -> None:
    text = (
        "The benchmark of Cohan et al. (2020) covers seven tasks. Cohan et al. also release the "
        "training data. We follow Cohan and colleagues for the splits used in every run."
    )
    signals = term_signals([("c-cohan", "cohan")], {}, _chunks(("a", text)))["c-cohan"]
    assert (signals.mentions, signals.et_al) == (3, 2)
    assert signals.after is None or signals.after.word != "et"


def test_a_term_the_prose_never_uses_has_an_entry_with_zeros() -> None:
    signals = term_signals([("c-x", "param1")], {}, _chunks(("a", DBS_A)))["c-x"]
    assert (signals.mentions, signals.documents, signals.offers) == (0, 0, ())
    assert signals.after is None and signals.before is None


def test_an_empty_library_gives_every_term_its_zeros() -> None:
    signals = term_signals([("c-dbs", "dbs"), ("c-empty", "  ")], {}, [])
    assert signals["c-dbs"].mentions == 0
    assert signals["c-empty"].offers == ()


def test_a_single_letter_and_a_possessive_are_not_split_into_neighbours() -> None:
    apostrophe = chr(0x2019)
    text = (
        "The VSV-G envelope is used for the first injection. We stain VSV-G positive cells too. "
        f"Cajal{apostrophe}s accessory body lies near the nucleolus in every one of these neurons."
    )
    signals = term_signals([("c-vsv", "vsv"), ("c-acc", "accessory")], {}, _chunks(("a", text)))
    assert signals["c-vsv"].mentions == 2
    assert signals["c-vsv"].after is None  # "g" is a letter, not a word
    before = signals["c-acc"].before
    assert before is not None and before.word == f"cajal{apostrophe}s"


def test_the_payload_is_plain_data() -> None:
    signals = term_signals([("c-dbs", "dbs")], {}, _chunks(("a", DBS_A)))["c-dbs"]
    payload = signal_payload(signals)
    assert payload["label"] == "dbs" and payload["one_word"] is True
    assert payload["offers"][0]["form"] == "deep brain stimulation"
    assert payload["offers"][0]["known"] is False


def test_the_loader_reads_the_vocabulary_and_the_prose_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patched on the modules that own the names (a re-export would be a second binding).
    from doc_assistant.knowledge import concept_skeleton, written_forms

    monkeypatch.setattr(
        written_forms, "load_vocabulary", lambda: ([("c-cre", "cre")], {"c-cre": []})
    )
    monkeypatch.setattr(written_forms, "load_written_forms", lambda: {("c-cre", "cre"): "Cre"})
    monkeypatch.setattr(
        concept_skeleton,
        "load_presence_inputs",
        lambda: _chunks(("a", "Binding at the cAMP response element (CRE) fell after a week.")),
    )
    signals = load_term_signals()
    assert list(signals) == ["c-cre"]
    assert signals["c-cre"].offers == ()  # the stored written form reached the matcher


def test_a_label_and_a_long_form_that_join_their_words_otherwise_are_one_form() -> None:
    # KI-63: `ssp bfd` is used in prose written "SSp-bfd" (it was "never in the text"), and a
    # label with a hyphen is offered the short form of the long form written with a space.
    text = "Six Cre lines in SSp-bfd were imaged, and state space (SS) models fit the traces."
    concepts = [("c-ssp", "ssp bfd"), ("c-ss", "state-space")]
    signals = term_signals(concepts, {}, _chunks(("a", text)))
    assert signals["c-ssp"].mentions == 1
    (offer,) = signals["c-ss"].offers
    assert (offer.form, offer.kind) == ("SS", "short")


# ---------------------------------------------------------------- what the panel shows


def _marked(text: str, marks: tuple[tuple[int, int], ...]) -> list[str]:
    return [text[a:b] for a, b in marks]


def test_an_offer_marks_its_two_forms_on_word_edges_never_inside_a_word() -> None:
    # The review page lit the "co" of "communication" (baseline section 1); the marks come from
    # the rule that found the pair, so they land on the short form as written.
    text = (
        "Communicability (CO) measures communication between two nodes of a network in every "
        "one of the graphs we study here."
    )
    (offer,) = term_signals([("c-co", "co")], {}, _chunks(("a", text)))["c-co"].offers
    assert _marked(offer.sentence, offer_marks(offer)) == ["Communicability", "CO"]


def test_an_offer_marks_both_orders_and_leaves_quotes_and_citations_out() -> None:
    dpr = (
        "DPR (dense passage retriever) encodes the questions and the passages separately, in two "
        "towers of one model."
    )
    found = term_signals(
        [("c-dpr", "dpr"), ("c-pddl", "pddl")], {}, _chunks(("a", dpr), ("b", PDDL))
    )
    (first,) = found["c-dpr"].offers
    assert _marked(first.sentence, offer_marks(first)) == ["DPR", "dense passage retriever"]
    (second,) = found["c-pddl"].offers
    example = offer_example(second)
    assert (example.document_id, example.chunk_key, example.text) == ("b", "b:p0", PDDL)
    assert _marked(example.text, example.marks) == ["planning domain definition language", "PDDL"]


def test_a_label_that_is_the_long_form_marks_the_short_form_it_is_offered() -> None:
    signals = term_signals([("c-dbs", "deep brain stimulation")], {}, _chunks(("a", DBS_A)))
    (offer,) = signals["c-dbs"].offers
    assert offer.kind == "short"
    assert _marked(offer.sentence, offer_marks(offer)) == ["deep brain stimulation", "DBS"]


def test_a_cited_author_keeps_its_first_et_al_sentence_with_the_name_marked() -> None:
    text = (
        "The benchmark of Cohan et al. (2020) covers seven tasks. Cohan et al. also release the "
        "training data. We follow Cohan and colleagues for the splits used in every run."
    )
    signals = term_signals([("c-cohan", "cohan")], {}, _chunks(("b", DBS_A), ("a", text)))
    cohan = signals["c-cohan"]
    assert cohan.cited  # 2 of 3 mentions
    example = cohan.et_al_example
    assert example is not None and example.chunk_key == "a:p0"
    assert example.text.startswith("The benchmark of Cohan et al.")
    assert _marked(example.text, example.marks) == ["Cohan"]


def test_a_name_followed_by_et_al_less_than_half_the_time_is_not_shown_as_cited() -> None:
    # `sporns`: 8 of 73 mentions. The rule is a majority, and below it the name is not "cited".
    text = (
        "Following Sporns et al. we built the network from tract tracing in every animal. "
        "Sporns argues the hubs matter. The hubs of Sporns form a rich club in all of them."
    )
    signals = term_signals([("c-s", "sporns")], {}, _chunks(("a", text)))["c-s"]
    assert (signals.et_al, signals.mentions, signals.cited) == (1, 3, False)


def test_majority_is_half_or_more_and_never_of_nothing() -> None:
    assert majority(1, 2) and majority(2, 2)
    assert not majority(1, 3)
    assert not majority(0, 0)


# A document that holds one label in every place but a body sentence, and once in a sentence
# written with a comma between its words. The reference list is the shorter part, so its
# heading lies past the halfway mark the bibliography cut wants.
KOONCE_BODY = (
    "Copyright Koonce, Emerson and colleagues hold the rights to every figure in this article "
    "today. The table lists them.\n\n| Author | Year |\n|---|---|\n| Koonce, Emerson | 2024 |\n\n"
    "## Koonce Emerson notes\n\n- Koonce, Emerson et al. list entry\n\n"
    "The rest of this document describes the methods we used, and why each one was chosen."
)
KOONCE_REFERENCES = "References\n\nKoonce, Emerson et al. 2024. Cells. eLife 13.\n"
KOONCE_FIGURE = (
    "Figure 3. The Koonce Emerson atlas of cells.\n\n---\n\n"
    "Copyright Koonce, Emerson and colleagues hold the rights to every figure in this article."
)


def test_a_label_no_sentence_uses_is_placed_where_the_keyword_extractor_found_it() -> None:
    prose = [("d:p0", "d", KOONCE_BODY), ("d:p1", "d", KOONCE_REFERENCES)]
    places = where_it_occurs("koonce emerson", prose, [("d:p2", "d", KOONCE_FIGURE)])
    assert [(p.place, p.mentions, p.documents) for p in places] == [
        ("sentence", 1, 1),
        ("table", 1, 1),
        ("heading", 1, 1),
        ("figure", 1, 1),  # the figure's own text; the copy of its citing passage is not read
        ("references", 1, 1),
        ("elsewhere", 1, 1),
    ]
    by_place = {p.place: p.example for p in places}
    assert _marked(by_place["sentence"].text, by_place["sentence"].marks) == ["Koonce, Emerson"]
    assert _marked(by_place["heading"].text, by_place["heading"].marks) == ["Koonce Emerson"]
    assert by_place["references"].chunk_key == "d:p1"
    assert by_place["figure"].chunk_key == "d:p2"


def test_the_mention_matcher_still_decides_that_no_sentence_uses_the_label() -> None:
    # The places are looked up the extractor's way, the use is counted the product's way: a
    # comma between the words is not the form, so the label stays one no sentence uses.
    signals = term_signals([("c-k", "koonce emerson")], {}, _chunks(("d", KOONCE_BODY)))
    assert signals["c-k"].mentions == 0


def test_places_keep_word_edges_and_an_empty_label_or_library_has_none() -> None:
    text = "Computer vision models were compared on every one of the benchmark sets we hold."
    assert where_it_occurs("comput vis", [("d:p0", "d", text)]) == ()
    assert where_it_occurs("  ", [("d:p0", "d", KOONCE_BODY)]) == ()
    assert where_it_occurs("koonce emerson", []) == ()


def test_the_figure_join_is_the_one_ingest_writes() -> None:
    from doc_assistant.ingest.figures import figure_parent_text
    from doc_assistant.knowledge.term_signals import _FIGURE_CONTEXT_JOIN

    assert figure_parent_text("Own.", "Cited.") == f"Own.{_FIGURE_CONTEXT_JOIN}Cited."
