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
    signal_payload,
    spelled_out,
    spells_initials,
    term_signals,
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
