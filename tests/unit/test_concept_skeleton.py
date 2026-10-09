"""Pure-core guard tests for the deterministic concept skeleton (Node A).

Fixed toy inputs, no DB / LLM / network. The no-edge-creation guard
(citation/similarity annotate-never-create) is the density-control invariant.
"""

from __future__ import annotations

import pytest

from doc_assistant.knowledge.concept_skeleton import (
    PRESENCE_BOUNDARY,
    PRESENCE_SUBSTRING,
    ConceptNode,
    ConceptPresence,
    SkeletonEdge,
    add_citation_provenance,
    add_similarity_provenance,
    analyze_skeleton,
    compile_boundary_pattern,
    cooccurrence_edges,
    edge_weight,
    form_matcher,
    forms_fingerprints,
    is_cased,
    match_broad_presence,
    match_presence,
    prose_parents,
    skeleton_from_dict,
    skeleton_to_dict,
    word_case_key,
)

# ---- written forms: case-aware matching (ADR-053 decision 3) ---------------


def test_a_capital_that_starts_a_word_is_not_part_of_the_word() -> None:
    # A capital that starts a word is systematic (titles, sentences); inside a word it is identity.
    assert word_case_key("Cre") == word_case_key("cre") != word_case_key("CRE")
    assert word_case_key("Natural Questions") == word_case_key("natural questions")
    assert word_case_key("dIN") != word_case_key("Din")  # "vitamin Din" is not dIN
    assert word_case_key("ColBERT") != word_case_key("Colbert")  # nor is the surname ColBERT
    assert word_case_key("SPECTER") != word_case_key("Specter")  # nor a book title SPECTER
    assert word_case_key("3D medical") != word_case_key("3d medical")  # 3D: "D" is inside a word


def test_only_a_written_form_with_a_capital_makes_matching_case_aware() -> None:
    assert not is_cased(None)
    assert not is_cased("beta")  # lower case: any case, exactly as before written forms
    assert is_cased("Cre") and is_cased("dIN") and is_cased("BM25")


def test_form_matcher_is_case_folded_without_a_written_form_or_with_a_stale_one() -> None:
    for written in (None, "beta", "Cre"):  # "Cre" is not a spelling of "din": ignored
        matcher = form_matcher("din", written)
        assert matcher.search("the Din join", "the din join")
        assert matcher.count("dIN, Din and DIN", "din, din and din") == 3


def test_presence_is_case_aware_where_the_library_writes_a_case() -> None:
    concepts = [("d", "din"), ("c", "cre"), ("b", "beta")]
    chunks = [
        ("n1:p0", "n1", "Recordings from the dIN population."),
        ("v1:p0", "v1", "A diet with vitamin Din adults."),
        ("m1:p0", "m1", "Expression of Cre recombinase in mice."),
        ("m2:p0", "m2", "A cre-dependent virus was injected."),
        ("x1:p0", "x1", "The CRE element binds the complex."),
        ("o1:p0", "o1", "Beta oscillations and beta power."),
    ]
    written = {("d", "din"): "dIN", ("c", "cre"): "Cre"}
    presences = match_presence(concepts, {}, chunks, written=written)
    docs: dict[str, set[str]] = {}
    for p in presences:
        docs.setdefault(p.concept_id, set()).add(p.document_id)
    assert docs["d"] == {"n1"}  # not the "vitamin Din" join
    assert docs["c"] == {"m1", "m2"}  # Cre and cre-dependent, not CRE
    assert docs["b"] == {"o1"}  # lower-case written form: any case
    assert {p.document_id: p.n_mentions for p in presences if p.concept_id == "b"} == {"o1": 2}

    # Without written forms, matching is case-folded exactly as before.
    folded = {p.document_id for p in match_presence(concepts, {}, chunks) if p.concept_id == "d"}
    assert folded == {"n1", "v1"}


def test_a_title_cased_written_form_still_matches_the_phrase_in_prose() -> None:
    # "Natural Questions" wins the vote from titles; prose that says "natural questions" is still
    # a mention. An all-caps rendering is another casing: the price of telling CRE from Cre.
    chunks = [
        ("a:p0", "a", "We train on Natural Questions and TriviaQA."),
        ("b:p0", "b", "Users ask natural questions in plain language."),
        ("c:p0", "c", "NATURAL QUESTIONS heads the table."),
    ]
    written = {("q", "natural questions"): "Natural Questions"}
    presences = match_presence([("q", "natural questions")], {}, chunks, written=written)
    assert {p.document_id for p in presences} == {"a", "b"}


def test_substring_mode_ignores_written_forms() -> None:
    chunks = [("v1:p0", "v1", "A diet with vitamin Din adults.")]
    written = {("d", "din"): "dIN"}
    presences = match_presence(
        [("d", "din")], {}, chunks, mode=PRESENCE_SUBSTRING, written=written
    )
    assert [p.document_id for p in presences] == ["v1"]


def test_a_node_written_form_survives_the_skeleton_round_trip() -> None:
    nodes = [
        ConceptNode(id="a", label="din", doc_ids=("d1",), degree=0, community=-1, written="dIN"),
        ConceptNode(id="b", label="beta", doc_ids=("d1",), degree=0, community=-1),
    ]
    skeleton = analyze_skeleton(nodes, [], seed=1)
    data = skeleton_to_dict(skeleton)
    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id["a"]["written"] == "dIN"
    assert "written" not in by_id["b"]  # unchanged shape for a node without one
    back = {n.id: n.written for n in skeleton_from_dict(data).nodes}
    assert back == {"a": "dIN", "b": None}


# ---- exact and broad forms (ADR-054) ----------------------------------------


def test_broad_forms_are_matched_on_their_own_form_by_form() -> None:
    chunks = [
        ("d1:p0", "d1", "Knowledge distillation is combined with pruning."),
        ("d2:p0", "d2", "We apply distillation to the ranker."),
        ("d3:p0", "d3", "A contrastive encoder, and distillation of its scores."),
        ("d4:p0", "d4", "Nothing relevant is said here."),
    ]
    found = match_broad_presence({"kd": ["distillation"], "cl": ["Contrastive", "nce"]}, chunks)
    # Keyed by the casefolded form; a form that occurs nowhere (`nce`) is absent.
    assert found == {
        "kd": {"distillation": {"d1", "d2", "d3"}},
        "cl": {"contrastive": {"d3"}},
    }


def test_broad_forms_match_whole_words_and_respect_the_written_case() -> None:
    chunks = [
        ("a:p0", "a", "Expression of Cre in layer 6."),
        ("b:p0", "b", "The CRE element binds the complex."),
        ("c:p0", "c", "A concrete example."),  # "cre" inside a word is not a mention
    ]
    folded = match_broad_presence({"x": ["cre"]}, chunks)
    assert folded == {"x": {"cre": {"a", "b"}}}
    cased = match_broad_presence({"x": ["cre"]}, chunks, written={("x", "cre"): "Cre"})
    assert cased == {"x": {"cre": {"a"}}}


def test_no_broad_forms_means_nothing_beside_presence() -> None:
    chunks = [("d1:p0", "d1", "Any text at all.")]
    assert match_broad_presence({}, chunks) == {}
    assert match_broad_presence({"kd": []}, chunks) == {}
    assert match_broad_presence({"kd": ["distillation"]}, []) == {}


def test_a_nodes_broad_documents_survive_the_round_trip_and_stay_out_when_empty() -> None:
    nodes = [
        ConceptNode(
            id="a",
            label="knowledge distillation",
            doc_ids=("d1",),
            degree=0,
            community=-1,
            broad_doc_ids=("d2", "d3"),
            broad_forms=("distillation",),
        ),
        ConceptNode(id="b", label="pruning", doc_ids=("d1",), degree=0, community=-1),
    ]
    skeleton = analyze_skeleton(nodes, [], seed=1)
    data = skeleton_to_dict(skeleton)
    by_id = {n["id"]: n for n in data["nodes"]}
    assert by_id["a"]["broad_doc_ids"] == ["d2", "d3"]
    assert by_id["a"]["broad_forms"] == ["distillation"]
    # A node with nothing beside its presence serialises exactly as it did before ADR-054.
    assert "broad_doc_ids" not in by_id["b"] and "broad_forms" not in by_id["b"]
    back = {n.id: (n.broad_doc_ids, n.broad_forms) for n in skeleton_from_dict(data).nodes}
    assert back == {"a": (("d2", "d3"), ("distillation",)), "b": ((), ())}


def test_broad_documents_do_not_change_the_graph_version() -> None:
    """The version fingerprints structure: nodes and edges. What sits beside presence is neither,
    so a library where nothing is marked broad keeps the version it had."""
    plain = ConceptNode(id="a", label="x", doc_ids=("d1",), degree=0, community=-1)
    beside = ConceptNode(
        id="a",
        label="x",
        doc_ids=("d1",),
        degree=0,
        community=-1,
        broad_doc_ids=("d2",),
        broad_forms=("y",),
    )
    assert (
        analyze_skeleton([plain], [], seed=1).meta["graph_version"]
        == analyze_skeleton([beside], [], seed=1).meta["graph_version"]
    )


def test_forms_fingerprint_moves_with_what_matching_reads_and_nothing_else() -> None:
    """One fingerprint per concept over its name, exact forms and broad forms (ADR-054). Case and
    order are not part of it, because matching ignores both."""
    concepts = [("kd", "knowledge distillation"), ("cre", "cre")]
    base = forms_fingerprints(concepts, {"kd": ["distillation", "KD"]}, {})
    assert set(base) == {"kd", "cre"}
    assert base == forms_fingerprints(concepts, {"kd": ["kd", " Distillation "]}, {})

    # Marked broad: the same strings, counted differently.
    broad = forms_fingerprints(concepts, {"kd": ["KD"]}, {"kd": ["distillation"]})
    assert broad["kd"] != base["kd"]
    assert broad["cre"] == base["cre"]  # one concept's edit is that concept's change

    assert forms_fingerprints(concepts, {"kd": ["KD"]}, {})["kd"] != base["kd"]  # a form removed
    added = forms_fingerprints(concepts, {"kd": ["distillation", "KD"], "cre": ["cre line"]}, {})
    assert added["cre"] != base["cre"]


def test_forms_fingerprint_sees_a_rename_that_keeps_the_same_forms() -> None:
    """Swapping the name with one of its forms matches the same text, but the graph still shows
    the old name — so the name is fingerprinted on its own."""
    before = forms_fingerprints([("c", "cre")], {"c": ["cre recombinase"]}, {})
    after = forms_fingerprints([("c", "cre recombinase")], {"c": ["cre"]}, {})
    assert before["c"] != after["c"]


def test_forms_fingerprint_of_no_concepts_is_empty() -> None:
    assert forms_fingerprints([], {}, {}) == {}


# ---- presence (Decision 2) -------------------------------------------------


def test_presence_matches_label_and_alias_and_skips_absent() -> None:
    concepts = [
        ("c_rag", "RAG"),
        ("c_dpr", "Dense Passage Retrieval"),
        ("c_bm25", "BM25"),  # never appears → no presence
    ]
    aliases = {"c_dpr": ["DPR"]}
    chunks = [
        ("d1:p0", "d1", "We use RAG for grounding."),
        ("d1:p1", "d1", "DPR is a dense retriever."),  # matched via the alias
        ("d1:p2", "d1", "Nothing relevant here."),
    ]
    presences = match_presence(concepts, aliases, chunks)
    by_concept = {p.concept_id: p for p in presences}

    assert set(by_concept) == {"c_rag", "c_dpr"}  # c_bm25 absent → no row
    assert by_concept["c_rag"].chunk_keys == ("d1:p0",)
    assert by_concept["c_dpr"].chunk_keys == ("d1:p1",)
    assert by_concept["c_rag"].document_id == "d1"


def test_presence_counts_mentions_across_chunks() -> None:
    concepts = [("c", "rerank")]
    chunks = [
        ("d1:p0", "d1", "rerank rerank twice"),
        ("d1:p1", "d1", "rerank once"),
        ("d2:p0", "d2", "rerank in another doc"),
    ]
    presences = match_presence(concepts, {}, chunks)
    by_doc = {p.document_id: p for p in presences}
    assert by_doc["d1"].chunk_keys == ("d1:p0", "d1:p1")
    assert by_doc["d1"].n_mentions == 3  # two + one occurrences
    assert by_doc["d2"].chunk_keys == ("d2:p0",)


# ---- presence match mode (R2 / RG-009 — word-boundary vs substring) --------


def test_presence_boundary_rejects_substring_hits() -> None:
    # The confound R2 fixes: "bert" must NOT fire inside sbert / colbert / roberta.
    concepts = [("c_bert", "BERT")]
    chunks = [
        ("d1:p0", "d1", "SBERT and ColBERT build on RoBERTa."),  # no standalone token
        ("d2:p0", "d2", "BERT is the base model."),  # standalone → the only real hit
    ]
    presences = match_presence(concepts, {}, chunks, mode=PRESENCE_BOUNDARY)
    by_doc = {p.document_id: p for p in presences}
    assert set(by_doc) == {"d2"}  # d1 contributes nothing under boundary
    assert by_doc["d2"].chunk_keys == ("d2:p0",)
    assert by_doc["d2"].n_mentions == 1


def test_presence_boundary_matches_at_punctuation_and_string_edges() -> None:
    concepts = [("c_bert", "BERT")]
    chunks = [
        ("d1:p0", "d1", "We use BERT."),  # trailing period
        ("d2:p0", "d2", "(BERT) is strong"),  # wrapped in parens
        ("d3:p0", "d3", "bert"),  # whole string, no surrounding chars
    ]
    presences = match_presence(concepts, {}, chunks, mode=PRESENCE_BOUNDARY)
    assert {p.document_id for p in presences} == {"d1", "d2", "d3"}
    assert all(p.n_mentions == 1 for p in presences)


def test_compile_boundary_pattern_matches_presence_matchers_output() -> None:
    # KI-15: epistemics.concepts_in_text shares this exact pattern builder so chunk-level
    # attribution and Node-A presence matching can never diverge on boundary semantics.
    pattern = compile_boundary_pattern("gpt-4")
    assert pattern.search("we benchmarked gpt-4 on the task.")
    assert not pattern.search("gpt-4o is a different model.")  # non-word edge char, not \b
    # Same behavior as match_presence's own boundary mode on the identical form.
    presences = match_presence(
        [("c", "gpt-4")], {}, [("d1:p0", "d1", "gpt-4 was released."), ("d1:p1", "d1", "gpt-4o.")]
    )
    assert {p.document_id: p.chunk_keys for p in presences} == {"d1": ("d1:p0",)}


def test_presence_boundary_handles_hyphen_and_plus_forms() -> None:
    # \b would mishandle these edge chars; alnum lookarounds get them right.
    concepts = [("c_gpt4", "GPT-4"), ("c_cpp", "C++")]
    chunks = [
        ("d1:p0", "d1", "GPT-4 is used, not GPT-4o."),  # matches gpt-4, not inside gpt-4o
        ("d2:p0", "d2", "Written in C++ here."),
    ]
    presences = match_presence(concepts, {}, chunks, mode=PRESENCE_BOUNDARY)
    by_concept = {p.concept_id: p for p in presences}
    assert by_concept["c_gpt4"].n_mentions == 1  # the gpt-4o occurrence is excluded
    assert by_concept["c_gpt4"].document_id == "d1"
    assert by_concept["c_cpp"].document_id == "d2"  # c++ matched despite '+' edges
    # Deterministic, sorted by (concept_id, document_id).
    assert [(p.concept_id, p.document_id) for p in presences] == sorted(
        (p.concept_id, p.document_id) for p in presences
    )


def test_presence_substring_mode_reproduces_raw_count() -> None:
    concepts = [("c_bert", "BERT")]
    chunks = [("d1:p0", "d1", "SBERT and ColBERT and BERT.")]
    substring = match_presence(concepts, {}, chunks, mode=PRESENCE_SUBSTRING)
    assert substring[0].n_mentions == 3  # sbert + colbert + bert (old behaviour)
    boundary = match_presence(concepts, {}, chunks, mode=PRESENCE_BOUNDARY)
    assert boundary[0].n_mentions == 1  # only the standalone token


def test_presence_default_mode_is_boundary() -> None:
    concepts = [("c_bert", "BERT")]
    chunks = [("d1:p0", "d1", "SBERT only, no standalone.")]
    # No mode passed → boundary → sbert does not count → no presence row at all.
    assert match_presence(concepts, {}, chunks) == []


def test_presence_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="presence mode"):
        match_presence([("c", "bert")], {}, [("d:p0", "d", "bert")], mode="bogus")


# ---- co-occurrence (Decision 4) --------------------------------------------


def test_cooccurrence_threshold() -> None:
    presences = [
        ConceptPresence("a", "d1", ("d1:p0", "d1:p1"), 2),
        ConceptPresence("b", "d1", ("d1:p0", "d1:p1"), 2),
        ConceptPresence("c", "d1", ("d1:p0",), 1),
    ]
    edges2 = cooccurrence_edges(presences, min_cooccurrence=2)
    assert {(e.source_concept_id, e.target_concept_id) for e in edges2} == {("a", "b")}
    edge = edges2[0]
    assert edge.provenance == frozenset({"cooccurrence"})
    assert edge.n_cooccurrence_chunks == 2

    edges1 = cooccurrence_edges(presences, min_cooccurrence=1)
    assert {(e.source_concept_id, e.target_concept_id) for e in edges1} == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }


# ---- what the layer reads: prose parents, not figure chunks (ROADMAP 97) ---

_CITING_PASSAGE = "As Fig. 1 shows, BM25 and dense retrieval disagree on long queries."


def _child(index: int, text: str, **extra: object) -> dict[str, object]:
    return {"document_id": "d1", "parent_index": index, "parent_text": text, **extra}


def _figure_parent(index: int, description: str, cited: str) -> dict[str, object]:
    """A described figure as ingest stores it: own text, a rule, a copy of the citing parent."""
    return _child(
        index,
        f"Figure 1: Recall by query length.\n\n{description}\n\n---\n\n{cited}",
        chunk_type="figure",
        figure_id="f1",
        figure_context="cited",
    )


def test_prose_parents_keeps_one_entry_per_parent_in_key_order() -> None:
    rows = [
        _child(1, "Second parent."),
        _child(0, "First parent."),
        _child(0, "First parent."),  # the parent text is on every child row
        _child(2, ""),  # nothing to read
        {"document_id": "d1", "parent_text": "no index"},
        "not a row",
    ]
    assert prose_parents(rows) == [
        ("d1:p0", "d1", "First parent."),
        ("d1:p1", "d1", "Second parent."),
    ]
    assert prose_parents([]) == []  # an empty library reads as nothing, not as an error


def test_a_figure_parent_is_not_the_documents_text() -> None:
    rows = [
        _child(0, _CITING_PASSAGE),
        _child(1, "An unrelated closing paragraph."),
        _figure_parent(2, "The curve reaches a plateau after ten queries.", _CITING_PASSAGE),
    ]
    assert [key for key, _, _ in prose_parents(rows)] == ["d1:p0", "d1:p1"]


def test_a_figure_adds_neither_a_presence_nor_a_shared_chunk() -> None:
    """The two things a figure parent did when it was read as prose (measured 2026-10-09).

    Its description is a model's wording: ``plateau`` is in no sentence the document wrote. Its
    copy of the citing passage is that passage again: one co-occurrence counted twice, which is
    all an edge needs.
    """
    rows = [
        _child(0, _CITING_PASSAGE),
        _figure_parent(1, "The curve reaches a plateau after ten queries.", _CITING_PASSAGE),
    ]
    concepts = [("bm25", "BM25"), ("dr", "dense retrieval"), ("plateau", "plateau")]

    def read(chunks: list[tuple[str, str, str]]) -> tuple[set[str], list[SkeletonEdge]]:
        presences = match_presence(concepts, {}, chunks)
        return (
            {p.concept_id for p in presences},
            cooccurrence_edges(presences, min_cooccurrence=2),
        )

    present, edges = read(prose_parents(rows))
    assert present == {"bm25", "dr"}
    assert edges == []  # one passage is one shared chunk, below the two an edge needs

    # What every parent gave, so the test fails if the fixture stops showing the fault.
    every_parent = [(f"d1:p{r['parent_index']}", "d1", str(r["parent_text"])) for r in rows]
    present_before, edges_before = read(every_parent)
    assert "plateau" in present_before
    assert [(e.source_concept_id, e.target_concept_id) for e in edges_before] == [("bm25", "dr")]


# ---- the no-edge-creation invariant (Decision 5) ---------------------------


def test_citation_similarity_never_create_edges() -> None:
    # a,b co-occur in d1; x only in d2 and never co-occurs with a or b.
    presences = [
        ConceptPresence("a", "d1", ("d1:p0",), 1),
        ConceptPresence("b", "d1", ("d1:p0",), 1),
        ConceptPresence("x", "d2", ("d2:p0",), 1),
    ]
    edges = cooccurrence_edges(presences, min_cooccurrence=1)
    assert {(e.source_concept_id, e.target_concept_id) for e in edges} == {("a", "b")}
    doc_index = {"a": {"d1"}, "b": {"d1"}, "x": {"d2"}}

    # A d1->d2 citation could "link" a/b to x — but (a,x)/(b,x) are not co-occurrence
    # edges, so NOTHING is created (the density-control invariant).
    cited = add_citation_provenance(edges, [("d1", "d2")], doc_index)
    simd = add_similarity_provenance(cited, [("d1", "d2")], doc_index)
    assert len(cited) == 1 and len(simd) == 1
    assert {(e.source_concept_id, e.target_concept_id) for e in simd} == {("a", "b")}


def test_provenance_added_to_an_existing_edge_only() -> None:
    # a,b co-occur in d1 (the edge); a also in d2, b also in d3.
    presences = [
        ConceptPresence("a", "d1", ("d1:p0",), 1),
        ConceptPresence("b", "d1", ("d1:p0",), 1),
        ConceptPresence("a", "d2", ("d2:p0",), 1),
        ConceptPresence("b", "d3", ("d3:p0",), 1),
    ]
    edges = cooccurrence_edges(presences, min_cooccurrence=1)
    assert {(e.source_concept_id, e.target_concept_id) for e in edges} == {("a", "b")}
    doc_index = {"a": {"d1", "d2"}, "b": {"d1", "d3"}}

    cited = add_citation_provenance(edges, [("d2", "d3")], doc_index)
    assert cited[0].provenance == frozenset({"cooccurrence", "citation"})
    simd = add_similarity_provenance(cited, [("d2", "d3")], doc_index)
    assert simd[0].provenance == frozenset({"cooccurrence", "citation", "similarity"})
    # weight grew with provenance.
    assert simd[0].weight > edges[0].weight


# ---- graded provenance strength (R4 — ratio, not boolean) ------------------


def test_provenance_strength_is_ratio_on_a_partial_graph() -> None:
    # a,b co-occur; both present in d1,d2,d3. Only d1<->d2 are similar → a partial graph.
    edge = SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.0, 4)
    doc_index = {"a": {"d1", "d2", "d3"}, "b": {"d1", "d2", "d3"}}
    out = add_similarity_provenance([edge], [("d1", "d2")], doc_index)
    e = out[0]
    assert e.provenance == frozenset({"cooccurrence", "similarity"})  # token kept (strength>0)
    # 6 ordered candidate pairs among {d1,d2,d3}; 2 linked (d1->d2, d2->d1) → 2/6.
    assert dict(e.provenance_strength)["similarity"] == pytest.approx(1 / 3, abs=1e-6)


def test_provenance_strength_saturated_graph_is_one() -> None:
    # Every candidate endpoint-doc pair is linked → strength pins at 1.0 (the honest
    # "no discrimination on a saturated graph" case; R4 payoff is on partial graphs).
    edge = SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.0, 2)
    doc_index = {"a": {"d1", "d2"}, "b": {"d1", "d2"}}
    out = add_citation_provenance([edge], [("d1", "d2")], doc_index)
    assert dict(out[0].provenance_strength)["citation"] == 1.0


def test_provenance_strength_absent_when_token_not_added() -> None:
    # Concepts share only one doc → no da≠db candidate pair → no token, no strength.
    edge = SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.0, 1)
    doc_index = {"a": {"d1"}, "b": {"d1"}}
    out = add_citation_provenance([edge], [("d1", "d2")], doc_index)
    assert out[0].provenance == frozenset({"cooccurrence"})  # unchanged
    assert out[0].provenance_strength == ()  # co-occurrence carries no strength


# ---- edge weight (Decision 5 + R4 graded tiebreak) -------------------------


def test_edge_weight_deterministic_and_ranks_multiprovenance_higher() -> None:
    single = edge_weight(frozenset({"cooccurrence"}), 5)
    assert single == edge_weight(frozenset({"cooccurrence"}), 5)  # deterministic
    multi = edge_weight(frozenset({"cooccurrence", "citation"}), 1)
    assert multi > single  # provenance count dominates co-occurrence count
    assert single < 2.0 <= multi
    # equal provenance → more co-occurrence chunks ranks higher
    assert edge_weight(frozenset({"cooccurrence"}), 10) > edge_weight(
        frozenset({"cooccurrence"}), 1
    )


def test_edge_weight_strength_refines_tiebreak_within_band() -> None:
    # Same tokens + same co-occurrence count → the graded strength breaks the tie...
    prov = frozenset({"cooccurrence", "citation"})
    strong = edge_weight(prov, 3, (("citation", 1.0),))
    weak = edge_weight(prov, 3, (("citation", 0.1),))
    assert strong > weak
    assert weak >= 2.0 and strong < 3.0  # ...but both stay inside the 2-token band


def test_edge_weight_token_count_dominates_strength() -> None:
    # The locked invariant: a co-occurrence-only edge, even with a huge co-occurrence count,
    # never outranks a 2-token edge with the weakest possible strength.
    single = edge_weight(frozenset({"cooccurrence"}), 10_000)
    multi = edge_weight(frozenset({"cooccurrence", "citation"}), 1, (("citation", 0.0),))
    assert single < 2.0 <= multi


# ---- serialisation round-trip (Decision 7 carry-over) ----------------------


def test_skeleton_dict_roundtrip_is_exact() -> None:
    nodes = [
        ConceptNode("a", "Alpha", ("d1",), 0, -1),
        ConceptNode("b", "Beta", ("d1",), 0, -1),
        ConceptNode("c", "Gamma", (), 0, -1),  # isolated curated concept
    ]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "citation"}),
            2.5,
            3,
            provenance_strength=(("citation", 0.4),),
            stance_by_doc=(("d1", "supports"),),
            relation="uses",
        )
    ]
    sk = analyze_skeleton(nodes, edges, seed=42)
    back = skeleton_from_dict(skeleton_to_dict(sk))

    assert skeleton_to_dict(back) == skeleton_to_dict(sk)
    assert back.edges[0].provenance == frozenset({"cooccurrence", "citation"})
    assert back.edges[0].provenance_strength == (("citation", 0.4),)  # R4 round-trips exactly
    assert back.edges[0].stance_by_doc == (("d1", "supports"),)
    assert back.edges[0].relation == "uses"


# ---- G3 (SPRINT-003) — doc_years threading, round-trip, back-compat --------


def test_doc_years_round_trips_via_skeleton_meta() -> None:
    # Threaded at the skeleton/meta level (not a new ConceptNode field — see the sprint's
    # blast-radius note); skeleton_to_dict/from_dict already carry `meta` verbatim.
    nodes = [ConceptNode("a", "Alpha", ("d1", "d2"), 0, -1)]
    edges: list[SkeletonEdge] = []
    sk = analyze_skeleton(
        nodes, edges, seed=42, meta_extra={"doc_years": {"d1": 2020, "d2": 2022}}
    )
    back = skeleton_from_dict(skeleton_to_dict(sk))
    assert back.meta["doc_years"] == {"d1": 2020, "d2": 2022}


def test_year_less_skeleton_json_still_loads_and_has_no_doc_years_key() -> None:
    # A pre-G3 skeleton.json has no "doc_years" key in meta at all — back-compat invariant.
    nodes = [ConceptNode("a", "Alpha", ("d1",), 0, -1)]
    sk = analyze_skeleton(nodes, [], seed=42)  # no meta_extra at all
    back = skeleton_from_dict(skeleton_to_dict(sk))
    assert "doc_years" not in back.meta


def test_graph_version_changes_when_doc_years_change() -> None:
    nodes = [
        ConceptNode("a", "Alpha", ("d1", "d2"), 0, -1),
        ConceptNode("b", "Beta", ("d1", "d2"), 0, -1),
    ]
    edges = [
        SkeletonEdge(
            "a", "b", frozenset({"cooccurrence"}), 1.5, 2, stance_by_doc=(("d1", "supports"),)
        )
    ]
    s1 = analyze_skeleton(nodes, edges, seed=42, meta_extra={"doc_years": {"d1": 2020}})
    s2 = analyze_skeleton(nodes, edges, seed=42, meta_extra={"doc_years": {"d1": 2021}})
    s3 = analyze_skeleton(nodes, edges, seed=42, meta_extra={"doc_years": {"d1": 2020}})
    assert s1.meta["graph_version"] != s2.meta["graph_version"]  # a year backfill busts the cache
    assert (
        s1.meta["graph_version"] == s3.meta["graph_version"]
    )  # identical inputs -> identical hash


# ---- Louvain determinism (ADR-1) -------------------------------------------


def test_louvain_communities_deterministic_for_seed() -> None:
    nodes = [ConceptNode(c, c.upper(), ("d1",), 0, -1) for c in ("a", "b", "c", "d", "e", "f")]
    tri = [("a", "b"), ("a", "c"), ("b", "c"), ("d", "e"), ("d", "f"), ("e", "f")]
    edges = [SkeletonEdge(s, t, frozenset({"cooccurrence"}), 1.5, 3) for s, t in tri]
    s1 = analyze_skeleton(nodes, edges, seed=42)
    s2 = analyze_skeleton(nodes, edges, seed=42)

    assert skeleton_to_dict(s1) == skeleton_to_dict(s2)
    assert s1.meta["graph_version"] == s2.meta["graph_version"]
    assert len({n.community for n in s1.nodes}) == 2  # two disjoint triangles


def test_isolated_concept_is_a_zero_degree_node() -> None:
    nodes = [
        ConceptNode("a", "A", ("d1",), 0, -1),
        ConceptNode("b", "B", ("d1",), 0, -1),
        ConceptNode("lonely", "Lonely", (), 0, -1),
    ]
    edges = [SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.5, 2)]
    sk = analyze_skeleton(nodes, edges, seed=42)
    by_id = {n.id: n for n in sk.nodes}
    assert by_id["lonely"].degree == 0
    assert by_id["a"].degree == 1
