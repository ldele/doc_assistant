"""Guard tests for definition candidates (ADR-053, ROADMAP 93 slice a).

The contract the user set: definitions from the text, from a model and from the user live side by
side, **none overwrites another**, and the user chooses case by case. Every test here pins one half
of that — the extraction finds and grades, the store never replaces, choosing is undoable, and a
merge or the boot migration cannot lose a definition. No model, no network, no vector store: the
extraction takes its chunks as an argument.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from doc_assistant.db.models import (
    Base,
    Concept,
    ConceptDefinition,
    ConceptDefinitionEvent,
    Document,
)
from doc_assistant.knowledge.definitions import (
    DefinitionError,
    PassageHit,
    add_user_definition,
    choose_definition,
    dismiss_definition,
    extract_definitions,
    find_passages,
    find_usages,
    load_definitions,
    passage_evidence,
    restore_definition,
    sentence_spans,
    undo_last,
    write_passages,
)


@pytest.fixture
def temp_db(monkeypatch):
    """A fresh temp SQLite with the current schema, engine + session factory patched."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


# Built with chr() so the file stays plain ASCII-escaped: a newline, and two title-block marks.
NL = chr(10)
AST, DAGGER = chr(0x2217), chr(0x2020)

# A two-document mini-library: a survey that introduces the term, and a paper that uses it.
SURVEY = (
    "## 3.5.1 Knowledge Distillation"
    + NL
    + "Knowledge distillation refers to a general set of techniques "
    "where a smaller student model learns to mimic a larger teacher model [Hinton et al., 2015]. "
    "It is widely used. Knowledge distillation is often combined with pruning."
)
PAPER = (
    "We compress the ranker with knowledge distillation, following Hinton et al. (2015) closely. "
    "Our results hold across both datasets. "
    "In our set-up, knowledge distillation is a way to shrink the ranker."
)
CHUNKS = [
    ("survey:p0", "survey", SURVEY),
    ("paper:p0", "paper", PAPER),
]


def _texts(hits: list[PassageHit]) -> list[str]:
    return [h.text for h in hits]


# ============================================================
# Finding and grading — pure
# ============================================================


def test_every_candidate_is_a_verbatim_substring_of_its_chunk():
    """ADR-043: a quoted sentence is quoted, not rewritten — even the heading cut keeps it a
    substring of what the document says."""
    hits = find_passages([("kd", "knowledge distillation")], CHUNKS)["kd"]
    by_key = dict((k, t) for k, _, t in CHUNKS)
    assert hits
    for h in hits:
        assert h.text in by_key[h.chunk_key]


def test_the_defining_sentence_ranks_first_and_is_strong():
    hits = find_passages([("kd", "knowledge distillation")], CHUNKS)["kd"]
    top = hits[0]
    assert top.text.startswith("Knowledge distillation refers to")  # heading cut off the front
    assert top.form == "defining" and top.opens
    evidence = passage_evidence(top)
    assert evidence["grade"] == "strong"
    assert any("opens with the term" in r for r in evidence["reasons"])


def test_a_definition_shape_the_sentence_is_not_about_is_not_strong():
    """ "… within SPECTER is an item of future work" has the shape and says nothing."""
    chunks = [
        (
            "d:p0",
            "d",
            "Exploring how to use the full text within SPECTER is an item of "
            "future work. We call our model SPECTER, which learns paper embeddings.",
        )
    ]
    hits = find_passages([("s", "specter")], chunks)["s"]
    forms = {h.form: passage_evidence(h)["grade"] for h in hits}
    assert forms["coined"] == "strong"
    assert forms["defining"] == "some"


def test_et_al_does_not_end_a_sentence():
    spans = sentence_spans(PAPER)
    assert spans[0].endswith("closely.")
    assert "Hinton et al. (2015)" in spans[0]


def test_a_sentence_cut_by_the_chunk_boundary_is_not_offered():
    text = (
        "tail of the previous chunk's sentence ends here. "
        "A complete sentence about cre sits here. "
        "And this one is cut off by the"
    )
    assert sentence_spans(text) == ["A complete sentence about cre sits here."]


def test_reference_entries_and_title_blocks_are_not_prose():
    text = (
        "Information Retrieval, volume 14, issue five, pages 14(5):441-465, in 2011. "
        "Some Title Here Author One"
        + AST
        + " Author Two"
        + DAGGER
        + " wrote this line about re-ranking. "
        "Re-ranking reorders a candidate list with a stronger model."
    )
    assert sentence_spans(text) == ["Re-ranking reorders a candidate list with a stronger model."]


def test_nothing_after_the_references_heading_counts_as_a_mention():
    """A bibliography title ("Efficient … re-ranking for large web datasets.") reads as prose one
    sentence at a time; only its place in the document gives it away."""
    body = "The model reorders a candidate list with a stronger scorer. " * 20
    bibliography = (
        NL
        + "## References"
        + NL
        + "Efficient and effective spam filtering and re-ranking for large web datasets."
    )
    chunks = [("d:p0", "d", body), ("d:p1", "d", bibliography)]
    assert find_passages([("r", "re-ranking")], chunks) == {}
    assert find_usages([("r", "re-ranking")], chunks) == {}


def test_a_first_mention_is_a_usage_not_a_candidate():
    """On the user's labels a first mention defined the term 3 times in 45
    (definition_labels_2026-09-22.md): it shows how the library uses a word, so it is kept apart
    from the options, and a sentence shaped as a definition is never offered as a usage."""
    hits = find_passages([("kd", "knowledge distillation")], CHUNKS)["kd"]
    assert {h.form for h in hits} == {"defining"}
    assert all("We compress" not in h.text for h in hits)
    uses = find_usages([("kd", "knowledge distillation")], CHUNKS)["kd"]
    # two mentions each; the tie goes to the lower document id
    assert [(u.document_id, u.doc_rank) for u in uses] == [("paper", 1), ("survey", 2)]
    assert uses[0].text.startswith("We compress the ranker")
    assert uses[1].text == "Knowledge distillation is often combined with pruning."
    assert not {u.text for u in uses} & {h.text for h in hits}
    assert find_usages([("kd", "knowledge distillation")], CHUNKS, per_concept=1)["kd"] == uses[:1]


def test_denoted_as_a_notation_is_not_coining():
    chunks = [
        ("d:p0", "d", "Expansion on top of BM25 is popular (usually denoted as BM25 + RM3).")
    ]
    hits = find_passages([("b", "bm25")], chunks).get("b", [])
    assert all(h.form != "coined" for h in hits)


def test_a_concept_that_appears_nowhere_gets_nothing_and_an_empty_library_is_fine():
    assert find_passages([("x", "quantum chromodynamics")], CHUNKS) == {}
    assert find_passages([("x", "anything")], []) == {}


# ============================================================
# The store: nothing overwrites anything
# ============================================================


def _concept(session, cid="kd", label="knowledge distillation"):
    session.add(Concept(id=cid, label=label, kind="concept"))
    session.add(
        Document(
            id="survey",
            filename="s.pdf",
            source_original="s",
            doc_hash="1",
            format="pdf",
            title="A survey",
        )
    )
    session.add(
        Document(id="paper", filename="p.pdf", source_original="p", doc_hash="2", format="pdf")
    )
    session.flush()


def test_extraction_is_idempotent_and_never_touches_a_decision(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s)
    first = extract_definitions(apply=True, chunks=CHUNKS)
    assert first.n_added >= 2
    with session_scope() as s:
        rows = s.execute(select(ConceptDefinition)).scalars().all()
        choose_definition(s, rows[0].id)
        dismiss_definition(s, rows[1].id)
        chosen_id, dismissed_id = rows[0].id, rows[1].id
    again = extract_definitions(apply=True, chunks=CHUNKS)
    assert again.n_added == 0
    with session_scope() as s:
        assert s.get(ConceptDefinition, chosen_id).status == "chosen"
        assert s.get(ConceptDefinition, dismissed_id).status == "dismissed"


def test_dry_run_writes_nothing(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s)
    result = extract_definitions(chunks=CHUNKS)
    assert result.n_hits >= 2 and not result.applied
    with session_scope() as s:
        assert s.execute(select(ConceptDefinition)).scalars().all() == []


def test_writing_your_own_keeps_the_previous_choice_beside_it(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s)
        first = add_user_definition(s, "kd", "A teacher model trains a smaller student.")
        second = add_user_definition(s, "kd", "Compressing a model by imitation.")
        assert s.get(Concept, "kd").definition == "Compressing a model by imitation."
        assert s.get(ConceptDefinition, first.id).status == "suggested"  # kept, not replaced
        assert second.status == "chosen"
        # the same words twice are one candidate, not two
        again = add_user_definition(s, "kd", "Compressing a model by imitation.")
        assert again.id == second.id
        assert len(s.execute(select(ConceptDefinition)).scalars().all()) == 2
        with pytest.raises(DefinitionError):
            add_user_definition(s, "kd", "   ")


def test_undo_walks_back_one_step_at_a_time(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s)
        write_passages(s, find_passages([("kd", "knowledge distillation")], CHUNKS))
        passage = s.execute(select(ConceptDefinition)).scalars().first()
        mine = add_user_definition(s, "kd", "My own words.")
        choose_definition(s, passage.id)  # step 2: switch to the passage
        dismiss_definition(s, passage.id)  # step 3: dismiss it — no definition left
        assert s.get(Concept, "kd").definition is None

        undo_last(s, "kd")  # back to step 2: the passage chosen
        assert s.get(ConceptDefinition, passage.id).status == "chosen"
        assert s.get(Concept, "kd").definition == passage.text
        undo_last(s, "kd")  # back to step 1: my own words chosen
        assert s.get(ConceptDefinition, mine.id).status == "chosen"
        assert s.get(Concept, "kd").definition == "My own words."
        undo_last(s, "kd")  # back to nothing chosen
        assert s.get(Concept, "kd").definition is None
        assert undo_last(s, "kd") is None  # nothing left to undo
        # every candidate is still there
        assert len(s.execute(select(ConceptDefinition)).scalars().all()) >= 2


def test_restore_brings_a_dismissed_candidate_back(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s)
        row = add_user_definition(s, "kd", "Words.", choose=False)
        dismiss_definition(s, row.id)
        restore_definition(s, row.id)
        assert s.get(ConceptDefinition, row.id).status == "suggested"
        with pytest.raises(DefinitionError):
            restore_definition(s, row.id)  # only a dismissed one can be restored


def test_the_mirror_always_matches_the_chosen_candidate(temp_db):
    """`Concept.definition` is read by the merge text and the semantic layer; it must never say
    something the chosen candidate does not."""
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s)
        write_passages(s, find_passages([("kd", "knowledge distillation")], CHUNKS))
        rows = s.execute(select(ConceptDefinition)).scalars().all()
        for row in rows:
            choose_definition(s, row.id)
            chosen = (
                s.execute(select(ConceptDefinition).where(ConceptDefinition.status == "chosen"))
                .scalars()
                .all()
            )
            assert [c.id for c in chosen] == [row.id]
            assert s.get(Concept, "kd").definition == row.text
        events = s.execute(select(ConceptDefinitionEvent)).scalars().all()
        assert [e.action for e in events] == ["chose"] * len(rows)


def test_the_read_model_orders_candidates_and_says_when_evidence_is_thin(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s)
        s.add(Concept(id="lonely", label="quantum chromodynamics", kind="concept"))
        s.add(Concept(id="field", label="Physics", kind="domain"))
        write_passages(s, find_passages([("kd", "knowledge distillation")], CHUNKS))
        view = load_definitions(s, "kd")
        assert view is not None and not view.thin and view.extracted
        assert view.candidates[0].grade == "strong"
        assert view.candidates[0].document_title == "A survey"
        assert view.candidates[0].chunk_key == "survey:p0"
        lonely = load_definitions(s, "lonely")
        assert lonely is not None and lonely.thin and not lonely.extracted
        assert load_definitions(s, "field") is None  # a taxonomy field is not a concept
        assert load_definitions(s, "nope") is None


# ============================================================
# Nothing else can lose a definition
# ============================================================


def test_the_boot_migration_turns_an_old_definition_into_a_chosen_candidate(temp_db):
    from doc_assistant.db.migrations import _migrate_legacy_definitions
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        s.add(Concept(id="old", label="dbs", kind="concept", definition="Deep brain stimulation."))
        s.add(Concept(id="none", label="beta", kind="concept"))
    assert _migrate_legacy_definitions(temp_db) is not None
    assert _migrate_legacy_definitions(temp_db) is None  # idempotent
    with session_scope() as s:
        rows = s.execute(select(ConceptDefinition)).scalars().all()
        assert [(r.concept_id, r.source, r.status, r.text) for r in rows] == [
            ("old", "user", "chosen", "Deep brain stimulation.")
        ]
        assert json.loads(rows[0].provenance_json) == {"legacy": True}
        # writing a new one keeps the old one beside it, one undo away
        add_user_definition(s, "old", "A neurosurgical therapy.")
        undo_last(s, "old")
        assert s.get(Concept, "old").definition == "Deep brain stimulation."


def test_add_concept_goes_through_the_candidates(temp_db):
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.concept_skeleton import add_concept

    cid = add_concept("pddl", definition="A planning language.")
    add_concept("pddl", definition="The Planning Domain Definition Language.")
    with session_scope() as s:
        rows = (
            s.execute(select(ConceptDefinition).where(ConceptDefinition.concept_id == cid))
            .scalars()
            .all()
        )
        assert sorted((r.text, r.status) for r in rows) == [
            ("A planning language.", "suggested"),
            ("The Planning Domain Definition Language.", "chosen"),
        ]


# ============================================================
# The one-concept path: the keyword index picks the documents
# ============================================================


def _mini_index(tmp_path, docs):
    """A real on-disk keyword index over ``{doc_hash: [parent texts]}`` — one child per parent."""
    from doc_assistant.sparse_index import SparseIndex

    pages = [
        (text, {"doc_hash": h, "parent_index": i, "parent_text": text})
        for h, parents in docs.items()
        for i, text in enumerate(parents)
    ]
    path = tmp_path / "sparse_index.sqlite3"
    SparseIndex.build(path, "fp", iter(pages)).close()
    return path


def test_the_keyword_index_picks_the_documents_including_hyphenated_forms(temp_db, tmp_path):
    """`actor-critic` is one word to the index and a mention of `actor` to the matcher; the prefix
    query keeps the two in agreement (357 of 357 concepts identical to a full read, 2026-09-21)."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.definitions import chunks_mentioning

    with session_scope() as s:
        for doc_id, doc_hash in (("d1", "h1"), ("d2", "h2"), ("d3", "h3")):
            s.add(
                Document(
                    id=doc_id,
                    filename=f"{doc_id}.pdf",
                    source_original=doc_id,
                    doc_hash=doc_hash,
                    format="pdf",
                )
            )
    index = _mini_index(
        tmp_path,
        {
            "h1": ["An actor chooses an action.", "Second block of d1."],
            "h2": ["We train in an actor-critic set-up."],
            "h3": ["Nothing relevant here."],
        },
    )
    chunks = chunks_mentioning(["actor"], index_file=index)
    assert chunks is not None
    assert sorted({doc for _, doc, _ in chunks}) == ["d1", "d2"]
    assert ("d1:p1", "d1", "Second block of d1.") in chunks  # the whole document, not just hits
    assert chunks_mentioning(["missing phrase"], index_file=index) == []
    assert chunks_mentioning(["actor"], index_file=tmp_path / "absent.sqlite3") is None


def test_a_tie_between_documents_does_not_depend_on_read_order():
    one = [
        ("b:p0", "b", "The pose is held for the whole of the recording session."),
        ("a:p0", "a", "Each pose is scored by the tracker on every frame it sees."),
    ]
    forward = [u.document_id for u in find_usages([("p", "pose")], one)["p"]]
    backward = [u.document_id for u in find_usages([("p", "pose")], list(reversed(one)))["p"]]
    assert forward == backward == ["a", "b"]  # one mention each: the lower id ranks first


def test_an_address_block_is_not_a_first_mention_but_a_year_is_still_prose():
    text = (
        "Neuroanatomy goes viral! Born, Department of Neurobiology, Harvard Medical School, "
        "220 Longwood Ave., Boston, MA 02115, USA. "
        "Researchers at the Institute showed in 2015 that viral tracers cross synapses."
    )
    assert sentence_spans(text) == [
        "Researchers at the Institute showed in 2015 that viral tracers cross synapses."
    ]


def test_a_sentence_that_names_the_term_is_strong_even_with_the_term_last():
    chunks = [
        (
            "d:p0",
            "d",
            "Feeding the query and the candidate text into one transformer together is called "
            "a cross-encoder. Segmenting documents into passages for ranking is commonly "
            "referred to as passage retrieval.",
        )
    ]
    for label in ("cross-encoder", "passage retrieval"):
        (hit, *_) = find_passages([("x", label)], chunks)["x"]
        assert hit.form == "named" and passage_evidence(hit)["grade"] == "strong"


def test_a_single_mention_reads_as_once():
    hit = PassageHit(
        concept_id="c",
        text="t",
        document_id="d",
        chunk_key="d:p0",
        form="defining",
        doc_mentions=1,
        doc_rank=2,
    )
    assert "From a document that mentions it once" in passage_evidence(hit)["reasons"]


def test_a_first_mention_stored_before_the_split_is_hidden_unless_chosen(temp_db):
    """Rows written while a first mention was a candidate stay in the table; the options stop
    showing them, and one the user chose stays theirs."""
    from doc_assistant.db.session import session_scope

    evidence = json.dumps(
        {"grade": "some", "reasons": ["The first sentence that uses it"], "form": "first_mention"}
    )
    with session_scope() as s:
        _concept(s)
        old = ConceptDefinition(
            concept_id="kd",
            text="We compress the ranker with knowledge distillation.",
            source="passage",
            provenance_key="paper:p0:old",
            provenance_json=json.dumps({"document_id": "paper", "chunk_key": "paper:p0"}),
            evidence_json=evidence,
            status="suggested",
        )
        s.add(old)
        s.flush()
        view = load_definitions(s, "kd")
        assert view is not None and view.candidates == () and not view.extracted
        choose_definition(s, old.id)
        view = load_definitions(s, "kd")
        assert view is not None and [c.id for c in view.candidates] == [old.id]
        assert view.chosen_id == old.id


def test_the_index_can_return_only_the_documents_that_use_a_label_most(temp_db, tmp_path):
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.definitions import chunks_mentioning

    with session_scope() as s:
        for doc_id, doc_hash in (("d1", "h1"), ("d2", "h2"), ("d3", "h3")):
            s.add(
                Document(
                    id=doc_id,
                    filename=f"{doc_id}.pdf",
                    source_original=doc_id,
                    doc_hash=doc_hash,
                    format="pdf",
                )
            )
    index = _mini_index(
        tmp_path,
        {
            "h1": ["The actor moves.", "The actor waits.", "The actor stops."],
            "h2": ["One actor here."],
            "h3": ["An actor acts.", "Another actor rests."],
        },
    )
    chunks = chunks_mentioning(["actor"], index_file=index, top_docs=2)
    assert chunks is not None and sorted({doc for _, doc, _ in chunks}) == ["d1", "d3"]


def test_usage_reads_a_few_documents_and_says_when_it_could_not(temp_db, tmp_path):
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.definitions import load_usage

    with session_scope() as s:
        s.add(Concept(id="a", label="actor", kind="concept"))
        s.add(Concept(id="field", label="Robotics", kind="domain"))
        s.add(
            Document(
                id="d1",
                filename="d1.pdf",
                source_original="d1",
                doc_hash="h1",
                format="pdf",
                title="Acting agents",
            )
        )
    index = _mini_index(
        tmp_path,
        {"h1": ["The actor moves to the goal.", "An actor is an agent that acts in a world."]},
    )
    usage = load_usage("a", index_file=index)
    assert usage is not None and usage.available
    (line,) = usage.examples
    assert line.text == "The actor moves to the goal."  # the definition shape is not a usage
    assert line.document_title == "Acting agents" and line.chunk_key == "d1:p0"
    assert line.doc_mentions == 2
    missing = load_usage("a", index_file=tmp_path / "absent.sqlite3")
    assert missing is not None and not missing.available and missing.examples == ()
    assert load_usage("field", index_file=index) is None
    assert load_usage("nope", index_file=index) is None
