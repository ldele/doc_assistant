"""Guards for written forms — how the library writes a concept's surface forms (ADR-053 D3).

The vote is the contract: documents vote, not occurrences; a capital at the start of a sentence
says nothing; a label is never rewritten. Pure cases first, then the table round trip on a temp
SQLite. Plain ASCII on purpose (the corpus's own spellings are ASCII here).
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from doc_assistant.db.models import Base, Concept
from doc_assistant.knowledge.written_forms import (
    WrittenForm,
    cased_label,
    derive_written_forms,
    label_spellings,
    load_written_forms,
    replace_written_forms,
    spelling_map,
)


def _by_form(forms: list[WrittenForm]) -> dict[str, WrittenForm]:
    return {f.form: f for f in forms}


def test_documents_vote_so_one_heavy_document_cannot_outvote_the_library() -> None:
    # One paper names a dataset PERSONA five times; two others use the word persona.
    heavy = " ".join("We evaluate on the PERSONA benchmark in this study." for _ in range(5))
    chunks = [
        ("a:p0", "a", heavy),
        ("b:p0", "b", "Each persona in the dialogue keeps a steady voice."),
        ("c:p0", "c", "The agent adopts a persona chosen by the user."),
    ]
    (form,) = derive_written_forms([("p", "persona")], {}, chunks)
    assert form.written == "persona"
    assert form.variants == {"PERSONA": 5, "persona": 2}
    assert form.votes == {"persona": 2, "PERSONA": 1}
    assert not form.cased


def test_din_is_not_din_and_cre_is_not_cre() -> None:
    # The user's two cases (definition labels 2026-09-22): dIN the interneuron and Din the join in
    # "vitamin Din"; Cre recombinase and CRE, the cAMP response element.
    chunks = [
        (
            "a:p0",
            "a",
            "Recordings from the dIN population show a steady rhythm. "
            "In every larva the dIN cells fire before the motor neurons.",
        ),
        ("b:p0", "b", "A diet with vitamin Din adults was not studied here."),
        ("c:p0", "c", "Expression of Cre recombinase was confirmed in all mice."),
        ("d:p0", "d", "We crossed the line with a second Cre driver for this work."),
        ("e:p0", "e", "The CRE element binds the activated protein complex."),
    ]
    forms = _by_form(derive_written_forms([("d", "din"), ("c", "cre")], {}, chunks))
    # One document each: the tie goes to the spelling used more (2 against 1).
    assert forms["din"].written == "dIN"
    assert forms["din"].votes == {"dIN": 1, "Din": 1}
    assert forms["cre"].written == "Cre"
    assert forms["cre"].votes == {"Cre": 2, "CRE": 1}
    assert forms["din"].cased and forms["cre"].cased


def test_a_capital_at_the_start_of_a_sentence_does_not_vote() -> None:
    chunks = [
        ("a:p0", "a", "Retrieval is the first stage of the whole pipeline."),
        ("a:p1", "a", "Retrieval quality bounds what the reader can answer."),
        ("b:p0", "b", "We tune dense retrieval with care in the second stage."),
        ("c:p0", "c", "Tokenizers split the text into pieces before indexing."),
    ]
    forms = _by_form(derive_written_forms([("r", "retrieval"), ("t", "tokenizers")], {}, chunks))
    assert forms["retrieval"].written == "retrieval"
    assert forms["retrieval"].uses == 1  # only the mid-sentence one
    assert "tokenizers" not in forms  # only ever the first word: no evidence, no written form


def test_aliases_get_their_own_written_forms() -> None:
    chunks = [("a:p0", "a", "We compare BM25 against the Okapi scoring used in older systems.")]
    forms = _by_form(derive_written_forms([("b", "bm25")], {"b": ["okapi"]}, chunks))
    assert forms["bm25"].written == "BM25"
    assert forms["okapi"].written == "Okapi"


def test_an_empty_library_has_no_written_forms() -> None:
    assert derive_written_forms([("x", "anything")], {}, []) == []
    assert (
        derive_written_forms([], {}, [("a:p0", "a", "Some prose sentence of fair length.")]) == []
    )


def test_cased_label_shows_the_written_form_only_when_it_is_the_labels_own() -> None:
    assert cased_label("din", "dIN") == "dIN"
    assert cased_label("cre", "Cre") == "Cre"
    assert cased_label("beta", "beta") is None  # lower case: show the label as stored
    assert cased_label("din", None) is None
    assert cased_label("din", "Cre") is None  # stale: the label changed since the last build


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


def _concepts(*labels: str) -> list[str]:
    from doc_assistant.db.session import session_scope

    ids = []
    with session_scope() as session:
        for label in labels:
            row = Concept(label=label)
            session.add(row)
            session.flush()
            ids.append(str(row.id))
    return ids


def test_the_table_is_replaced_whole_and_read_back(temp_db) -> None:
    din, beta = _concepts("din", "beta")
    first = [
        WrittenForm(din, "din", "dIN", uses=43, documents=2, votes={"dIN": 1, "Din": 1}),
        WrittenForm(beta, "beta", "beta", uses=127, documents=9),
    ]
    assert replace_written_forms(first) == 2
    assert load_written_forms() == spelling_map(first)
    assert label_spellings([(din, "din"), (beta, "beta")]) == {din: "dIN", beta: "beta"}

    # A later build that no longer sees beta leaves nothing of it behind.
    assert replace_written_forms(first[:1]) == 1
    assert load_written_forms() == {(din, "din"): "dIN"}
    assert load_written_forms([beta]) == {}


def test_a_library_never_built_reads_as_no_written_forms(temp_db) -> None:
    (din,) = _concepts("din")
    assert load_written_forms() == {}
    assert label_spellings([(din, "din")]) == {}
