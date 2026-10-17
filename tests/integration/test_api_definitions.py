"""Integration tests for the definitions router (ADR-053, ROADMAP 93 slice a).

The rules live in ``knowledge.definitions`` and are unit-tested in
``tests/unit/test_definitions.py``; here the HTTP side: every mutation answers with the refreshed
candidates, a candidate from another concept is a 404 rather than a silent write, and the
vocabulary search reaches concepts the graph does not show. Temp DB, fake controller — no model
load, no vector store, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.db.models import Concept
from doc_assistant.db.session import session_scope


class FakeController:
    def chunk_count(self) -> int:
        return 0


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield
    engine.dispose()


@pytest.fixture
def client(temp_db: None) -> TestClient:
    return TestClient(create_app(controller=FakeController()))


def _seed() -> None:
    with session_scope() as s:
        s.add(Concept(id="kd", label="knowledge distillation", kind="concept", graph_include=True))
        s.add(Concept(id="viral", label="viral", kind="concept", graph_include=False))
        s.add(Concept(id="virus", label="virus", kind="concept", graph_include=True))
        s.add(Concept(id="field", label="Virology", kind="domain"))


def test_a_concept_with_nothing_yet_says_so(client: TestClient) -> None:
    _seed()
    body = client.get("/api/concepts/kd/definitions").json()
    assert body["candidates"] == [] and body["chosen_id"] is None
    assert body["thin"] is True and body["extracted"] is False and body["can_undo"] is False
    assert client.get("/api/concepts/nope/definitions").status_code == 404
    assert client.get("/api/concepts/field/definitions").status_code == 404  # a field


def test_write_choose_dismiss_undo_round_trip(client: TestClient) -> None:
    _seed()
    first = client.post(
        "/api/concepts/kd/definitions", json={"text": "A student mimics a teacher."}
    )
    assert first.status_code == 201
    body = first.json()
    mine = body["chosen_id"]
    assert body["candidates"][0]["source"] == "user" and body["candidates"][0]["grade"] == "user"

    second = client.post(
        "/api/concepts/kd/definitions", json={"text": "Model compression.", "choose": False}
    ).json()
    other = next(c["id"] for c in second["candidates"] if c["id"] != mine)
    assert second["chosen_id"] == mine  # choose=False replaced nothing

    chosen = client.post(f"/api/concepts/kd/definitions/{other}/choose").json()
    assert chosen["chosen_id"] == other
    assert {c["id"]: c["status"] for c in chosen["candidates"]}[mine] == "suggested"

    dismissed = client.post(f"/api/concepts/kd/definitions/{other}/dismiss").json()
    assert dismissed["chosen_id"] is None

    undone = client.post("/api/concepts/kd/definitions/undo").json()
    assert undone["chosen_id"] == other
    client.post("/api/concepts/kd/definitions/undo")
    back = client.post("/api/concepts/kd/definitions/undo").json()
    assert back["chosen_id"] is None  # the first write's choice undone too
    assert len(back["candidates"]) == 2  # and nothing was ever deleted
    assert client.post("/api/concepts/kd/definitions/undo").status_code == 409


def test_a_candidate_of_another_concept_is_refused(client: TestClient) -> None:
    _seed()
    theirs = client.post("/api/concepts/virus/definitions", json={"text": "An infectious agent."})
    theirs_id = theirs.json()["chosen_id"]
    assert client.post(f"/api/concepts/kd/definitions/{theirs_id}/choose").status_code == 404
    assert client.post(f"/api/concepts/kd/definitions/{theirs_id}/dismiss").status_code == 404
    assert client.post(f"/api/concepts/virus/definitions/{theirs_id}/restore").status_code == 409


def test_an_empty_definition_is_refused(client: TestClient) -> None:
    _seed()
    assert client.post("/api/concepts/kd/definitions", json={"text": ""}).status_code == 422
    assert client.post("/api/concepts/kd/definitions", json={"text": "   "}).status_code == 400


def test_extraction_from_the_app_stores_suggestions_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel's "look in my library" runs the same extractor as the CLI, on this concept only.
    The vector store is replaced by two chunks; nothing gets chosen."""
    _seed()
    chunks = [
        ("s:p0", "s", "Knowledge distillation refers to training a small model on a large one."),
    ]
    monkeypatch.setattr(
        "doc_assistant.knowledge.definitions.chunks_mentioning", lambda _labels: chunks
    )
    body = client.post("/api/concepts/kd/definitions/extract").json()
    assert body["extracted"] is True and body["chosen_id"] is None
    (candidate,) = body["candidates"]
    assert candidate["source"] == "passage" and candidate["grade"] == "strong"
    assert candidate["chunk_key"] == "s:p0" and candidate["reasons"]
    again = client.post("/api/concepts/kd/definitions/extract").json()
    assert len(again["candidates"]) == 1  # idempotent


def test_usage_is_read_only_and_says_when_the_index_is_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel's "How your library uses it": plain uses, never candidates, nothing stored."""
    _seed()
    chunks = [
        (
            "s:p0",
            "s",
            "Knowledge distillation refers to training a small model on a large one. "
            "We then apply knowledge distillation to the ranker.",
        ),
    ]
    monkeypatch.setattr(
        "doc_assistant.knowledge.definitions.chunks_mentioning", lambda _labels, **_kw: chunks
    )
    body = client.get("/api/concepts/kd/usage").json()
    assert body["available"] is True
    (line,) = body["examples"]
    assert line["text"] == "We then apply knowledge distillation to the ranker."
    assert line["chunk_key"] == "s:p0" and line["doc_mentions"] == 2
    assert client.get("/api/concepts/kd/definitions").json()["candidates"] == []  # nothing stored
    assert client.get("/api/concepts/nope/usage").status_code == 404
    assert client.get("/api/concepts/field/usage").status_code == 404

    monkeypatch.setattr(
        "doc_assistant.knowledge.definitions.chunks_mentioning", lambda _labels, **_kw: None
    )
    assert client.get("/api/concepts/kd/usage").json() == {
        "concept_id": "kd",
        "available": False,
        "examples": [],
    }


def test_vocabulary_search_reaches_concepts_off_the_graph(client: TestClient) -> None:
    _seed()
    matches = client.get("/api/concepts/search", params={"q": "vir"}).json()
    assert [m["label"] for m in matches] == ["virus", "viral"]  # graph first among prefix matches
    assert matches[1]["on_graph"] is False
    assert [m["label"] for m in client.get("/api/concepts/search?q=viral").json()] == ["viral"]
    assert client.get("/api/concepts/search?q=").json() == []
    assert client.get("/api/concepts/search?q=%25").json() == []  # a literal %, not a wildcard


def test_vocabulary_search_carries_the_name_as_the_library_writes_it(client: TestClient) -> None:
    """ADR-054: the name on every surface. A hit shows `Cre` for the stored `cre` once a graph
    build has recorded how the library writes it; a row without a cased form shows its label."""
    from doc_assistant.db.models import ConceptWrittenForm

    with session_scope() as s:
        s.add(Concept(id="cre", label="cre", kind="concept", graph_include=True))
        s.add(Concept(id="crest", label="crest", kind="concept", graph_include=False))
        s.add(ConceptWrittenForm(concept_id="cre", form="cre", written="Cre"))
    matches = client.get("/api/concepts/search", params={"q": "cre"}).json()
    assert [(m["label"], m["written"], m["on_graph"]) for m in matches] == [
        ("cre", "Cre", True),
        ("crest", None, False),
    ]


# --- a term is shown, never written (ADR-054) -------------------------------------------------


def test_a_term_shows_what_the_library_says_and_stores_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`viral` is a term: a row nobody has taken on. "Look in my library" still answers with the
    passages it finds, marked unsaved, and a second read shows that nothing was stored."""
    _seed()
    chunks = [("s:p0", "s", "Viral refers to anything a virus produces, in this review's usage.")]
    monkeypatch.setattr(
        "doc_assistant.knowledge.definitions.chunks_mentioning", lambda _labels: chunks
    )
    bare = client.get("/api/concepts/viral/definitions").json()
    assert bare["is_concept"] is False and bare["candidates"] == []
    assert client.get("/api/concepts/kd/definitions").json()["is_concept"] is True

    body = client.post("/api/concepts/viral/definitions/extract").json()
    assert body["is_concept"] is False and body["extracted"] is True
    (candidate,) = body["candidates"]
    assert candidate["id"].startswith("unsaved:") and candidate["status"] == "suggested"
    assert candidate["source"] == "passage" and candidate["chunk_key"] == "s:p0"

    after = client.get("/api/concepts/viral/definitions").json()
    assert after["candidates"] == [] and after["extracted"] is False


def test_a_definition_write_on_a_term_is_refused_until_it_is_taken_on(client: TestClient) -> None:
    _seed()
    refused = client.post("/api/concepts/viral/definitions", json={"text": "Of a virus."})
    assert refused.status_code == 409 and "is a term" in refused.json()["detail"]
    assert client.post("/api/concepts/viral/definitions/undo").status_code == 409
    for action in ("choose", "dismiss", "restore"):
        r = client.post(f"/api/concepts/viral/definitions/anything/{action}")
        assert r.status_code == 409, action
    assert client.get("/api/concepts/viral/definitions").json()["candidates"] == []

    # An unknown id is still a 404, not a 409: there is nothing to take on.
    assert client.post("/api/concepts/nope/definitions", json={"text": "x"}).status_code == 404

    with session_scope() as s:
        s.get(Concept, "viral").graph_include = True  # type: ignore[union-attr]
    taken_on = client.post("/api/concepts/viral/definitions", json={"text": "Of a virus."})
    assert taken_on.status_code == 201 and taken_on.json()["is_concept"] is True


def test_signals_read_the_index_and_offer_what_the_text_says(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /signals over a real on-disk keyword index: an offer with its sentence and its two
    forms marked, a cited author with an "et al." sentence, and the place a label no sentence
    uses does occur. Read-only, for a term as much as a concept; nothing is stored."""
    from doc_assistant.db.models import Document
    from doc_assistant.sparse_index import SparseIndex

    with session_scope() as s:
        s.add(Concept(id="dbs", label="dbs", kind="concept", graph_include=True))
        s.add(Concept(id="cohan", label="cohan", kind="concept", graph_include=False))
        s.add(Concept(id="ap10k", label="ap10k", kind="concept", graph_include=False))
        s.add(Concept(id="field", label="Neurology", kind="domain"))
        s.add(
            Document(
                id="d1",
                filename="d1.pdf",
                source_original="d1",
                doc_hash="h1",
                format="pdf",
                title="Stimulation in practice",
            )
        )
        s.add(
            Document(id="d2", filename="d2.pdf", source_original="d2", doc_hash="h2", format="pdf")
        )
    dbs = (
        "For these patients, neuromodulation therapies such as deep brain stimulation (DBS) "
        "have been investigated in several centres."
    )
    texts = [
        ("h1", 0, f"{dbs} The benchmark of Cohan et al. (2020) covers seven tasks."),
        ("h2", 0, "Cohan et al. also release the training data used in every one of their runs."),
        ("h2", 1, "| Model | AP10k |\n|---|---|\n| ours | 0.65 |"),
    ]
    pages = [
        (text, {"doc_hash": h, "parent_index": i, "parent_text": text}) for h, i, text in texts
    ]
    index = tmp_path / "sparse_index.sqlite3"
    SparseIndex.build(index, "fp", iter(pages)).close()
    monkeypatch.setattr("doc_assistant.sparse_index.index_path", lambda _root: index)

    body = client.get("/api/concepts/dbs/signals").json()
    assert body["available"] is True and body["silent"] is False
    assert (body["documents_read"], body["documents_matched"]) == (1, 1)
    (offer,) = body["offers"]
    assert (offer["form"], offer["beside"], offer["kind"]) == (
        "deep brain stimulation",
        "DBS",
        "long",
    )
    assert offer["known"] is False and offer["documents"] == 1
    example = offer["example"]
    assert (
        example["document_title"] == "Stimulation in practice" and example["chunk_key"] == "d1:p0"
    )
    assert [example["text"][a:b] for a, b in example["marks"]] == ["deep brain stimulation", "DBS"]
    assert body["cited"] is None and body["places"] == []

    cohan = client.get("/api/concepts/cohan/signals").json()  # a term: offered all the same
    assert cohan["cited"]["et_al"] == 2 and cohan["cited"]["mentions"] == 2
    cited = cohan["cited"]["example"]
    assert cited["chunk_key"] == "d1:p0"
    assert [cited["text"][a:b] for a, b in cited["marks"]] == ["Cohan"]

    silent = client.get("/api/concepts/ap10k/signals").json()
    assert silent["silent"] is True and silent["mentions"] == 0
    (place,) = silent["places"]
    assert (place["place"], place["mentions"], place["documents"]) == ("table", 1, 1)
    assert place["example"]["document_title"] == "d2.pdf"  # no title: the filename

    assert client.get("/api/concepts/dbs/definitions").json()["candidates"] == []  # nothing stored
    assert client.get("/api/concepts/nope/signals").status_code == 404
    assert client.get("/api/concepts/field/signals").status_code == 404

    monkeypatch.setattr(
        "doc_assistant.sparse_index.index_path", lambda _root: tmp_path / "absent.sqlite3"
    )
    missing = client.get("/api/concepts/dbs/signals").json()
    assert missing["available"] is False and missing["offers"] == [] and missing["silent"] is False
