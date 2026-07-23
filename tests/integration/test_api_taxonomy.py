"""Integration tests for the taxonomy router (ADR-028 increment 2a).

Drive the read + write endpoints over a temp DB through ``create_app`` with a fake controller — no
model load, no network, no LLM. The write seam's invariants (cycle rejection, domain-only document
attach) are unit-tested in ``tests/unit/test_taxonomy.py``; here we prove the HTTP status mapping.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.db.models import Concept, Document
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


def _seed_field(fid: str, label: str, *, kind: str = "domain") -> None:
    with session_scope() as s:
        s.add(Concept(id=fid, label=label, kind=kind))


def test_get_taxonomy_zero_state(client: TestClient) -> None:
    _seed_field("div", "Information and computing sciences")
    _seed_field("grp", "Machine learning")
    _seed_field("c1", "Embeddings", kind="concept")
    client.post(
        "/api/taxonomy/hierarchy",
        json={"source_id": "grp", "target_id": "div", "type": "in_field"},
    )

    r = client.get("/api/taxonomy")
    assert r.status_code == 200
    body = r.json()
    assert len(body["fields"]) == 2  # only domains are fields; the concept is not
    assert body["roots"] == ["div"]
    assert body["n_concepts_total"] == 1
    assert body["n_unassigned_concepts"] == 1
    assert all(f["n_concepts_rollup"] == 0 for f in body["fields"])


def test_attach_concept_rolls_up(client: TestClient) -> None:
    _seed_field("div", "ICS")
    _seed_field("grp", "ML")
    _seed_field("c1", "Embeddings", kind="concept")
    client.post(
        "/api/taxonomy/hierarchy",
        json={"source_id": "grp", "target_id": "div", "type": "in_field"},
    )

    # attach the concept to the group via the same hierarchy endpoint
    r = client.post(
        "/api/taxonomy/hierarchy",
        json={"source_id": "c1", "target_id": "grp", "type": "in_field"},
    )
    assert r.status_code == 201

    fields = {f["id"]: f for f in client.get("/api/taxonomy").json()["fields"]}
    assert fields["grp"]["n_concepts_direct"] == 1
    assert fields["div"]["n_concepts_rollup"] == 1  # crossed grp -> div


def test_hierarchy_cycle_is_409(client: TestClient) -> None:
    _seed_field("a", "A")
    _seed_field("b", "B")
    client.post(
        "/api/taxonomy/hierarchy", json={"source_id": "a", "target_id": "b", "type": "in_field"}
    )
    r = client.post(
        "/api/taxonomy/hierarchy", json={"source_id": "b", "target_id": "a", "type": "in_field"}
    )
    assert r.status_code == 409


def test_hierarchy_bad_type_is_422(client: TestClient) -> None:
    _seed_field("a", "A")
    _seed_field("b", "B")
    # `type` is a Literal on the request model — an unknown value is a 422 (pydantic), not a 400.
    r = client.post(
        "/api/taxonomy/hierarchy", json={"source_id": "a", "target_id": "b", "type": "related"}
    )
    assert r.status_code == 422


def test_hierarchy_missing_id_is_404(client: TestClient) -> None:
    _seed_field("a", "A")
    r = client.post(
        "/api/taxonomy/hierarchy",
        json={"source_id": "a", "target_id": "ghost", "type": "in_field"},
    )
    assert r.status_code == 404


def test_remove_hierarchy_is_idempotent(client: TestClient) -> None:
    _seed_field("a", "A")
    _seed_field("b", "B")
    client.post(
        "/api/taxonomy/hierarchy", json={"source_id": "a", "target_id": "b", "type": "in_field"}
    )
    r1 = client.request(
        "DELETE",
        "/api/taxonomy/hierarchy",
        json={"source_id": "a", "target_id": "b", "type": "in_field"},
    )
    assert r1.status_code == 200 and r1.json()["removed"] == 1
    r2 = client.request(
        "DELETE",
        "/api/taxonomy/hierarchy",
        json={"source_id": "a", "target_id": "b", "type": "in_field"},
    )
    assert r2.json()["removed"] == 0


def test_attach_document_field(client: TestClient) -> None:
    _seed_field("grp", "ML")
    _seed_field("concept-node", "Embeddings", kind="concept")
    with session_scope() as s:
        s.add(Document(id="d1", filename="p.pdf", source_original="p", doc_hash="h", format="pdf"))

    # to a domain -> 201, idempotent
    r = client.post("/api/taxonomy/documents/d1/fields/grp")
    assert r.status_code == 201
    assert client.post("/api/taxonomy/documents/d1/fields/grp").status_code == 201
    # to a non-domain node -> 400
    assert client.post("/api/taxonomy/documents/d1/fields/concept-node").status_code == 400
    # non-existent document -> 404
    assert client.post("/api/taxonomy/documents/ghost/fields/grp").status_code == 404


def test_field_detail_404_for_non_field(client: TestClient) -> None:
    _seed_field("grp", "ML")
    assert client.get("/api/taxonomy/fields/grp").status_code == 200
    _seed_field("c1", "Embeddings", kind="concept")
    assert client.get("/api/taxonomy/fields/c1").status_code == 404  # a concept is not a field
    assert client.get("/api/taxonomy/fields/ghost").status_code == 404
