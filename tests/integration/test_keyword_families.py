"""Integration tests for tag families (ADR-015, feature-tag-families.md PR-1).

A family is a curated Concept whose ConceptAlias rows carry member Keyword names. Covers:
CRUD (create/rename/add-remove-member/delete), the move-on-reassign invariant, the
union doc_count, and the API routes (200/404/400). Temp file-backed SQLite, no LLM.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base, Concept, ConceptAlias, Document, Keyword
from doc_assistant.db.session import session_scope


class _FakeController:
    def chunk_count(self) -> int:
        return 0


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed_doc_with_keywords(filename: str, *keyword_names: str) -> str:
    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=f"/tmp/{filename}",
            doc_hash=f"hash-{filename}",
            format="pdf",
        )
        for name in keyword_names:
            keyword = session.execute(
                select(Keyword).where(Keyword.name == name)
            ).scalar_one_or_none()
            if keyword is None:
                keyword = Keyword(name=name, source="extracted")
                session.add(keyword)
            doc.keywords.append(keyword)
        session.add(doc)
        session.flush()
        return str(doc.id)


def _client() -> TestClient:
    return TestClient(create_app(controller=_FakeController()))  # type: ignore[arg-type]


# --- library.py CRUD -------------------------------------------------------------------------- #


def test_create_family_seeds_concept_and_aliases(env: Path) -> None:
    from doc_assistant.library import create_keyword_family

    _seed_doc_with_keywords("a.pdf", "llm", "llms")
    family = create_keyword_family("large language models", ["llm", "llms"])
    assert family.canonical == "large language models"
    assert family.aliases == ["llm", "llms"]
    assert family.doc_count == 1
    with session_scope() as session:
        assert session.execute(select(Concept)).scalar_one().source == "manual"
        assert {a.alias for a in session.execute(select(ConceptAlias)).scalars()} == {
            "llm",
            "llms",
        }


def test_create_family_is_idempotent_by_canonical(env: Path) -> None:
    from doc_assistant.library import create_keyword_family

    first = create_keyword_family("large language models", ["llm"])
    second = create_keyword_family("large language models", ["llms"])
    assert first.id == second.id
    assert second.aliases == ["llm", "llms"]  # union across calls, never replaced
    with session_scope() as session:
        assert len(list(session.execute(select(Concept)).scalars())) == 1


def test_doc_count_is_union_over_all_members(env: Path) -> None:
    from doc_assistant.library import create_keyword_family

    _seed_doc_with_keywords("a.pdf", "llm")
    _seed_doc_with_keywords("b.pdf", "llms")
    _seed_doc_with_keywords("c.pdf", "llm", "llms")  # carries both -> counted once
    _seed_doc_with_keywords("d.pdf", "bm25")  # unrelated -> excluded

    family = create_keyword_family("large language models", ["llm", "llms"])
    assert family.doc_count == 3


def test_add_member_moves_keyword_off_other_family(env: Path) -> None:
    from doc_assistant.library import add_family_member, create_keyword_family, get_keyword_family

    llm_family = create_keyword_family("large language models", ["llm", "llms"])
    other_family = create_keyword_family("connectome", [])

    updated_other = add_family_member(other_family.id, "llms")
    assert updated_other is not None
    assert "llms" in updated_other.aliases

    refreshed_llm = get_keyword_family(llm_family.id)
    assert refreshed_llm is not None
    assert "llms" not in refreshed_llm.aliases  # moved, not duplicated
    assert refreshed_llm.aliases == ["llm"]


def test_add_member_is_idempotent(env: Path) -> None:
    from doc_assistant.library import add_family_member, create_keyword_family

    family = create_keyword_family("large language models", ["llm"])
    again = add_family_member(family.id, "llm")
    assert again is not None
    assert again.aliases == ["llm"]  # no duplicate


def test_add_member_unknown_family_returns_none(env: Path) -> None:
    from doc_assistant.library import add_family_member

    assert add_family_member("nope", "llm") is None


def test_remove_member_is_a_noop_when_absent(env: Path) -> None:
    from doc_assistant.library import create_keyword_family, remove_family_member

    family = create_keyword_family("large language models", ["llm"])
    result = remove_family_member(family.id, "not-a-member")
    assert result is not None
    assert result.aliases == ["llm"]


def test_remove_member_drops_the_alias(env: Path) -> None:
    from doc_assistant.library import create_keyword_family, remove_family_member

    family = create_keyword_family("large language models", ["llm", "llms"])
    result = remove_family_member(family.id, "llms")
    assert result is not None
    assert result.aliases == ["llm"]


def test_rename_family(env: Path) -> None:
    from doc_assistant.library import create_keyword_family, rename_keyword_family

    family = create_keyword_family("llm", [])
    renamed = rename_keyword_family(family.id, "large language models")
    assert renamed is not None
    assert renamed.canonical == "large language models"


def test_rename_unknown_family_returns_none(env: Path) -> None:
    from doc_assistant.library import rename_keyword_family

    assert rename_keyword_family("nope", "x") is None


def test_delete_family(env: Path) -> None:
    from doc_assistant.library import create_keyword_family, delete_keyword_family

    family = create_keyword_family("large language models", ["llm"])
    assert delete_keyword_family(family.id) is True
    assert delete_keyword_family(family.id) is False  # already gone
    with session_scope() as session:
        assert session.execute(select(Concept)).first() is None
        assert session.execute(select(ConceptAlias)).first() is None  # cascade


def test_list_keyword_families_sorted_by_canonical(env: Path) -> None:
    from doc_assistant.library import create_keyword_family, list_keyword_families

    create_keyword_family("zeta family", [])
    create_keyword_family("alpha family", [])
    families = list_keyword_families()
    assert [f.canonical for f in families] == ["alpha family", "zeta family"]


# --- API routes ------------------------------------------------------------------------------ #


def test_routes_create_rename_and_get(env: Path) -> None:
    client = _client()
    r = client.post(
        "/api/library/keyword-families", json={"canonical": "llm family", "members": ["llm"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["canonical"] == "llm family" and body["aliases"] == ["llm"]

    r = client.patch(f"/api/library/keyword-families/{body['id']}", json={"canonical": "renamed"})
    assert r.status_code == 200 and r.json()["canonical"] == "renamed"

    r = client.get("/api/library/keyword-families")
    assert r.status_code == 200
    assert [f["canonical"] for f in r.json()] == ["renamed"]


def test_routes_member_add_remove(env: Path) -> None:
    client = _client()
    family_id = client.post(
        "/api/library/keyword-families", json={"canonical": "llm family", "members": []}
    ).json()["id"]

    r = client.post(f"/api/library/keyword-families/{family_id}/members", json={"keyword": "llm"})
    assert r.status_code == 200 and r.json()["aliases"] == ["llm"]

    r = client.delete(f"/api/library/keyword-families/{family_id}/members/llm")
    assert r.status_code == 200 and r.json()["aliases"] == []


def test_routes_delete(env: Path) -> None:
    client = _client()
    family_id = client.post(
        "/api/library/keyword-families", json={"canonical": "llm family", "members": []}
    ).json()["id"]
    assert client.delete(f"/api/library/keyword-families/{family_id}").json() == {"ok": True}
    assert client.delete(f"/api/library/keyword-families/{family_id}").status_code == 404


def test_routes_404_on_unknown_family(env: Path) -> None:
    client = _client()
    assert (
        client.patch("/api/library/keyword-families/nope", json={"canonical": "x"}).status_code
        == 404
    )
    assert (
        client.post(
            "/api/library/keyword-families/nope/members", json={"keyword": "x"}
        ).status_code
        == 404
    )
    assert client.delete("/api/library/keyword-families/nope/members/x").status_code == 404
    assert client.delete("/api/library/keyword-families/nope").status_code == 404


def test_routes_400_on_blank_canonical(env: Path) -> None:
    client = _client()
    r = client.post("/api/library/keyword-families", json={"canonical": "   ", "members": []})
    assert r.status_code == 400
