"""Integration tests for tag families (ADR-015, feature-tag-families.md PR-1 + PR-2).

A family is a curated Concept whose ConceptAlias rows carry member Keyword names. Covers:
CRUD (create/rename/add-remove-member/delete), the move-on-reassign invariant, the
union doc_count, and the API routes (200/404/400). Temp file-backed SQLite, no LLM. The
PR-2 ``/detect`` route section uses a fake ``.rag.embeddings`` stub — no real bge load.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

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


class _FakeEmbeddings:
    """Deterministic toy embedder for the /detect route — no real bge load. Known keywords get a
    fixed near-cosine-1 pair (connectome/connectomics); everything else gets its own orthogonal
    one-hot dimension, so unrelated keywords never spuriously cluster."""

    _KNOWN: ClassVar[dict[str, tuple[float, float]]] = {
        "connectome": (1.0, 0.0),
        "connectomics": (0.99, 0.14),
    }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        dim = 2 + len(texts)
        vectors: list[list[float]] = []
        for i, t in enumerate(texts):
            vec = [0.0] * dim
            if t in self._KNOWN:
                vec[0], vec[1] = self._KNOWN[t]
            else:
                vec[2 + i] = 1.0
            vectors.append(vec)
        return vectors


class _FakeRag:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


class _FakeControllerWithRag(_FakeController):
    def __init__(self) -> None:
        self.rag = _FakeRag()


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


# --- detect route (PR-2) -------------------------------------------------------------------- #


def _client_with_rag() -> TestClient:
    return TestClient(create_app(controller=_FakeControllerWithRag()))  # type: ignore[arg-type]


def test_detect_route_proposes_both_tiers_and_excludes_familied(env: Path) -> None:
    _seed_doc_with_keywords("a.pdf", "llm", "llms", "connectome", "connectomics", "bm25", "rag")
    client = _client_with_rag()
    client.post(
        "/api/library/keyword-families",
        json={"canonical": "retrieval-augmented generation", "members": ["rag"]},
    )

    r = client.post("/api/library/keyword-families/detect")
    assert r.status_code == 200
    proposals = r.json()
    assert {p["tier"] for p in proposals} == {"morphological", "embedding"}

    morph = next(p for p in proposals if p["tier"] == "morphological")
    assert morph["canonical"] == "llm"
    assert morph["members"] == ["llms"]
    assert morph["confidence"] == 1.0

    embed = next(p for p in proposals if p["tier"] == "embedding")
    assert embed["canonical"] == "connectome"
    assert embed["members"] == ["connectomics"]
    assert embed["confidence"] > 0.9

    proposed_names = {n for p in proposals for n in [p["canonical"], *p["members"]]}
    assert "rag" not in proposed_names  # already a family member -> excluded
    assert "bm25" not in proposed_names  # no group -> no proposal


def test_detect_route_empty_corpus_returns_empty_list(env: Path) -> None:
    client = _client()  # no .rag stub needed — embed_fn is never called with <2 candidates
    r = client.post("/api/library/keyword-families/detect")
    assert r.status_code == 200
    assert r.json() == []


# --- PR-2.5: the under-guarded write paths (D1/D2/D3) ----------------------------------------- #
# Each of these is a verified repro from the post-commit review of `0c3b0d4`+`0af43db`. They
# describe the *correct* behaviour, so they fail against the code that shipped — that is the point:
# none of the 977 tests passing at the time caught any of them.


def test_rename_onto_an_existing_canonical_is_refused_not_duplicated(env: Path) -> None:
    """D1 — duplicate `Concept.label` rows poison the repo, not just this feature.

    `Concept.label` has no unique constraint and `rename_concept` defers the check to callers.
    Once two concepts share a label, `add_concept`'s get-or-create raises `MultipleResultsFound`
    for that label **forever** — so the create route 500s with no UI recovery, and
    `promote_keyword` / `scripts/seed_concepts.py` break repo-wide on a name they never touched.
    """
    from doc_assistant.library import create_keyword_family, list_keyword_families

    from doc_assistant.library import rename_keyword_family  # isort: skip

    create_keyword_family("llm", [])
    other = create_keyword_family("vector search", [])

    with pytest.raises(ValueError):
        rename_keyword_family(other.id, "llm")

    assert sorted(f.canonical for f in list_keyword_families()) == ["llm", "vector search"]
    # The repo-wide blast radius: get-or-create still resolves, so nothing downstream is poisoned.
    again = create_keyword_family("llm", [])
    assert again.canonical == "llm"


def test_rename_keeps_the_old_canonical_as_a_member(env: Path) -> None:
    """D2 — rename used to silently drop the family's own canonical keyword.

    `create_keyword_family` never seeds an alias for the label (unlike `promote_keyword`), so the
    canonical is only an *implicit* member. Re-pointing the label therefore dropped the original
    keyword out of the family, where it reappears as the standalone chip the feature exists to
    remove — and the doc_count silently falls.
    """
    from doc_assistant.library import create_keyword_family, rename_keyword_family

    _seed_doc_with_keywords("a.pdf", "llm")
    _seed_doc_with_keywords("b.pdf", "llms")
    _seed_doc_with_keywords("c.pdf", "llm", "llms")

    family = create_keyword_family("llm", ["llms"])
    assert family.doc_count == 3

    renamed = rename_keyword_family(family.id, "large language models")

    assert renamed is not None
    assert renamed.canonical == "large language models"
    assert renamed.aliases == ["llm", "llms"], "the old canonical must stay a member"
    assert renamed.doc_count == 3, "renaming a family must not change which documents it covers"


def test_a_keyword_cannot_end_up_in_two_families_via_a_free_text_canonical(env: Path) -> None:
    """D3 — `add_family_member` moves a keyword off its old family; the *canonical* did not.

    "New family" takes the canonical as unchecked free text, so naming it after a keyword that
    already belongs elsewhere left two families claiming it. `familyCanonicalMap` then resolves
    order-dependently, and the overlay, its tooltip and the Manage view show three different
    numbers for the same keyword.
    """
    from doc_assistant.library import create_keyword_family, list_keyword_families

    _seed_doc_with_keywords("a.pdf", "llm", "llms")
    create_keyword_family("large language models", ["llm", "llms"])
    create_keyword_family("llm", [])

    # A family's members are its aliases *plus* its own canonical label (the implicit member),
    # which is exactly the half the guard missed.
    claimants = [
        f
        for f in list_keyword_families()
        if "llm" in {f.canonical.casefold(), *(a.casefold() for a in f.aliases)}
    ]
    assert len(claimants) <= 1, "a keyword belongs to at most one family (ADR-015)"


def test_rename_collision_is_a_409_not_a_400(env: Path) -> None:
    """The invariant lives at the library boundary; the API shell only maps it (spec PR-2.5)."""
    from doc_assistant.library import create_keyword_family

    create_keyword_family("llm", [])
    other = create_keyword_family("vector search", [])

    response = _client().patch(
        f"/api/library/keyword-families/{other.id}", json={"canonical": "llm"}
    )

    assert response.status_code == 409
    assert "llm" in response.json()["detail"]


def test_rename_to_its_own_label_is_still_allowed(env: Path) -> None:
    """The collision guard must not make a no-op rename (or a case change) fail."""
    from doc_assistant.library import create_keyword_family, rename_keyword_family

    family = create_keyword_family("llm", [])

    assert rename_keyword_family(family.id, "llm") is not None
    renamed = rename_keyword_family(family.id, "LLM")
    assert renamed is not None and renamed.canonical == "LLM"


# --- graph vocabulary (ADR-018's in-app curation, ROADMAP 23) --------------------------------- #
#
# ADR-018 added `graph_include` and left the toggle as an explicit follow-up: "Curation has no UI
# yet... Its natural home is the Manage-keywords view." These cover the seam that follow-up added.
# The polarity is the load-bearing part: opt-in is what makes re-flooding the graph structurally
# impossible, so a family that nobody has touched must never read as included.


def test_a_new_family_is_not_graph_vocabulary(env: Path) -> None:
    """Opt-in, and it survives the round trip through `KeywordFamily`, not just the column."""
    from doc_assistant.library import create_keyword_family

    family = create_keyword_family("llm family", ["llm"])
    assert family.graph_include is False


def test_set_family_graph_include_round_trips(env: Path) -> None:
    from doc_assistant.library import (
        create_keyword_family,
        list_keyword_families,
        set_family_graph_include,
    )

    family_id = create_keyword_family("llm family", ["llm"]).id

    assert set_family_graph_include(family_id, True).graph_include is True  # type: ignore[union-attr]
    with session_scope() as session:
        assert session.get(Concept, family_id).graph_include is True  # type: ignore[union-attr]
    assert [f.graph_include for f in list_keyword_families()] == [True]

    # Idempotent, and reversible — no row is deleted at any point (ADR-018's last consequence).
    assert set_family_graph_include(family_id, True).graph_include is True  # type: ignore[union-attr]
    assert set_family_graph_include(family_id, False).graph_include is False  # type: ignore[union-attr]
    assert len(list_keyword_families()) == 1


def test_set_family_graph_include_unknown_family_returns_none(env: Path) -> None:
    from doc_assistant.library import set_family_graph_include

    assert set_family_graph_include("nope", True) is None


def test_toggling_graph_include_moves_the_graph_vocabulary(env: Path) -> None:
    """The point of the whole feature: `load_concepts` is what the skeleton builds over.

    Without this the toggle could write a column nobody reads. `load_concepts` is the single
    consumer ADR-018 scoped, so it is the honest end of the assertion.
    """
    from doc_assistant.knowledge.concept_skeleton import load_concepts
    from doc_assistant.library import create_keyword_family, set_family_graph_include

    family_id = create_keyword_family("llm family", ["llm"]).id
    concepts, _aliases = load_concepts()
    assert concepts == [], "a family must not enter the graph vocabulary by existing"

    set_family_graph_include(family_id, True)
    concepts, _aliases = load_concepts()
    assert [label for _cid, label in concepts] == ["llm family"]

    set_family_graph_include(family_id, False)
    concepts, _aliases = load_concepts()
    assert concepts == []


def test_route_patches_graph_include_alone(env: Path) -> None:
    client = _client()
    body = client.post(
        "/api/library/keyword-families", json={"canonical": "llm family", "members": []}
    ).json()
    assert body["graph_include"] is False

    r = client.patch(f"/api/library/keyword-families/{body['id']}", json={"graph_include": True})
    assert r.status_code == 200
    assert r.json()["graph_include"] is True and r.json()["canonical"] == "llm family"

    # The rename path predates the flag and must still work sending only `canonical` — and must
    # not reset the flag it never mentioned.
    r = client.patch(f"/api/library/keyword-families/{body['id']}", json={"canonical": "renamed"})
    assert r.status_code == 200
    assert r.json()["canonical"] == "renamed" and r.json()["graph_include"] is True


def test_route_patches_both_fields_together(env: Path) -> None:
    client = _client()
    family_id = client.post(
        "/api/library/keyword-families", json={"canonical": "llm family", "members": []}
    ).json()["id"]

    r = client.patch(
        f"/api/library/keyword-families/{family_id}",
        json={"canonical": "renamed", "graph_include": True},
    )
    assert r.status_code == 200
    assert r.json()["canonical"] == "renamed" and r.json()["graph_include"] is True


def test_route_refuses_an_empty_patch(env: Path) -> None:
    """A body with neither field is a caller bug; 200 would hide it behind a correct payload."""
    client = _client()
    family_id = client.post(
        "/api/library/keyword-families", json={"canonical": "llm family", "members": []}
    ).json()["id"]
    assert client.patch(f"/api/library/keyword-families/{family_id}", json={}).status_code == 400


def test_route_409_leaves_the_flag_alone(env: Path) -> None:
    """Rename-first ordering: a refused rename must not half-apply the flag beside it."""
    client = _client()
    client.post("/api/library/keyword-families", json={"canonical": "taken", "members": []})
    family_id = client.post(
        "/api/library/keyword-families", json={"canonical": "llm family", "members": []}
    ).json()["id"]

    r = client.patch(
        f"/api/library/keyword-families/{family_id}",
        json={"canonical": "taken", "graph_include": True},
    )
    assert r.status_code == 409
    still = [f for f in client.get("/api/library/keyword-families").json() if f["id"] == family_id]
    assert still[0]["canonical"] == "llm family" and still[0]["graph_include"] is False


def test_a_taxonomy_field_is_not_a_family_and_cannot_join_the_graph(env: Path) -> None:
    """A `kind="domain"` node must be invisible to every family path, writes included.

    `list_keyword_families` has always excluded ANZSRC field nodes (ADR-028 D4) while a lookup
    by id returned one as a family, so an id-addressed family write could land on a taxonomy
    node. The graph toggle is what made that matter: `load_concepts` filters on `graph_include`
    **alone**, so a flagged field node would enter the graph vocabulary — and presence-assuming
    code must read
    only `kind="concept"` (`db/models.py`). The write guard is asserted at the column, because a
    404 with the flag set would be the failure this is here to prevent.
    """
    from doc_assistant.library import get_keyword_family, set_family_graph_include

    with session_scope() as session:
        field = Concept(label="Biological sciences", source="anzsrc", kind="domain")
        session.add(field)
        session.flush()
        field_id = str(field.id)

    assert get_keyword_family(field_id) is None
    assert set_family_graph_include(field_id, True) is None
    with session_scope() as session:
        assert not session.get(Concept, field_id).graph_include  # type: ignore[union-attr]

    client = _client()
    r = client.patch(f"/api/library/keyword-families/{field_id}", json={"graph_include": True})
    assert r.status_code == 404
    with session_scope() as session:
        assert not session.get(Concept, field_id).graph_include  # type: ignore[union-attr]
