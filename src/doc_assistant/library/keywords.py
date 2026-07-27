"""Keyword families (feature-tag-families.md, PR-1/PR-2).

A family is a curated Concept whose ConceptAlias rows carry the member keyword names (ADR-015);
a keyword belongs to at most one family, so assigning it moves it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from doc_assistant.db.session import session_scope

if TYPE_CHECKING:
    from doc_assistant.knowledge.keyword_families import FamilyProposal


# ============================================================
# Keyword families (feature-tag-families.md — PR-1)
# ============================================================
# A family = a curated Concept whose ConceptAlias rows carry member Keyword names
# (ADR-015). Reuses the existing concept vocabulary — no new schema. A keyword
# belongs to at most one family; assigning it to a second family moves it.


@dataclass
class KeywordFamily:
    """A canonical tag + its member keyword names, with a union doc_count.

    ``aliases`` excludes the canonical label itself (mirrors ``GlossaryEntry``).
    ``doc_count`` is the number of documents carrying *any* member keyword (canonical or
    alias), matched case-insensitively against ``Keyword.name``.
    """

    id: str
    canonical: str
    aliases: list[str] = field(default_factory=list)
    doc_count: int = 0


def _family_doc_count(session: Any, names: list[str]) -> int:
    """Union of documents carrying any of ``names`` as a Keyword, case-insensitive."""
    from doc_assistant.db.models import Keyword, document_keywords

    if not names:
        return 0
    lowered = {n.casefold() for n in names}
    stmt = (
        select(func.count(func.distinct(document_keywords.c.document_id)))
        .select_from(document_keywords)
        .join(Keyword, Keyword.id == document_keywords.c.keyword_id)
        .where(func.lower(Keyword.name).in_(lowered))
    )
    return int(session.execute(stmt).scalar() or 0)


def _build_family(session: Any, concept: Any) -> KeywordFamily:
    aliases = sorted(a.alias for a in concept.aliases if a.alias != concept.label)
    doc_count = _family_doc_count(session, [concept.label, *aliases])
    return KeywordFamily(
        id=str(concept.id), canonical=concept.label, aliases=aliases, doc_count=doc_count
    )


def list_keyword_families() -> list[KeywordFamily]:
    """All curated concepts as keyword families, each with its union doc_count.

    Excludes ``kind="domain"`` taxonomy field nodes (ADR-028 D4) — an abstract ANZSRC field is not
    a keyword family, and the seeded ~236 of them would otherwise flood the Library filter."""
    from doc_assistant.db.models import Concept

    with session_scope() as session:
        concepts = list(
            session.execute(select(Concept).where(Concept.kind == "concept")).scalars()
        )
        families = [_build_family(session, c) for c in concepts]
    families.sort(key=lambda f: f.canonical.casefold())
    return families


def get_keyword_family(concept_id: str) -> KeywordFamily | None:
    """One keyword family by id, or None if unknown."""
    from doc_assistant.db.models import Concept

    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None:
            return None
        return _build_family(session, concept)


def create_keyword_family(canonical: str, members: list[str] | None = None) -> KeywordFamily:
    """Create a keyword family (a curated Concept) with initial member keywords.

    Idempotent by canonical label (matches ``add_concept``'s get-or-create). Any member
    keyword already belonging to another family is moved (a keyword belongs to at most one
    family — ADR-015).

    Families are **not** graph vocabulary (``graph_include=False``, ADR-018): grouping keywords
    is library organisation, not a claim that the concept belongs on the map. This is what stops
    the families feature from re-flooding the graph as it grows.
    """
    from doc_assistant.knowledge.concept_skeleton import add_concept

    canonical = canonical.strip()
    if not canonical:
        raise ValueError("canonical must not be blank")
    concept_id = add_concept(label=canonical, graph_include=False)
    # The canonical is a member too (an implicit one — `_build_family`). "New family" takes it as
    # unchecked free text, so without this a keyword already claimed elsewhere ended up in two
    # families and `familyCanonicalMap` resolved it order-dependently (PR-2.5 D3). Routing it
    # through `add_family_member` reuses the move-on-reassign guard rather than restating it: the
    # call detaches the name from any other family and, being the label, adds no self-alias.
    add_family_member(concept_id, canonical)
    for member in members or []:
        add_family_member(concept_id, member)
    family = get_keyword_family(concept_id)
    if family is None:  # pragma: no cover - add_concept above guarantees the row exists
        raise RuntimeError(f"keyword family {concept_id!r} vanished immediately after creation")
    return family


class KeywordFamilyExists(ValueError):
    """Another family already uses this canonical label (the API shell maps it to 409)."""


def rename_keyword_family(concept_id: str, new_canonical: str) -> KeywordFamily | None:
    """Rename a family's canonical label. Returns None if the family is unknown.

    Two guards, both PR-2.5 defects that shipped in PR-1:

    * **D1 — the label must stay unique.** ``Concept.label`` has no unique constraint and
      ``rename_concept`` defers the check to callers, so a rename onto an existing canonical
      created duplicate rows — after which ``add_concept``'s get-or-create raises
      ``MultipleResultsFound`` for that label **forever**, breaking the create route *and*
      ``promote_keyword`` repo-wide, with no way back through the UI. Compared case-insensitively
      because the client's ``familyCanonicalMap`` lowercases its keys, so two families differing
      only by case would collide there anyway. Raises :class:`KeywordFamilyExists` (a
      ``ValueError``, so existing 400 handlers still catch it).
    * **D2 — the old canonical stays a member.** The label is only an *implicit* member
      (``create_keyword_family`` seeds no alias for it), so re-pointing it dropped the original
      keyword out of the family, where it reappeared as the standalone chip the feature exists to
      remove — and ``doc_count`` silently fell. Carrying it into the alias set keeps the family
      covering the same documents, which is the whole invariant of a rename.
    """
    from doc_assistant.db.models import Concept, ConceptAlias
    from doc_assistant.knowledge.concept_skeleton import rename_concept

    new_canonical = new_canonical.strip()
    if not new_canonical:
        raise ValueError("new_canonical must not be blank")

    with session_scope() as session:
        clash = (
            session.execute(
                select(Concept).where(
                    func.lower(Concept.label) == new_canonical.casefold(),
                    Concept.id != concept_id,
                    Concept.kind == "concept",  # a family rename can't clash with a taxonomy field
                )
            )
            .scalars()
            .first()
        )
        if clash is not None:
            raise KeywordFamilyExists(f"a keyword family named {new_canonical!r} already exists")

        concept = session.get(Concept, concept_id)
        if concept is None:
            return None
        old_label = concept.label
        keeps_old = old_label.casefold() != new_canonical.casefold() and not any(
            a.alias.casefold() == old_label.casefold() for a in concept.aliases
        )
        if keeps_old:
            concept.aliases.append(ConceptAlias(alias=old_label))
        session.flush()

    if not rename_concept(concept_id, new_canonical):
        return None
    return get_keyword_family(concept_id)


def add_family_member(concept_id: str, keyword_name: str) -> KeywordFamily | None:
    """Assign a keyword to a family. Returns None if the family is unknown.

    A keyword belongs to at most one family: if it's already an alias of another family,
    it's removed from there first (moved, not duplicated). Idempotent — assigning an
    already-member keyword is a no-op. (Does not check whether ``keyword_name`` collides
    with *another* family's canonical label — an edge case left to the Manage view.)
    """
    from doc_assistant.db.models import Concept, ConceptAlias

    keyword_name = keyword_name.strip()
    if not keyword_name:
        raise ValueError("keyword_name must not be blank")
    lowered = keyword_name.casefold()
    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None:
            return None
        others = (
            session.execute(
                select(ConceptAlias).where(
                    func.lower(ConceptAlias.alias) == lowered,
                    ConceptAlias.concept_id != concept_id,
                )
            )
            .scalars()
            .all()
        )
        for other in others:
            session.delete(other)
        already_member = concept.label.casefold() == lowered or any(
            a.alias.casefold() == lowered for a in concept.aliases
        )
        if not already_member:
            concept.aliases.append(ConceptAlias(alias=keyword_name))
        session.flush()
        return _build_family(session, concept)


def remove_family_member(concept_id: str, keyword_name: str) -> KeywordFamily | None:
    """Remove a keyword from a family's alias set. Returns None if the family is unknown.

    A no-op if ``keyword_name`` isn't a member alias (idempotent) — the canonical label
    itself can't be "removed" this way; rename or delete the family instead.
    """
    from doc_assistant.db.models import Concept

    lowered = keyword_name.strip().casefold()
    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None:
            return None
        row = next((a for a in concept.aliases if a.alias.casefold() == lowered), None)
        if row is not None:
            concept.aliases.remove(row)
        session.flush()
        return _build_family(session, concept)


def delete_keyword_family(concept_id: str) -> bool:
    """Delete a family. Returns True if it existed."""
    from doc_assistant.knowledge.concept_skeleton import delete_concept

    return delete_concept(concept_id)


def _all_keyword_names() -> list[str]:
    """Every distinct ``Keyword.name`` in the corpus, sorted."""
    from doc_assistant.db.models import Keyword

    with session_scope() as session:
        return sorted({k.name for k in session.execute(select(Keyword)).scalars()})


def detect_family_candidates(
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    *,
    embedding_threshold: float | None = None,
) -> list[FamilyProposal]:
    """Run the zero-LLM detection tiers (PR-2) over every un-familied keyword.

    A keyword already a family's canonical or alias is excluded before detection runs — nothing
    here writes to the DB or promotes a proposal; reviewing + accepting one is done through the
    existing family CRUD above. ``embed_fn`` (see ``keyword_families.detect_family_proposals``)
    is optional — omit it for a Tier-1-only (morphological) pass. ``embedding_threshold`` defaults
    to ``keyword_families.DEFAULT_EMBEDDING_THRESHOLD`` when omitted.
    """
    from doc_assistant.knowledge.keyword_families import (
        DEFAULT_EMBEDDING_THRESHOLD,
        detect_family_proposals,
    )

    names = _all_keyword_names()
    familied: set[str] = set()
    for f in list_keyword_families():
        familied.add(f.canonical.casefold())
        familied.update(a.casefold() for a in f.aliases)
    candidates = [n for n in names if n.casefold() not in familied]
    return detect_family_proposals(
        candidates,
        embed_fn=embed_fn,
        embedding_threshold=(
            embedding_threshold if embedding_threshold is not None else DEFAULT_EMBEDDING_THRESHOLD
        ),
    )
