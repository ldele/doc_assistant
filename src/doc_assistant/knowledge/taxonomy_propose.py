"""Taxonomy auto-propose (ADR-028 Decision 8, increment 3) — quarantined placement proposals.

The curated taxonomy (``knowledge/taxonomy.py``) can be filled by hand through the increment-2
surfaces, but on a real corpus nothing is placed, so ADR-028 D6's coverage math reads 0 everywhere.
This module is the first-pass filler: for one unplaced concept or unclassified document it asks a
local model which **existing** taxonomy field it belongs under, and returns that as a *proposal*.

It is ADR-019 E1 ("the LLM proposes only where the link IS NULL, never overwriting") applied to
``concept_hierarchy`` / ``document_field`` — no new pattern, and quality is explicitly unmeasured
(RG-015), which is why nothing here is written as fact.

**Two-stage narrowing.** ANZSRC seeds a 2-level trunk (23 divisions, 213 groups). Asking a small
local model to pick from 236 labels at once is the weakest form of the question; asking it for a
division (~23 options) and then for a group *inside that division* (~10 options) is two easy
questions, and the intermediate answer is itself a valid placement target — both levels are
``kind="domain"`` nodes. So a stage-2 abstention degrades to an honest coarse placement rather than
to nothing.

**Abstention is first-class.** Either stage may answer "none". A wrong placement is worse than a
missing one: it inflates the very coverage counts this layer exists to make trustworthy.

Confinement (by construction, mirrors ``gap_suggest`` / Node B):

* Takes an already-built ``LLMClient`` — this module makes **no provider decision** (the caller,
  ``scripts/propose_taxonomy.py``, resolves ``TAXONOMY_PROPOSE_LLM_PROVIDER``/``_MODEL`` and routes
  ``--apply`` through ``llm.assert_provider_intent`` first).
* :func:`propose_placements` — the pass itself — is **DB-free and writes nothing**: it reads a
  ``load_taxonomy`` DiGraph and returns values. :func:`run_propose` owns the session either side of
  it (load the unplaced items, write the accepted-later proposals), the way ``gaps.build_gaps``
  wraps the pure detectors.
* Every write lands as ``origin="proposed"`` through the ``taxonomy.py`` seam — never as curated
  fact, and never over a curated row.
* Never invents a field: every proposal points at a node already in the graph.
* A per-item transport/parse failure is logged and skipped; it does not sink the run.
* No items, or no field nodes, ⇒ zero LLM calls (checked before the loop, not caught after).
* No ``--apply`` ⇒ zero LLM calls, because ``assert_provider_intent`` deliberately no-ops on a dry
  run: if a dry run made calls, a paid ``--provider`` would bill without ever tripping the guard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal

import networkx as nx
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from doc_assistant.db.models import Concept, ConceptAlias, ConceptPresenceRow, Document
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.taxonomy import (
    add_hierarchy_edge,
    attach_document_field,
    load_taxonomy,
    unclassified_documents,
    unplaced_concepts,
)
from doc_assistant.llm import LLMClient, Message

log = structlog.get_logger(__name__)

#: One choice is a tiny JSON object — a small budget keeps a local model fast and on-shape.
DEFAULT_MAX_TOKENS = 256
#: How many source-document titles a concept's context carries. A structural readability bound on
#: one prompt (not a corpus-tuned threshold): the titles disambiguate a bare label like "cre", and
#: the 4th adds prompt length without adding a domain signal.
CONTEXT_TITLES = 3

ItemKind = Literal["concept", "document"]

_SYSTEM = """You classify one research item into a research-field classification.

You are given the item and a NUMBERED list of candidate fields. Pick the single
best-fitting candidate by its number.

If none of the candidates fits the item, answer "none" — a wrong field is worse than no field.

Respond with ONLY a JSON object of this exact shape, nothing else:
{"choice": 3, "confidence": 0.7}

Use {"choice": "none", "confidence": 0.0} when nothing fits. "confidence" is a number in [0, 1]."""


@dataclass(frozen=True)
class FieldCandidate:
    """One taxonomy field node offered to the model (``kind="domain"``)."""

    id: str
    label: str


@dataclass(frozen=True)
class ProposalItem:
    """One thing to place: a concept or a document, plus whatever context the caller has.

    ``context`` is free text the caller assembles deterministically (document titles a concept
    appears in, a document's authors/year, …). It is passed verbatim to the model and recorded in
    the proposal's evidence, so what the placement was based on is always inspectable.
    """

    kind: ItemKind
    id: str
    label: str
    context: str = ""


@dataclass(frozen=True)
class Choice:
    """A parsed model answer: ``index`` is 1-based into the offered candidates.

    ``index is None`` = an explicit abstention ("none" — nothing fits). ``confidence is None`` =
    the model gave no usable number; it is left absent rather than fabricated.
    """

    index: int | None
    confidence: float | None = None


@dataclass(frozen=True)
class PlacementProposal:
    """One proposed ``in_field`` placement (concept→field) or document classification.

    ``field_id`` is the *placement target*: the group when stage 2 chose one, otherwise the
    division (a coarse but honest placement). ``division_id``/``_label`` always record the
    stage-1 answer, so a division-level placement is recognisable as one (``field_id ==
    division_id``). ``evidence`` records the exact LLM inputs and both stage answers.
    """

    item_kind: ItemKind
    item_id: str
    item_label: str
    field_id: str
    field_label: str
    division_id: str
    division_label: str
    confidence: float | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProposeResult:
    """The pass's outcome — proposals plus the counts that make the misses visible.

    ``n_abstained`` items got no placement at all (stage 1 said "none", failed, or answered
    unparseably); ``n_division_only`` were placed at division level because stage 2 abstained or
    the division has no groups. Both are reported, never silently dropped.
    """

    proposals: tuple[PlacementProposal, ...] = ()
    n_items: int = 0
    n_abstained: int = 0
    n_division_only: int = 0
    n_calls: int = 0
    #: Per-item notes for the misses (item label → why), for the runner's report.
    misses: tuple[tuple[str, str], ...] = dataclass_field(default_factory=tuple)


# ============================================================
# Candidate reads over the taxonomy graph (pure, no DB)
# ============================================================


def _is_domain(graph: nx.DiGraph, node: str) -> bool:
    return str(graph.nodes[node].get("kind", "")) == "domain"


def _label(graph: nx.DiGraph, node: str) -> str:
    return str(graph.nodes[node].get("label", ""))


def division_candidates(graph: nx.DiGraph) -> list[FieldCandidate]:
    """The top-level fields: domain nodes with no broader domain (an ``in_field`` successor).

    Structural, not ANZSRC-specific — a hand-added root field is a division here too. Sorted by
    label so the numbering the model sees is stable across runs (a reproducibility floor: the same
    corpus and model produce the same prompts).
    """
    out: list[FieldCandidate] = []
    for node in graph.nodes:
        if not _is_domain(graph, node):
            continue
        broader = [
            t
            for _, t, data in graph.out_edges(node, data=True)
            if data.get("type") == "in_field" and _is_domain(graph, t)
        ]
        if not broader:
            out.append(FieldCandidate(id=node, label=_label(graph, node)))
    return sorted(out, key=lambda c: c.label.casefold())


def group_candidates(graph: nx.DiGraph, division_id: str) -> list[FieldCandidate]:
    """The narrower domain nodes directly under ``division_id`` (its ``in_field`` predecessors)."""
    if division_id not in graph:
        return []
    out = [
        FieldCandidate(id=s, label=_label(graph, s))
        for s, _, data in graph.in_edges(division_id, data=True)
        if data.get("type") == "in_field" and _is_domain(graph, s)
    ]
    return sorted(out, key=lambda c: c.label.casefold())


# ============================================================
# Prompt + parse (pure)
# ============================================================


def build_choice_messages(
    item: ProposalItem, candidates: list[FieldCandidate], *, level: str
) -> list[Message]:
    """The one-shot prompt for one narrowing step: the item + a numbered candidate list."""
    numbered = "\n".join(f"{i}. {c.label}" for i, c in enumerate(candidates, start=1))
    lines = [f"Item ({item.kind}): {item.label}"]
    if item.context:
        lines.append(f"Context: {item.context}")
    lines.append(f"Candidate {level} (choose ONE number):")
    lines.append(numbered)
    lines.append("")
    lines.append("Return the JSON object.")
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def _parse_confidence(raw: object) -> float | None:
    """A usable confidence, or ``None`` — never a fabricated one.

    Out-of-range (a model answering ``5`` to a 0-1 question) reads as *no usable number*, not as a
    reason to throw the choice away: the placement is still the model's answer, it just carries no
    rating.
    """
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def parse_choice(text: str, n_candidates: int) -> Choice | None:
    """Parse one model answer into a :class:`Choice`, or ``None`` when unparseable.

    Tolerant by design (local models drift): strips a ``json`` code fence, accepts a bare integer,
    ``{"choice": 3}``, ``{"choice": "3"}``, and ``"none"``/``null`` for an abstention. An index
    outside ``1..n_candidates`` is a **parse failure**, not a clamp — silently snapping an
    out-of-range answer to a real field would manufacture a placement the model never chose. A
    missing ``choice`` key is likewise a failure, not an abstention: the model said nothing, which
    is a different event from it saying "nothing fits" (they log differently).
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw[:4].lower() == "json" else raw
        raw = raw.strip()

    data: object
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    confidence: float | None = None
    if isinstance(data, dict):
        confidence = _parse_confidence(data.get("confidence"))
        if "choice" not in data:
            return None  # no answer at all — not the same as an explicit "none"
        choice: object = data["choice"]
    else:
        choice = data

    if choice is None:
        return Choice(index=None, confidence=confidence)
    if isinstance(choice, str):
        stripped = choice.strip().lower()
        if stripped in {"none", "null", ""}:
            return Choice(index=None, confidence=confidence)
        choice = stripped
    # A bool, list or dict where a number belongs is unusable, not guessable. `bool` is checked
    # first because it is an int subclass: `{"choice": true}` would otherwise select candidate 1.
    if isinstance(choice, bool) or not isinstance(choice, str | int | float):
        return None
    try:
        index = int(choice)
    except (TypeError, ValueError):
        return None
    if not 1 <= index <= n_candidates:
        return None
    return Choice(index=index, confidence=confidence)


# ============================================================
# The pass
# ============================================================


def _ask(
    client: LLMClient,
    item: ProposalItem,
    candidates: list[FieldCandidate],
    *,
    level: str,
    temperature: float,
    max_tokens: int,
) -> Choice | None:
    """One narrowing step. ``None`` = no usable answer (transport failure or unparseable)."""
    messages = build_choice_messages(item, candidates, level=level)
    try:
        text = client.complete(messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:  # transport failure — one bad item must not sink the run
        log.warning("taxonomy_propose_transport_failed", item=item.id, level=level, error=str(exc))
        return None
    choice = parse_choice(text, len(candidates))
    if choice is None:
        log.warning("taxonomy_propose_unparseable", item=item.id, level=level)
    return choice


def _weakest(*values: float | None) -> float | None:
    """The chain's confidence: its weakest present link.

    A group placement is only as good as the division it was narrowed inside, so the pair's
    minimum is the honest number to carry (not the more flattering second answer).
    """
    present = [v for v in values if v is not None]
    return min(present) if present else None


def propose_placements(
    items: list[ProposalItem],
    graph: nx.DiGraph,
    client: LLMClient,
    *,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ProposeResult:
    """Propose one ``in_field`` placement per item, two-stage (division → group).

    ``graph`` is a ``taxonomy.load_taxonomy`` DiGraph, read-only and never mutated. Returns a
    :class:`ProposeResult`; writing the proposals (as ``origin="proposed"`` rows) is the caller's
    job. Zero items or a graph with no field nodes ⇒ zero LLM calls.
    """
    divisions = division_candidates(graph)
    if not items or not divisions:
        return ProposeResult(n_items=len(items))

    proposals: list[PlacementProposal] = []
    misses: list[tuple[str, str]] = []
    n_calls = 0
    n_abstained = 0
    n_division_only = 0

    for item in items:
        top = _ask(
            client,
            item,
            divisions,
            level="fields",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        n_calls += 1
        if top is None or top.index is None:
            n_abstained += 1
            misses.append((item.label, "no division" if top else "no answer at division level"))
            continue
        division = divisions[top.index - 1]

        groups = group_candidates(graph, division.id)
        target, confidence, group_note = division, top.confidence, "no groups seeded"
        if groups:
            narrow = _ask(
                client,
                item,
                groups,
                level=f"groups within {division.label}",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            n_calls += 1
            if narrow is not None and narrow.index is not None:
                target = groups[narrow.index - 1]
                confidence = _weakest(top.confidence, narrow.confidence)
                group_note = target.label
            else:
                group_note = "abstained" if narrow else "no answer"

        if target.id == division.id:
            n_division_only += 1

        proposals.append(
            PlacementProposal(
                item_kind=item.kind,
                item_id=item.id,
                item_label=item.label,
                field_id=target.id,
                field_label=target.label,
                division_id=division.id,
                division_label=division.label,
                confidence=confidence,
                evidence=(
                    f"item={item.label}",
                    f"context={item.context}",
                    f"division={division.label} (conf={top.confidence})",
                    f"group={group_note}",
                ),
            )
        )

    return ProposeResult(
        proposals=tuple(proposals),
        n_items=len(items),
        n_abstained=n_abstained,
        n_division_only=n_division_only,
        n_calls=n_calls,
        misses=tuple(misses),
    )


# ============================================================
# The runner side — session in, proposals out (the `build_gaps` shape)
# ============================================================


@dataclass(frozen=True)
class ProposeRunResult:
    """One runner pass: the scope it found, what the model proposed, what was written."""

    n_unplaced_concepts: int = 0
    n_unclassified_documents: int = 0
    #: Unplaced concepts outside the graph vocabulary — reported, never silently dropped.
    n_concepts_out_of_scope: int = 0
    #: Items dropped by ``--limit``, so a bounded run can never read as a complete one.
    n_truncated: int = 0
    result: ProposeResult = ProposeResult()
    n_hierarchy_written: int = 0
    n_document_fields_written: int = 0
    applied: bool = False


def _concept_context(session: Session, concept: Concept) -> str:
    """Deterministic context for one concept: its aliases, gloss, and source-document titles.

    The titles come from the ``concept_presence`` sidecar when a skeleton has been built; with no
    skeleton the context is simply thinner (the label + gloss), never an error — a concept whose
    presence has never been computed is still placeable, just with less to go on.
    """
    parts: list[str] = []
    aliases = (
        session.execute(select(ConceptAlias.alias).where(ConceptAlias.concept_id == concept.id))
        .scalars()
        .all()
    )
    if aliases:
        parts.append(f"also written: {', '.join(sorted(aliases))}")
    if concept.definition:
        parts.append(f"definition: {concept.definition}")
    titles = session.execute(
        select(Document.title, Document.filename)
        .join(ConceptPresenceRow, ConceptPresenceRow.document_id == Document.id)
        .where(ConceptPresenceRow.concept_id == concept.id)
        .order_by(ConceptPresenceRow.n_mentions.desc())
        .limit(CONTEXT_TITLES)
    ).all()
    if titles:
        rendered = "; ".join(f'"{title or filename}"' for title, filename in titles)
        parts.append(f"appears in: {rendered}")
    return " | ".join(parts)


def _document_context(document: Document) -> str:
    """Deterministic context for one document: its authors + year, when the store has them."""
    parts: list[str] = []
    if document.authors:
        parts.append(f"authors: {document.authors}")
    if document.year:
        parts.append(f"year: {document.year}")
    if document.filename and document.title:
        parts.append(f"file: {document.filename}")
    return " | ".join(parts)


def load_concept_items(session: Session, *, graph_only: bool = True) -> list[ProposalItem]:
    """Unplaced concepts as :class:`ProposalItem`s, context assembled (ordered by label)."""
    concepts = sorted(
        unplaced_concepts(session, graph_only=graph_only), key=lambda c: c.label.casefold()
    )
    return [
        ProposalItem(kind="concept", id=c.id, label=c.label, context=_concept_context(session, c))
        for c in concepts
    ]


def load_document_items(session: Session) -> list[ProposalItem]:
    """Unclassified documents as :class:`ProposalItem`s (title, falling back to the filename)."""
    documents = unclassified_documents(session)
    items = [
        ProposalItem(
            kind="document",
            id=d.id,
            label=d.title or d.filename,
            context=_document_context(d),
        )
        for d in documents
    ]
    return sorted(items, key=lambda i: i.label.casefold())


def write_proposals(session: Session, proposals: tuple[PlacementProposal, ...]) -> tuple[int, int]:
    """Persist proposals as ``origin="proposed"`` links. Returns ``(n_hierarchy, n_document)``.

    Concept placements become ``concept --in_field--> field`` edges; document placements become
    ``document_field`` rows. Both go through the ``taxonomy.py`` seam, so the cycle check and the
    domain-target check still apply and a curated row is never overwritten. A per-proposal failure
    (a cycle, a target that turned out not to be a domain) is logged and skipped — one bad
    proposal does not lose the batch.
    """
    n_hierarchy = 0
    n_documents = 0
    for proposal in proposals:
        try:
            if proposal.item_kind == "concept":
                add_hierarchy_edge(
                    session, proposal.item_id, proposal.field_id, "in_field", origin="proposed"
                )
                n_hierarchy += 1
            else:
                attach_document_field(
                    session, proposal.item_id, proposal.field_id, origin="proposed"
                )
                n_documents += 1
        except ValueError as exc:  # cycle, non-domain target, bad origin — all ValueError-rooted
            log.warning(
                "taxonomy_propose_write_rejected",
                item=proposal.item_id,
                field=proposal.field_id,
                error=str(exc),
            )
    return n_hierarchy, n_documents


def run_propose(
    *,
    apply: bool = False,
    client: LLMClient | None = None,
    include_concepts: bool = True,
    include_documents: bool = True,
    all_concepts: bool = False,
    limit: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ProposeRunResult:
    """Load the unplaced items, propose placements, and (with ``apply``) write them as proposals.

    Without ``apply`` — or without a ``client`` — this is a **scope report**: it counts what would
    be placed and makes **zero** LLM calls (see the module docstring on why a dry run must not
    call). With both, it runs the two-stage pass and writes ``origin="proposed"`` links.
    """
    with session_scope() as session:
        graph = load_taxonomy(session)
        items: list[ProposalItem] = []
        n_concepts = 0
        n_out_of_scope = 0
        n_documents = 0
        if include_concepts:
            concept_items = load_concept_items(session, graph_only=not all_concepts)
            n_concepts = len(concept_items)
            if not all_concepts:
                n_out_of_scope = len(unplaced_concepts(session, graph_only=False)) - n_concepts
            items.extend(concept_items)
        if include_documents:
            document_items = load_document_items(session)
            n_documents = len(document_items)
            items.extend(document_items)

    n_truncated = 0
    if limit is not None and len(items) > limit:
        n_truncated = len(items) - limit
        items = items[:limit]

    scope = ProposeRunResult(
        n_unplaced_concepts=n_concepts,
        n_unclassified_documents=n_documents,
        n_concepts_out_of_scope=n_out_of_scope,
        n_truncated=n_truncated,
        result=ProposeResult(n_items=len(items)),
    )
    if not apply or client is None:
        return scope

    result = propose_placements(
        items, graph, client, temperature=temperature, max_tokens=max_tokens
    )
    with session_scope() as session:
        n_hierarchy, n_document_fields = write_proposals(session, result.proposals)

    return ProposeRunResult(
        n_unplaced_concepts=n_concepts,
        n_unclassified_documents=n_documents,
        n_concepts_out_of_scope=n_out_of_scope,
        n_truncated=n_truncated,
        result=result,
        n_hierarchy_written=n_hierarchy,
        n_document_fields_written=n_document_fields,
        applied=True,
    )
