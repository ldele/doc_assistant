"""Gap detection — Phase 7's headline capability (ADR-004 / feature-gap-detection).

One typed ``Gap`` object, split on the project's existing deterministic/stochastic
line (ADR-004 Decision 1). This module ships **Tier 1** (deterministic, over the
concept skeleton) and the **Tier-2a deterministic floor** (a query over already-
persisted answer-claim data), and orchestrates (``build_gaps(suggest=True, ...)``)
the **Tier-2a stochastic ceiling** — a quarantined LLM suggestion pass that lives
in its own module, ``gap_suggest.py`` (it never writes the skeleton as fact —
ADR-004 Decision 4; this module makes no provider decision, it only plumbs an
already-built ``LLMClient`` through).

Detectors are pure: no DB, no Chroma, no LLM. ``build_gaps`` (bottom of this module)
is the impure orchestration — load the skeleton + claims, run the detectors, write
the sidecar — mirroring ``epistemics.build_epistemics``'s pure-core/impure-boundary
split (Enrichment-Layer Pattern); ``scripts/build_gaps.py`` is its thin CLI wrapper.
Deterministic ``gaps`` rows are dropped + rebuilt on every run; a stochastic row's
``status`` (the compounding arrow) is untouched by that rebuild.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from doc_assistant.concept_skeleton import PRESENCE_BOUNDARY, ConceptSkeleton, match_presence
from doc_assistant.llm import LLMClient
from doc_assistant.synthesis import MARKER_UNSUPPORTED

GapTier = Literal["t1", "t2a", "t2b"]
Determinism = Literal["deterministic", "stochastic"]
GapKind = Literal[
    # Tier 1 (deterministic, over the curated skeleton)
    "isolated",
    "single_source",
    "thin_bridge",
    "under_connected",
    # Tier 2a floor (deterministic, over persisted answer/citation data)
    "unsourced_claim",
    "citation_missing",
    # Tier 2a ceiling + Tier 2b (stochastic, suggestions — gap_suggest.py, not here)
    "suggested_link",
    "suggested_concept",
    "thin_area",
]
GapStatus = Literal["surfaced", "promoted", "dismissed"]


@dataclass(frozen=True)
class GapEvidence:
    """What backs a ``Gap``. Deterministic: graph-fact ids (edge endpoints, document
    ids, or the contributing ``answer_claims`` ids an ``unsourced_claim`` aggregates).
    Stochastic (``gap_suggest.py``, not built here): the exact LLM inputs, for
    observability (ADR-004's "expose LLM inputs, rate output" mandate)."""

    fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Gap:
    """One detected corpus gap (ADR-004). ``determinism`` is first-class — a
    consumer reads it, never re-derives it. ``rating`` is ``None`` for every
    deterministic gap (a raw graph fact carries no confidence score); it is
    populated only by the stochastic ceiling (not built here)."""

    concept_id: str
    tier: GapTier
    determinism: Determinism
    kind: GapKind
    evidence: GapEvidence = field(default_factory=GapEvidence)
    rating: float | None = None
    status: GapStatus = "surfaced"


# ============================================================
# Tier 1 — deterministic detectors over the concept skeleton (pure)
# ============================================================


def detect_isolated(skeleton: ConceptSkeleton) -> list[Gap]:
    """Degree-0 curated concepts — mentioned, never related to another concept."""
    return [
        Gap(concept_id=n.id, tier="t1", determinism="deterministic", kind="isolated")
        for n in skeleton.nodes
        if n.degree == 0
    ]


def detect_single_source(skeleton: ConceptSkeleton) -> list[Gap]:
    """Concepts asserted by exactly one document (7d Decision 4, carried over): the
    corpus's only source on this topic — **flagged for attention, never a defect**.
    Distinguished from a contested claim by the absence of any disputing source;
    the skeleton alone can't tell "sole source" from "wrong", only surface it."""
    return [
        Gap(
            concept_id=n.id,
            tier="t1",
            determinism="deterministic",
            kind="single_source",
            evidence=GapEvidence(fact_ids=n.doc_ids),
        )
        for n in skeleton.nodes
        if len(n.doc_ids) == 1
    ]


def detect_thin_bridges(skeleton: ConceptSkeleton) -> list[Gap]:
    """Concepts whose only link to part of the graph is a single cut edge (the
    retired ``concept_graph`` 7c mechanism, re-homed here: ``networkx.bridges``
    over each connected component). One ``Gap`` per bridge endpoint, so a
    per-concept view surfaces it directly; ``evidence`` names both endpoints."""
    import networkx as nx

    graph = nx.Graph()
    for n in skeleton.nodes:
        graph.add_node(n.id)
    for e in skeleton.edges:
        graph.add_edge(e.source_concept_id, e.target_concept_id, weight=e.weight)

    flagged: set[str] = set()
    gaps: list[Gap] = []
    for comp in nx.connected_components(graph):
        sub = graph.subgraph(comp)
        if sub.number_of_edges() == 0:
            continue
        for u, v in sorted(tuple(sorted(pair)) for pair in nx.bridges(sub)):
            for node_id in (u, v):
                if node_id in flagged:
                    continue
                flagged.add(node_id)
                gaps.append(
                    Gap(
                        concept_id=node_id,
                        tier="t1",
                        determinism="deterministic",
                        kind="thin_bridge",
                        evidence=GapEvidence(fact_ids=(u, v)),
                    )
                )
    gaps.sort(key=lambda g: g.concept_id)
    return gaps


def detect_under_connected(skeleton: ConceptSkeleton, *, min_degree: int) -> list[Gap]:
    """Curated concepts with ``0 < degree < min_degree`` — the routing signal into
    the (separate, not built here) Tier-2a stochastic ceiling. Degree-0 concepts are
    excluded — those are ``isolated``, a distinct kind, not double-reported here.
    ``min_degree`` is corpus-derived, never a guessed absolute (see
    ``tests/eval/baselines/gap_min_degree_2026-07.md``)."""
    return [
        Gap(concept_id=n.id, tier="t1", determinism="deterministic", kind="under_connected")
        for n in skeleton.nodes
        if 0 < n.degree < min_degree
    ]


# ============================================================
# Tier 2a — deterministic floor over persisted answer-claim data (pure)
# ============================================================


@dataclass(frozen=True)
class ClaimForGap:
    """A minimal, DB-agnostic view of one persisted ``AnswerClaim`` row — just
    enough for the deterministic floor to run without touching the DB (pure core;
    the impure loader in ``scripts/build_gaps.py`` maps the ORM rows to this)."""

    id: str
    text: str
    marker: str


def detect_unsourced_claims(
    claims: list[ClaimForGap],
    concepts: list[tuple[str, str]],
    aliases: dict[str, list[str]],
    *,
    mode: str = PRESENCE_BOUNDARY,
) -> list[Gap]:
    """Aggregate ``unsupported``-marked claims onto the curated concept(s) their text
    matches (presence match, Decision C — reuses ``concept_skeleton.match_presence``
    so a claim and a chunk are attributed by the identical rule). A query over data
    that already exists (``synthesis.claim_marker`` → ``AnswerClaim.marker``); no new
    model (ADR-004 Decision 3). Cited (non-``unsupported``) claims produce nothing;
    an unsupported claim matching no curated concept also produces nothing (it isn't
    attributable to a vocabulary gap without a concept to hang it on)."""
    unsupported = [c for c in claims if c.marker == MARKER_UNSUPPORTED]
    if not unsupported:
        return []
    chunk_texts = [(c.id, c.id, c.text) for c in unsupported]
    presences = match_presence(concepts, aliases, chunk_texts, mode=mode)
    by_concept: dict[str, set[str]] = defaultdict(set)
    for p in presences:
        by_concept[p.concept_id].update(p.chunk_keys)  # chunk_key == claim id here
    return [
        Gap(
            concept_id=concept_id,
            tier="t2a",
            determinism="deterministic",
            kind="unsourced_claim",
            evidence=GapEvidence(fact_ids=tuple(sorted(claim_ids))),
        )
        for concept_id, claim_ids in sorted(by_concept.items())
    ]


# ============================================================
# Impure boundary — SQLite + sidecar reads, orchestration
# ============================================================


@dataclass(frozen=True)
class GapsResult:
    """What a ``build_gaps`` run produced (for the CLI report)."""

    gaps: list[Gap]
    graph_version: str
    n_t1: int
    n_t2a: int
    applied: bool
    n_suggested: int = (
        0  # Tier-2a stochastic ceiling rows written this run (0 unless suggest+apply)
    )


def load_unsupported_claims() -> list[ClaimForGap]:
    """Read every persisted ``unsupported``-marked ``AnswerClaim`` row.

    Read-only, free — a query over data ``synthesis.claim_marker`` already wrote;
    no new model (ADR-004 Decision 3)."""
    from sqlalchemy import select

    from doc_assistant.db.models import AnswerClaim
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        stmt = select(AnswerClaim).where(AnswerClaim.marker == MARKER_UNSUPPORTED)
        return [
            ClaimForGap(id=str(row.id), text=row.claim_text, marker=row.marker)
            for row in session.execute(stmt).scalars()
        ]


def _write_gap_rows(gaps: list[Gap], version: str) -> None:
    """Replace the *deterministic* ``gaps`` rows (idempotent). Stochastic rows (the
    deferred Tier-2a ceiling, ``gap_suggest.py``) are untouched — their ``status``
    is the compounding arrow and must survive a deterministic rebuild."""
    import json

    from sqlalchemy import delete

    from doc_assistant.db.models import GapRow
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.execute(delete(GapRow).where(GapRow.determinism == "deterministic"))
        session.add_all(
            GapRow(
                concept_id=g.concept_id,
                tier=g.tier,
                determinism=g.determinism,
                kind=g.kind,
                evidence_json=json.dumps(list(g.evidence.fact_ids)),
                rating=g.rating,
                status=g.status,
                graph_version=version,
            )
            for g in gaps
        )


def _write_stochastic_gap_rows(suggestions: list[Gap], version: str) -> int:
    """Upsert Tier-2a stochastic suggestions by concept identity (status-preserving).

    Unlike :func:`_write_gap_rows`' deterministic replace, this path never deletes: a
    concept already carrying a ``promoted``/``dismissed`` stochastic row is left alone
    (a fresh suggestion must not downgrade a human's curation decision — the
    "compounding arrow"); a concept with no stochastic row yet, or one still
    ``surfaced``, gets its row inserted/updated to the new suggestion. Suggestion
    identity is the concept: :func:`gap_suggest.suggest_for_thin` emits at most one
    suggestion per concept per call. Returns the number of rows written.
    """
    import json

    from sqlalchemy import select

    from doc_assistant.db.models import GapRow
    from doc_assistant.db.session import session_scope

    if not suggestions:
        return 0

    with session_scope() as session:
        existing = {
            row.concept_id: row
            for row in session.execute(
                select(GapRow).where(GapRow.determinism == "stochastic")
            ).scalars()
        }
        written = 0
        for g in suggestions:
            current = existing.get(g.concept_id)
            if current is not None and current.status in {"promoted", "dismissed"}:
                continue  # a human curation decision survives a re-suggest
            evidence_json = json.dumps(list(g.evidence.fact_ids))
            if current is None:
                session.add(
                    GapRow(
                        concept_id=g.concept_id,
                        tier=g.tier,
                        determinism=g.determinism,
                        kind=g.kind,
                        evidence_json=evidence_json,
                        rating=g.rating,
                        status=g.status,
                        graph_version=version,
                    )
                )
            else:
                current.tier = g.tier
                current.kind = g.kind
                current.evidence_json = evidence_json
                current.rating = g.rating
                current.graph_version = version
                # status stays "surfaced" — current.status was already checked above
            written += 1
    return written


def build_gaps(
    *,
    apply: bool,
    skeleton_dir: Path | None = None,
    min_degree: int,
    suggest: bool = False,
    client: LLMClient | None = None,
) -> GapsResult:
    """Compute Tier-1 + the Tier-2a deterministic floor; write the ``gaps`` sidecar.

    Read-only + free (no LLM): loads ``skeleton.json`` + the curated vocabulary +
    every ``unsupported``-marked claim, runs the pure detectors, and (on ``apply``)
    replaces the deterministic ``gaps`` rows (regenerable sidecar — dropped + rebuilt
    with the skeleton; stochastic rows persist across the rebuild, keyed by concept
    and status-preserving). A dry run computes + reports but writes nothing. Idempotent:
    same skeleton + same claims → identical row count + ``graph_version``. Never
    touches the chunk store or the curated vocabulary.

    ``suggest`` additionally runs the Tier-2a stochastic ceiling
    (``gap_suggest.suggest_for_thin``) over the ``under_connected`` Tier-1 gaps.
    This module makes **no provider decision** — the caller
    (``scripts/build_gaps.py``) resolves the provider/model, routes ``--apply``
    through ``llm.assert_provider_intent``, and hands an already-built ``client``
    here. ``suggest`` only calls the LLM when ``apply`` is also true (a dry run
    with ``--suggest`` reports zero suggested rows and makes zero LLM calls,
    matching every other enrichment CLI's dry-run contract); when ``apply`` and
    ``suggest`` are both true, ``client`` is required.
    """
    import json

    from doc_assistant.concept_skeleton import SKELETON_NAME, load_concepts, skeleton_from_dict
    from doc_assistant.config import CONCEPT_SKELETON_DIR

    root = skeleton_dir or CONCEPT_SKELETON_DIR
    skeleton_path = root / SKELETON_NAME
    if not skeleton_path.exists():
        raise FileNotFoundError(
            f"No concept skeleton at {skeleton_path} — run `python -m scripts."
            "build_concept_skeleton --apply` first (the gap layer is defined over it)."
        )
    skeleton = skeleton_from_dict(json.loads(skeleton_path.read_text(encoding="utf-8")))
    concepts, aliases = load_concepts()

    t1 = [
        *detect_isolated(skeleton),
        *detect_single_source(skeleton),
        *detect_thin_bridges(skeleton),
        *detect_under_connected(skeleton, min_degree=min_degree),
    ]
    claims = load_unsupported_claims()
    t2a = detect_unsourced_claims(claims, concepts, aliases)
    all_gaps = t1 + t2a

    version = str(skeleton.meta.get("graph_version", ""))
    if apply:
        _write_gap_rows(all_gaps, version)

    n_suggested = 0
    if suggest and apply:
        if client is None:
            raise ValueError("build_gaps(suggest=True, apply=True) requires an LLMClient")
        from doc_assistant.gap_suggest import suggest_for_thin

        suggestions = suggest_for_thin(t1, skeleton, client)
        n_suggested = _write_stochastic_gap_rows(suggestions, version)

    return GapsResult(
        gaps=all_gaps,
        graph_version=version,
        n_t1=len(t1),
        n_t2a=len(t2a),
        applied=apply,
        n_suggested=n_suggested,
    )
