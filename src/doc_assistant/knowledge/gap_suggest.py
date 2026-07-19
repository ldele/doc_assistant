"""Tier-2a stochastic ceiling — quarantined LLM suggestions atop the Gap floor.

The deferred second half of Tier 2a (ADR-004 Decision 4 / ``docs/specs/feature-gap-detection.md``).
``gaps.py`` builds the deterministic Tier-1 detectors + the Tier-2a floor with zero LLM calls;
this module adds a **suggestion-only** pass over the ``under_connected`` nodes those detectors
surface: one quarantined LLM call per concept, handed only that concept and its already-present
neighbours, asking what might be missing. The result is never written back as fact — it comes
out as ordinary ``Gap`` values with ``determinism="stochastic"``, a ``rating``, and
``status="surfaced"``, for a human to promote or dismiss (the compounding arrow lives in
``GapRow.status``, not here).

Confinement (by construction, mirrors ``concept_skeleton_enrich``'s Node-B guarantee):

* Takes an already-built ``LLMClient`` — this module makes **no provider decision** (the caller,
  ``scripts/build_gaps.py``, resolves ``GAP_SUGGEST_LLM_PROVIDER``/``_MODEL`` and routes
  ``--apply`` through ``llm.assert_provider_intent`` first).
* Never creates a ``Concept``, an edge, or a skeleton node, and never mutates the ``skeleton``
  argument — it is read-only input, unchanged on return (quarantine guarantee).
* A per-concept transport/parse failure is logged and skipped; it does not sink the run.
* Zero ``under_connected`` gaps ⇒ zero LLM calls (checked before the loop, not caught after).
"""

from __future__ import annotations

import json

import structlog

from doc_assistant.knowledge.concept_skeleton import ConceptSkeleton
from doc_assistant.knowledge.gaps import Gap, GapEvidence, GapKind
from doc_assistant.llm import LLMClient, Message

log = structlog.get_logger(__name__)

#: One suggestion is a short JSON object — a small budget is plenty and keeps a local model fast.
DEFAULT_MAX_TOKENS = 512

SUGGESTED_LINK: GapKind = "suggested_link"
SUGGESTED_CONCEPT: GapKind = "suggested_concept"
THIN_AREA: GapKind = "thin_area"
SUGGESTION_KINDS: tuple[GapKind, ...] = (SUGGESTED_LINK, SUGGESTED_CONCEPT, THIN_AREA)

_SYSTEM = """You help spot a possible gap in a curated concept graph around ONE under-connected
concept. You are given the concept and the OTHER concepts it is already linked to (its present
neighbours) in this corpus.

Suggest exactly ONE of:
- "suggested_link"    - an existing-sounding concept this one is probably related to in this
                        corpus but isn't linked to yet. Name it in "target".
- "suggested_concept" - a DIFFERENT concept, not among the neighbours, that likely belongs in
                        the vocabulary near this one. Name it in "target".
- "thin_area"         - you aren't confident in either of the above; describe the thin area in
                        "target" as a short phrase (not a citation, not a sentence).

Rate your confidence in the suggestion as "rating", a number in [0, 1].
Respond with ONLY a JSON object of this exact shape, nothing else:
{"kind": "suggested_link", "target": "...", "rating": 0.6}"""


def build_messages(label: str, neighbour_labels: list[str]) -> list[Message]:
    """The one-shot prompt for one under-connected concept: itself + its present neighbours."""
    neighbours = ", ".join(neighbour_labels) if neighbour_labels else "(none)"
    user = f"Concept: {label}\nPresent neighbours: {neighbours}\n\nReturn the JSON object."
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def parse_suggestion(text: str) -> tuple[GapKind, str, float] | None:
    """Parse the model's JSON into a validated ``(kind, target, rating)`` triple.

    Tolerant by design (local models drift): strips a ``json`` code fence, requires ``kind`` to
    be one of :data:`SUGGESTION_KINDS`, ``target`` to be non-empty, and ``rating`` to parse as a
    float in ``[0, 1]``. Any deviation — malformed JSON, wrong shape, out-of-range rating — is a
    parse failure and returns ``None``; the caller skips that concept rather than guessing.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw[:4].lower() == "json" else raw
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    kind = str(data.get("kind", "")).strip()
    target = str(data.get("target", "")).strip()
    try:
        rating = float(data.get("rating", float("nan")))
    except (TypeError, ValueError):
        return None
    if kind not in SUGGESTION_KINDS or not target or not (0.0 <= rating <= 1.0):
        return None
    return kind, target, rating


def _neighbour_labels(skeleton: ConceptSkeleton, concept_id: str) -> list[str]:
    """Present-neighbour labels of ``concept_id`` over the *existing* skeleton edges only."""
    label_by_id = {n.id: n.label for n in skeleton.nodes}
    neighbour_ids: set[str] = set()
    for e in skeleton.edges:
        if e.source_concept_id == concept_id:
            neighbour_ids.add(e.target_concept_id)
        elif e.target_concept_id == concept_id:
            neighbour_ids.add(e.source_concept_id)
    return sorted(label_by_id[n] for n in neighbour_ids if n in label_by_id)


def suggest_for_thin(
    gaps: list[Gap],
    skeleton: ConceptSkeleton,
    client: LLMClient,
    *,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Gap]:
    """The Tier-2a stochastic ceiling: one quarantined LLM call per ``under_connected`` gap.

    ``gaps`` is the Tier-1 detector output (only its ``under_connected`` kind routes here —
    everything else is ignored); ``skeleton`` supplies each concept's label and present
    neighbours, read-only. Returns one suggestion ``Gap`` per concept that produced a parseable
    response (``tier="t2a"``, ``determinism="stochastic"``, ``status="surfaced"``); a concept
    whose call transport-failed or whose response didn't parse contributes none, silently
    (logged), so one bad concept never sinks the batch. ``evidence.fact_ids`` records the exact
    LLM inputs — the concept label, its present neighbours, and the suggested target — for
    observability (ADR-004's "expose LLM inputs, rate output" mandate). Never mutates
    ``skeleton`` and never touches the curated vocabulary or the skeleton sidecar.
    """
    label_by_id = {n.id: n.label for n in skeleton.nodes}
    thin = sorted({g.concept_id for g in gaps if g.kind == "under_connected"})
    if not thin:
        return []  # zero under-connected concepts -> zero LLM calls, checked before any call

    suggestions: list[Gap] = []
    for concept_id in thin:
        label = label_by_id.get(concept_id, concept_id)
        neighbours = _neighbour_labels(skeleton, concept_id)
        messages = build_messages(label, neighbours)
        try:
            text = client.complete(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:  # transport failure — one bad concept must not sink the run
            log.warning("gap_suggest_transport_failed", concept_id=concept_id, error=str(exc))
            continue
        parsed = parse_suggestion(text)
        if parsed is None:
            log.warning("gap_suggest_unparseable", concept_id=concept_id)
            continue
        kind, target, rating = parsed
        suggestions.append(
            Gap(
                concept_id=concept_id,
                tier="t2a",
                determinism="stochastic",
                kind=kind,
                evidence=GapEvidence(
                    fact_ids=(
                        f"concept={label}",
                        f"neighbours={','.join(neighbours)}",
                        f"target={target}",
                    )
                ),
                rating=rating,
                status="surfaced",
            )
        )
    return suggestions
