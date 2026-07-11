"""Node B — confined LLM relation/stance enrichment of the concept skeleton.

The deferred second build node of the concept-graph redesign
(``docs/archive/concept-graph-redesign.md``, Decision 6). Handed **only** the concepts a
document already contains, one LLM call per document annotates the *existing* co-occurrence
edges between co-present concepts with a relation verb + a stance ∈ ``POLARITIES``. It
**never** creates a node or an edge (the by-construction confinement) and never touches
presence — Node A owns those. The pass is idempotent: it always re-derives each edge's
Node-B annotation from scratch (strips any prior ``llm_relation`` / stance first), so
re-running from an already-enriched skeleton is a no-op on unchanged inputs.

Provider isolation (KI-4 credit-leak guard): the caller resolves the provider from
``CONCEPT_SKELETON_LLM_PROVIDER`` (LOCAL Ollama by default, *not* ``LLM_PROVIDER``) and routes
the ``--apply`` run through ``llm.assert_provider_intent``. This module takes an already-built
``LLMClient`` and makes no provider decision itself.

Pure core (``annotate_relations`` / ``build_messages`` / ``parse_annotations``) is
LLM-behind-a-seam and DB-free — unit-testable with a fake ``LLMClient``. The impure boundary
(``load_present_by_doc`` / ``load_presence_rows``) reads only the derived ``concept_presence``
sidecar Node A wrote.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from itertools import combinations

import structlog

from doc_assistant.concept_skeleton import (
    POLARITIES,
    ConceptPresence,
    ConceptSkeleton,
    SkeletonEdge,
    _graph_version,
    edge_weight,
)
from doc_assistant.llm import LLMClient, Message

log = structlog.get_logger(__name__)

#: Node B is a JSON-shaped one-shot per document. Sized for corpus scale — a document can
#: co-present dozens of edge pairs, each ~15 output tokens; too small a budget truncates the
#: JSON and the whole document's annotations are dropped (parse failure).
DEFAULT_MAX_TOKENS = 2048
LLM_RELATION = "llm_relation"

_SYSTEM = """You annotate how research concepts relate *within a single document*.

You are given the concepts that co-occur in ONE document and a numbered list of concept
PAIRS that appear together in it. For EACH pair, decide two things from the document's
apparent framing:

- "relation": a short (1-4 word) verb phrase for how the first concept relates to the
  second (e.g. "is evaluated with", "improves on", "is a component of").
- "stance": EXACTLY one of:
    "supports"    - the document treats the two as compatible / mutually reinforcing.
    "refines"     - one extends, specialises, or improves the other.
    "contradicts" - the document puts them in tension, or one fails where the other holds.
    "supersedes"  - one is presented as replacing or obsoleting the other.

Annotate ONLY the pairs given, by their number. Never invent pairs, concepts, or numbers.
Respond with ONLY a JSON object of this exact shape:
{"annotations": [{"pair": 0, "relation": "...", "stance": "supports"}]}"""


def build_messages(present_labels: list[str], pair_labels: list[tuple[str, str]]) -> list[Message]:
    """The one-shot prompt for one document: its present concepts + numbered candidate pairs."""
    pairs_block = "\n".join(f"[{i}] {a} <-> {b}" for i, (a, b) in enumerate(pair_labels))
    user = (
        f"Document concepts present: {', '.join(present_labels)}\n\n"
        f"Pairs to annotate:\n{pairs_block}\n\n"
        "Return the JSON object."
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def parse_annotations(text: str, n_pairs: int) -> list[tuple[int, str, str]]:
    """Parse the model's JSON into validated ``(pair_index, relation, polarity)`` triples.

    Tolerant by design (local models drift): strips a ``json`` code fence, accepts either a
    bare list or ``{"annotations": [...]}``, and drops any item with an out-of-range pair
    index, an unknown stance, or a missing relation. A first-wins de-dup keeps one annotation
    per pair. Any parse failure returns ``[]`` — the caller then leaves those edges unchanged.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw[:4].lower() == "json" else raw
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    items = data.get("annotations", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item["pair"])
        except (KeyError, TypeError, ValueError):
            continue
        relation = str(item.get("relation", "")).strip()
        stance = str(item.get("stance", "")).strip().lower()
        if idx in seen or not (0 <= idx < n_pairs) or stance not in POLARITIES or not relation:
            continue
        seen.add(idx)
        out.append((idx, relation, stance))
    return out


def annotate_relations(
    skeleton: ConceptSkeleton,
    present_by_doc: dict[str, list[str]],
    client: LLMClient,
    *,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    require_provenance: frozenset[str] = frozenset(),
) -> ConceptSkeleton:
    """Annotate existing edges with an LLM relation verb + per-document stance (Node B).

    For each document (sorted for determinism), the LLM is handed only that document's present
    concepts and the subset of skeleton edges among them; returned stances are attached to the
    *existing* edge (``provenance`` gains ``"llm_relation"``). Pairs that are not already edges are
    never sent and never created (Decision 6). Idempotent: each edge's Node-B annotation is
    rebuilt from its Node-A base, so a re-run on unchanged inputs reproduces the skeleton
    byte-for-byte. A per-document LLM/parse failure is logged and skipped (that document simply
    contributes no stance); other documents still annotate.

    ``require_provenance`` caps the pass to *corroborated* edges: only edges whose provenance is
    a superset of it are eligible (e.g. ``{"citation", "similarity"}`` annotates just the
    cross-doc-backed edges, skipping the co-occurrence-only long tail). Empty (default) = every
    existing edge, the confined full pass.
    """
    label_by_id = {n.id: n.label for n in skeleton.nodes}
    edge_pairs = {
        (e.source_concept_id, e.target_concept_id)
        for e in skeleton.edges
        if e.provenance >= require_provenance
    }

    relation_by_pair: dict[tuple[str, str], str] = {}
    stances_by_pair: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    n_calls = 0
    for doc_id in sorted(present_by_doc):
        present = sorted({c for c in present_by_doc[doc_id] if c in label_by_id})
        candidates = [(a, b) for a, b in combinations(present, 2) if (a, b) in edge_pairs]
        if not candidates:
            continue
        messages = build_messages(
            [label_by_id[c] for c in present],
            [(label_by_id[a], label_by_id[b]) for a, b in candidates],
        )
        try:
            text = client.complete(messages, temperature=temperature, max_tokens=max_tokens)
            annotations = parse_annotations(text, len(candidates))
        except Exception as exc:  # transport/parse — one bad doc must not sink the run
            log.warning("node_b_doc_failed", document_id=doc_id, error=str(exc))
            continue
        n_calls += 1
        for idx, relation, stance in annotations:
            pair = candidates[idx]
            relation_by_pair.setdefault(pair, relation)  # first-wins → deterministic
            stances_by_pair[pair].append((doc_id, stance))

    new_edges: list[SkeletonEdge] = []
    n_annotated = 0
    for e in skeleton.edges:
        pair = (e.source_concept_id, e.target_concept_id)
        base_prov = e.provenance - {LLM_RELATION}  # strip any prior Node B → idempotent
        if pair in stances_by_pair:
            prov = base_prov | {LLM_RELATION}
            new_edges.append(
                replace(
                    e,
                    provenance=prov,
                    relation=relation_by_pair.get(pair),
                    stance_by_doc=tuple(stances_by_pair[pair]),
                    weight=edge_weight(prov, e.n_cooccurrence_chunks, e.provenance_strength),
                )
            )
            n_annotated += 1
        else:
            new_edges.append(
                replace(
                    e,
                    provenance=base_prov,
                    relation=None,
                    stance_by_doc=(),
                    weight=edge_weight(base_prov, e.n_cooccurrence_chunks, e.provenance_strength),
                )
            )
    new_edges.sort(key=lambda e: (e.source_concept_id, e.target_concept_id))

    seed = int(skeleton.meta.get("seed", 42))
    resolution = float(skeleton.meta.get("resolution", 1.0))
    version = _graph_version(
        list(skeleton.nodes),
        new_edges,
        seed=seed,
        resolution=resolution,
        doc_years=skeleton.meta.get("doc_years"),
    )
    meta = {
        **skeleton.meta,
        "graph_version": version,
        "n_llm_annotated_edges": n_annotated,
        "node_b_calls": n_calls,
    }
    return ConceptSkeleton(
        nodes=skeleton.nodes,
        edges=tuple(new_edges),
        communities=skeleton.communities,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Impure boundary — read the derived concept_presence sidecar Node A wrote
# --------------------------------------------------------------------------- #
def load_presence_rows() -> list[ConceptPresence]:
    """Read the ``concept_presence`` sidecar into ``ConceptPresence`` values (host, not sandbox).

    Node B reuses Node A's already-computed presence rather than re-matching against Chunk text.
    Requires a prior ``build_concept_skeleton --apply`` (empty list otherwise).
    """
    from sqlalchemy import select

    from doc_assistant.db.models import ConceptPresenceRow
    from doc_assistant.db.session import session_scope

    rows: list[ConceptPresence] = []
    with session_scope() as session:
        for r in session.execute(select(ConceptPresenceRow)).scalars():
            rows.append(
                ConceptPresence(
                    concept_id=str(r.concept_id),
                    document_id=str(r.document_id),
                    chunk_keys=tuple(json.loads(r.chunk_keys_json or "[]")),
                    n_mentions=int(r.n_mentions),
                )
            )
    rows.sort(key=lambda p: (p.concept_id, p.document_id))
    return rows


def present_by_doc(presences: list[ConceptPresence]) -> dict[str, list[str]]:
    """Invert presence rows into ``{document_id: [concept_id, ...]}`` for annotation."""
    out: dict[str, list[str]] = defaultdict(list)
    for p in presences:
        out[p.document_id].append(p.concept_id)
    return {doc: sorted(set(cids)) for doc, cids in out.items()}
