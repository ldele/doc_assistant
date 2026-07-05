"""Curate the auto-seeded concept vocabulary down to real concepts.

``seed_concepts --promote-all`` bootstraps a broad vocabulary from mined keywords, but at
corpus scale that vocabulary carries extraction noise — DOI/date/license fragments, single-
character tokens, author eponyms, and sentence fragments that are not reusable concepts. This
module is the pruning counterpart to seeding: it removes non-concepts and merges near-duplicates
so the curated ``Concept`` vocabulary (and everything derived from it — the skeleton, wiki, Node
B) sharpens.

Three stages, cheapest-first so the expensive one sees fewer candidates:

1. **Artifact filter (pure, free).** Deterministic regex: a label with a pure-digit token
   (``2015``, ``e78635``), a single-character token (``bert s``), or ≤2 chars is an extraction
   artifact, not a concept. High precision — it never fires on multi-word technical terms.
2. **LLM classification (Ollama, confined).** The survivors are batched to a local model that
   flags author names / dates / fragments the regex can't see (``wenckebach``, ``2015 volume``).
   Provider-isolated exactly like Node B (``assert_provider_intent``); the model only *labels* —
   it never invents or renames a concept.
3. **Near-duplicate merge (embeddings).** ``concept_semantics.nearest_pairs`` surfaces
   label pairs above a cosine threshold (``text embedding`` ↔ ``text embeddings``); the survivor
   with more document links keeps the pair's aliases.

Pure core (``is_artifact`` / ``build_classify_messages`` / ``parse_noise_indices`` /
``plan_merges``) is deterministic and unit-testable; the impure boundary applies the plan to the
``concepts`` / ``concept_aliases`` tables. Curation mutates *curated* data (like seed_concepts),
so it is dry-run by default and only writes on ``--apply``. The derived skeleton/presence tables
are regenerated afterwards by ``build_concept_skeleton`` — this module never touches them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import structlog

from doc_assistant.llm import LLMClient, Message

log = structlog.get_logger(__name__)

CLASSIFY_BATCH = 25
CLASSIFY_MAX_TOKENS = 512

_PURE_DIGIT = re.compile(r"^\d+$")


def is_artifact(label: str) -> bool:
    """True if ``label`` is an extraction artifact rather than a concept (pure, deterministic).

    Fires on a pure-digit token (dates, DOIs, article numbers: ``2015``, ``10``, ``e78635`` is
    kept but ``111`` drops the whole label), a single-character token (``bert s`` → ``s``), or a
    label of ≤2 non-space characters. Multi-word technical terms (``res2net-50``, ``gpt-4``,
    ``dense retrieval``) survive — no token is bare-digits or a lone char.
    """
    stripped = label.strip()
    if len(stripped.replace(" ", "")) <= 2:
        return True
    tokens = stripped.split()
    return any(_PURE_DIGIT.match(t) or len(t) == 1 for t in tokens)


# --------------------------------------------------------------------------- #
# LLM classification — pure prompt + parse (behind the LLMClient seam)
# --------------------------------------------------------------------------- #
_CLASSIFY_SYSTEM = """You curate a controlled vocabulary of scientific/technical CONCEPTS mined
automatically from research papers. Some mined terms are NOT concepts — they are author or person
names, dates, DOI/citation/license fragments, journal or venue strings, or sentence fragments.

A CONCEPT is a reusable topic, method, model, task, phenomenon, dataset, or entity a reader would
look up (e.g. "dense retrieval", "deeplabcut", "deep brain stimulation", "contrastive learning").
NOISE is anything else — a person ("wenckebach", "rockland"), a date/number ("2015 volume"), a
DOI/license fragment, a venue ("elife 2020"), or a broken phrase ("n system prompt").

You are given a numbered list of terms. Respond with ONLY a JSON object listing the numbers of the
terms that are NOISE (not concepts):
{"noise": [0, 3, 7]}
If every term is a real concept, return {"noise": []}."""


def build_classify_messages(labels: list[str]) -> list[Message]:
    """One classification prompt for a batch of labels (numbered; model returns noise indices)."""
    listing = "\n".join(f"[{i}] {label}" for i, label in enumerate(labels))
    user = f"Terms:\n{listing}\n\nReturn the JSON object of NOISE indices."
    return [{"role": "system", "content": _CLASSIFY_SYSTEM}, {"role": "user", "content": user}]


def parse_noise_indices(text: str, n: int) -> set[int]:
    """Parse ``{"noise": [...]}`` into a validated set of in-range indices (empty on failure).

    Tolerant of a code fence or a bare list. On any parse failure returns an empty set — the
    caller then keeps that whole batch (a failed classification never drops a concept).
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw[:4].lower() == "json" else raw
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return set()
    items = data.get("noise", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return set()
    out: set[int] = set()
    for x in items:
        try:
            idx = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n:
            out.add(idx)
    return out


# --------------------------------------------------------------------------- #
# Near-duplicate merge planning (pure, over precomputed cosine pairs)
# --------------------------------------------------------------------------- #
@dataclass
class MergePlan:
    """A single merge: ``drop`` is folded into ``keep`` (keep's label wins, aliases union)."""

    keep_id: str
    keep_label: str
    drop_id: str
    drop_label: str


def plan_merges(
    pairs: list[tuple[str, str, float]],
    doc_count: dict[str, int],
    label_by_id: dict[str, str],
) -> list[MergePlan]:
    """Turn near-duplicate ``(id_a, id_b, score)`` pairs into a deduped, acyclic merge list.

    The survivor is the concept present in more documents (ties broken by shorter label, then id),
    so the canonical/broader form absorbs the narrower duplicate. Union-find keeps a term from
    being both a keep and a drop across overlapping pairs (transitive dupes fold into one root).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def better(a: str, b: str) -> str:
        # higher doc count wins; then shorter label; then lexical id — all deterministic
        ka = (-doc_count.get(a, 0), len(label_by_id.get(a, "")), a)
        kb = (-doc_count.get(b, 0), len(label_by_id.get(b, "")), b)
        return a if ka <= kb else b

    for a, b, _score in sorted(pairs, key=lambda p: (-p[2], p[0], p[1])):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        winner = better(ra, rb)
        loser = rb if winner == ra else ra
        parent[loser] = winner

    plans: list[MergePlan] = []
    for node in parent:
        root = find(node)
        if node != root:
            plans.append(
                MergePlan(
                    keep_id=root,
                    keep_label=label_by_id.get(root, root),
                    drop_id=node,
                    drop_label=label_by_id.get(node, node),
                )
            )
    plans.sort(key=lambda m: (m.keep_label, m.drop_label))
    return plans


# --------------------------------------------------------------------------- #
# Classification orchestration (impure — batched LLM calls)
# --------------------------------------------------------------------------- #
@dataclass
class CurationPlan:
    """What a dry run computed: the ids to remove (with reason) and the merges to apply."""

    artifacts: list[tuple[str, str]] = field(default_factory=list)  # (id, label)
    llm_noise: list[tuple[str, str]] = field(default_factory=list)  # (id, label)
    merges: list[MergePlan] = field(default_factory=list)
    n_calls: int = 0

    @property
    def remove_ids(self) -> set[str]:
        return {cid for cid, _ in self.artifacts} | {cid for cid, _ in self.llm_noise}


def classify_noise(
    concepts: list[tuple[str, str]],
    client: LLMClient,
    *,
    batch_size: int = CLASSIFY_BATCH,
) -> list[tuple[str, str]]:
    """LLM-flag the noise concepts among ``(id, label)`` survivors (batched, graceful)."""
    noise: list[tuple[str, str]] = []
    calls = 0
    for start in range(0, len(concepts), batch_size):
        batch = concepts[start : start + batch_size]
        labels = [label for _, label in batch]
        try:
            text = client.complete(
                build_classify_messages(labels), temperature=0.0, max_tokens=CLASSIFY_MAX_TOKENS
            )
            idxs = parse_noise_indices(text, len(batch))
        except Exception as exc:  # one bad batch keeps its concepts, never sinks the run
            log.warning("curation_batch_failed", start=start, error=str(exc))
            continue
        calls += 1
        noise.extend(batch[i] for i in sorted(idxs))
    log.info("classify_noise_done", calls=calls, flagged=len(noise))
    return noise


# --------------------------------------------------------------------------- #
# Impure boundary — read the curated vocabulary, apply the plan
# --------------------------------------------------------------------------- #
def load_concepts() -> list[tuple[str, str]]:
    """All curated concepts as ``(id, label)``, sorted by id (stable)."""
    from sqlalchemy import select

    from doc_assistant.db.models import Concept
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        rows = [(str(c.id), c.label) for c in session.execute(select(Concept)).scalars()]
    rows.sort(key=lambda r: r[0])
    return rows


def doc_counts() -> dict[str, int]:
    """Concept id → number of distinct documents it is present in (from the presence sidecar)."""
    from sqlalchemy import func, select

    from doc_assistant.db.models import ConceptPresenceRow
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        stmt = select(
            ConceptPresenceRow.concept_id,
            func.count(func.distinct(ConceptPresenceRow.document_id)),
        ).group_by(ConceptPresenceRow.concept_id)
        return {str(cid): int(n) for cid, n in session.execute(stmt)}


def dedup_pairs(
    concepts: list[tuple[str, str]], threshold: float, model: str | None
) -> list[tuple[str, str, float]]:
    """Near-duplicate ``(id_a, id_b, cosine)`` pairs among ``concepts`` via label embeddings.

    Runs on the host (loads the embedder), not the sandbox. Pass the *post-prune* survivors so a
    kept concept is never merged into one that is about to be removed.
    """
    from doc_assistant.concept_semantics import embed_texts, nearest_pairs

    if len(concepts) < 2:
        return []
    labels = [label for _, label in concepts]
    id_by_label: dict[str, str] = {label: cid for cid, label in concepts}
    vectors = embed_texts(labels, model=model)
    return [
        (id_by_label[p.label_a], id_by_label[p.label_b], p.cosine)
        for p in nearest_pairs(labels, vectors, threshold=threshold)
    ]


def remove_concepts(ids: set[str]) -> int:
    """Delete concepts (and their aliases) by id. Returns the number removed."""
    if not ids:
        return 0
    from sqlalchemy import delete

    from doc_assistant.db.models import Concept, ConceptAlias
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.execute(delete(ConceptAlias).where(ConceptAlias.concept_id.in_(ids)))
        session.execute(delete(Concept).where(Concept.id.in_(ids)))
    return len(ids)


def apply_merges(plans: list[MergePlan]) -> int:
    """Fold each ``drop`` into its ``keep`` (move surface forms to aliases), then delete it."""
    if not plans:
        return 0
    from sqlalchemy import delete

    from doc_assistant.db.models import Concept, ConceptAlias
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        for plan in plans:
            keep = session.get(Concept, plan.keep_id)
            drop = session.get(Concept, plan.drop_id)
            if keep is None or drop is None:
                continue
            existing = {a.alias for a in keep.aliases} | {keep.label}
            drop_aliases = {a.alias for a in drop.aliases} | {drop.label}
            for alias in sorted(drop_aliases - existing):
                session.add(ConceptAlias(concept_id=keep.id, alias=alias))
            session.execute(delete(ConceptAlias).where(ConceptAlias.concept_id == drop.id))
            session.execute(delete(Concept).where(Concept.id == drop.id))
    return len(plans)
