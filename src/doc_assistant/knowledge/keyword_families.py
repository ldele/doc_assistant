"""Pure core: deterministic, zero-LLM keyword-family detection (feature-tag-families.md, PR-2).

Two tiers, both $0/local, run over the set of un-familied keyword names (the caller — normally
``library.detect_family_candidates`` — excludes anything already a family's canonical or alias):

1. **Morphological** (``_tier1_morphological``) — a conservative plural/suffix stem collapses
   ``llms`` → ``llm``, ``connectomes`` → ``connectome``. Exact structural match, confidence 1.0.
2. **Embedding** (``_tier2_embedding``) — bge cosine clustering catches semantic near-synonyms a
   stem can't (``connectome`` ≈ ``connectomics``). Edit-distance rides along as a *supporting*
   signal folded into the reported confidence, never a gate — that would defeat the tier's purpose
   (spelling-different, meaning-close pairs are exactly what it exists to catch).

Nothing here writes to the DB or calls an LLM; nothing auto-applies. Callers review proposals and
apply them through ``library``'s existing family CRUD (PR-1). The embedding step is injected via
``embed_fn`` so this module never loads a model itself — the API route hands it the already-loaded
retrieval embedder (no new model load); the CLI script hands it a fresh one (loading is fine
there).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

_MIN_STEM_LEN = 3
DEFAULT_EMBEDDING_THRESHOLD = 0.86


def _stem(word: str) -> str:
    """A conservative plural/suffix stem — collapses ``llms``→``llm``, ``connectomes``→
    ``connectome``.

    Deliberately narrow (only strips a handful of common plural suffixes) so it doesn't over-merge
    unrelated short words; anything it can't confidently normalize is returned unchanged.
    """
    w = word.strip().casefold()
    if len(w) <= _MIN_STEM_LEN:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, iterative DP. Hand-rolled (no fuzzy-match dependency) — this repo
    already prefers small deterministic scoring over adding a library for one use (cf.
    `keywords.py`'s ``weirdness``/``c_value_scores``)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _edit_similarity(a: str, b: str) -> float:
    """``1 - normalized edit distance``, in ``[0, 1]`` (1 = identical)."""
    longer = max(len(a), len(b))
    if longer == 0:
        return 1.0
    return 1.0 - (_edit_distance(a, b) / longer)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass(frozen=True)
class FamilyProposal:
    """One proposed family: a suggested canonical + the other keywords to group under it."""

    canonical: str
    members: tuple[str, ...]  # sorted, casefold order; excludes canonical
    tier: Literal["morphological", "embedding"]
    confidence: float  # 0..1; morphological proposals are always 1.0 (an exact structural match)


def _canonical_and_members(names: list[str]) -> tuple[str, tuple[str, ...]]:
    """Pick the shortest (ties: casefold-alphabetical) name as the suggested canonical."""
    ordered = sorted(names, key=lambda g: (len(g), g.casefold()))
    canonical, members = ordered[0], ordered[1:]
    return canonical, tuple(sorted(members, key=str.casefold))


def _tier1_morphological(names: list[str]) -> list[FamilyProposal]:
    """Group keywords whose conservative stem matches. Deterministic; confidence is always 1.0 (a
    structural match, not a probabilistic score)."""
    by_stem: dict[str, list[str]] = {}
    for n in names:
        by_stem.setdefault(_stem(n), []).append(n)
    proposals: list[FamilyProposal] = []
    for group in by_stem.values():
        if len(group) < 2:
            continue
        canonical, members = _canonical_and_members(group)
        proposals.append(
            FamilyProposal(
                canonical=canonical, members=members, tier="morphological", confidence=1.0
            )
        )
    proposals.sort(key=lambda p: p.canonical.casefold())
    return proposals


def _tier2_embedding(
    names: list[str],
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    threshold: float,
) -> list[FamilyProposal]:
    """Cluster keywords by bge cosine similarity — union-find over pairs ≥ ``threshold``, so a
    chain (A~B, B~C) proposes one family, not two overlapping pairs. Edit-distance is blended into
    the reported confidence as a supporting signal (never a gate)."""
    if len(names) < 2:
        return []
    vectors = embed_fn(names)
    n = len(names)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    pair_scores: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            cosine = _cosine(vectors[i], vectors[j])
            if cosine >= threshold:
                union(i, j)
                edit_sim = _edit_similarity(names[i].casefold(), names[j].casefold())
                pair_scores[(i, j)] = 0.75 * cosine + 0.25 * edit_sim

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    proposals: list[FamilyProposal] = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        canonical, members = _canonical_and_members([names[i] for i in idxs])
        scores = [
            pair_scores[(a, b)]
            for pos_a, a in enumerate(idxs)
            for b in idxs[pos_a + 1 :]
            if (a, b) in pair_scores
        ]
        confidence = sum(scores) / len(scores) if scores else threshold
        proposals.append(
            FamilyProposal(
                canonical=canonical,
                members=members,
                tier="embedding",
                confidence=round(confidence, 4),
            )
        )
    proposals.sort(key=lambda p: (-p.confidence, p.canonical.casefold()))
    return proposals


def detect_family_proposals(
    names: list[str],
    *,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
) -> list[FamilyProposal]:
    """All proposed families over ``names`` (the caller filters out already-familied keywords —
    this module has no DB access).

    Tier 1 always runs. Tier 2 runs only when ``embed_fn`` is given (omit it — e.g. in a cheap unit
    test or a fast CLI dry run — to get Tier-1-only results, $0 and instant). Names a Tier-1 group
    already claims are excluded from Tier 2 (no point re-clustering them). Tier 1 proposals come
    first (exact), then Tier 2 (fuzzy), each sorted for determinism within its tier.
    """
    tier1 = _tier1_morphological(list(dict.fromkeys(names)))
    consumed = {n.casefold() for p in tier1 for n in (p.canonical, *p.members)}
    remaining = [n for n in names if n.casefold() not in consumed]
    tier2 = (
        _tier2_embedding(remaining, embed_fn, threshold=embedding_threshold) if embed_fn else []
    )
    return [*tier1, *tier2]
