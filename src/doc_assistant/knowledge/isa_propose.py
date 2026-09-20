"""Propose ``is_a`` edges between concepts from their labels' heads (ROADMAP 51, ADR-028 D2/D8).

The concept→concept spine has never been built: 0 ``is_a`` edges against 357 concepts (ADR-045),
because nothing proposes one. ``taxonomy_propose`` places a concept **in a field** with a local
LLM; this module is its deterministic sibling for the other hierarchy edge — no model, no network,
no spend — and it writes ``origin="proposed"`` rows only, for a person to accept or reject.

**The rule.** A label whose tokens *end with* another concept's whole label is narrower than it:
``passage re-ranking`` → ``re-ranking``, ``beta oscillations`` → ``oscillations``. English puts the
head of a noun phrase last, so a shared tail is a shared head noun and the modifier in front of it
is what narrows the meaning.

**What it deliberately refuses**, both measured on the live vocabulary
(``tests/eval/baselines/isa_head_suffix_2026-09-20.md``):

- **A shared prefix is not hyponymy.** ``self-sorting memory`` is a kind of memory, not a kind of
  ``self-sorting``; ``saturation index`` is not a kind of ``saturation``. Both shapes sit in the
  merge baseline's "narrower" column, and both would be wrong as an edge in either direction.
- **Aliases are not word forms.** Matching a concept's aliases as well as its label adds 10
  candidates on this corpus, nearly all wrong (``ai benchmarks`` → ``benchmarks plateau``): an
  alias is a *different phrase*, so its head is not the concept's head. Labels only.

The rule proposes; it does not judge. A fragment in the vocabulary (``recog``, ``unlabeled``)
produces a candidate whose broader side is not a real concept — the reject half of D8 is the
answer to that, not a cleverer filter, and the fragments are row 93's problem (ADR-052).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from doc_assistant.db.models import Concept
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.keywords import tokenize
from doc_assistant.knowledge.taxonomy import add_hierarchy_edge

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class IsaCandidate:
    """One proposed ``narrow --is_a--> broad`` edge and the head the two labels share."""

    narrow_id: str
    narrow_label: str
    broad_id: str
    broad_label: str
    head: str

    def __str__(self) -> str:  # pragma: no cover - report formatting
        return f"{self.narrow_label!r} is_a {self.broad_label!r}"


@dataclass(frozen=True)
class IsaProposeResult:
    """What one pass found and (with ``apply``) wrote."""

    candidates: tuple[IsaCandidate, ...]
    n_concepts: int
    applied: bool = False
    n_written: int = 0
    n_skipped: int = 0


def head_suffix_candidates(concepts: Sequence[tuple[str, str]]) -> list[IsaCandidate]:
    """Candidate ``is_a`` edges among ``(id, label)`` pairs, by the shared-head rule.

    Pure — no DB, no model. A label is tokenised with the keyword tokenizer, so hyphenated
    technical terms stay whole (``cross-encoder``) and the match is case-insensitive. Two concepts
    with the *same* label produce nothing (no proper suffix), and neither does a one-token label,
    which has no modifier to drop. Only the **nearest** broader concept is proposed: an ancestor
    that another candidate already reaches is left out. One label's suffixes are nested, so its
    candidate parents form a chain and it ends up with a single proposed parent — a second parent
    is a judgement, and polyhierarchy (ADR-028 D1) stays something a person adds. Ordered by
    narrow label, then broad label, so a run is reproducible.
    """
    forms: dict[str, tuple[str, ...]] = {}
    labels: dict[str, str] = {}
    by_tokens: dict[tuple[str, ...], list[str]] = {}
    for cid, label in concepts:
        tokens = tuple(tokenize(label))
        if not tokens:
            continue
        forms[cid] = tokens
        labels[cid] = label
        by_tokens.setdefault(tokens, []).append(cid)

    candidates: list[IsaCandidate] = []
    for cid, tokens in forms.items():
        matches = [
            (suffix, broad_id)
            for cut in range(1, len(tokens))
            for suffix in (tokens[cut:],)
            for broad_id in by_tokens.get(suffix, [])
            if broad_id != cid
        ]
        for suffix, broad_id in matches:
            # Nearest broader only: `deep contrastive learning` is proposed under `contrastive
            # learning`, not also under `learning`, which that edge already reaches. The long way
            # round is not wrong, only redundant — and a review list is a person's time.
            if any(other != suffix and other[-len(suffix) :] == suffix for other, _ in matches):
                continue
            candidates.append(
                IsaCandidate(
                    narrow_id=cid,
                    narrow_label=labels[cid],
                    broad_id=broad_id,
                    broad_label=labels[broad_id],
                    head=" ".join(suffix),
                )
            )
    candidates.sort(key=lambda c: (c.narrow_label.casefold(), c.broad_label.casefold()))
    return candidates


def load_concept_labels(session: Session, *, graph_only: bool = False) -> list[tuple[str, str]]:
    """``(id, label)`` for every concept — the domain nodes are not concepts and never match.

    ``graph_only`` narrows to the graph vocabulary (``graph_include``). It is **off** by default,
    the opposite of ``taxonomy_propose``: 13 of 357 concepts are on the graph here, and the shared
    heads that make a spine (``pose``, ``oscillations``, ``passages``) sit in the other 344.
    """
    stmt = select(Concept.id, Concept.label).where(Concept.kind == "concept")
    if graph_only:
        stmt = stmt.where(Concept.graph_include.is_(True))
    return [(str(cid), label) for cid, label in session.execute(stmt).all()]


def write_candidates(session: Session, candidates: Sequence[IsaCandidate]) -> tuple[int, int]:
    """Write candidates as ``origin="proposed"`` ``is_a`` edges. Returns ``(written, skipped)``.

    Every write goes through the ``taxonomy.py`` seam, so a candidate that would close a cycle or
    name a non-concept is refused there, and an existing **curated** edge is left alone — a
    re-run cannot demote what the user accepted. One savepoint per candidate, for the reason
    ``taxonomy_propose.write_proposals`` has one: a rejected id fails the flush, and an unwound
    failed flush would cost the whole batch.
    """
    written = 0
    skipped = 0
    for candidate in candidates:
        try:
            with session.begin_nested():
                add_hierarchy_edge(
                    session, candidate.narrow_id, candidate.broad_id, "is_a", origin="proposed"
                )
        except (ValueError, IntegrityError) as exc:
            skipped += 1
            log.warning(
                "isa_propose_write_rejected",
                narrow=candidate.narrow_id,
                broad=candidate.broad_id,
                error=str(exc),
            )
            continue
        written += 1
    return written, skipped


def run_propose_isa(*, apply: bool = False, graph_only: bool = False) -> IsaProposeResult:
    """Find the candidates and, with ``apply``, write them as proposals.

    Without ``apply`` this reads the vocabulary and reports — it writes nothing, the same polarity
    every runner here has. Re-running with ``apply`` is idempotent: the seam keys on
    ``(source, target, type)``.
    """
    with session_scope() as session:
        concepts = load_concept_labels(session, graph_only=graph_only)
        candidates = tuple(head_suffix_candidates(concepts))
        if not apply:
            return IsaProposeResult(candidates=candidates, n_concepts=len(concepts))
        written, skipped = write_candidates(session, candidates)
        return IsaProposeResult(
            candidates=candidates,
            n_concepts=len(concepts),
            applied=True,
            n_written=written,
            n_skipped=skipped,
        )
