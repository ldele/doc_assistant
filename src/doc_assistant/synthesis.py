"""Phase 6 / Integrity Chunk 2a — dual interpretation (evidence + AI synthesis).

Turns one cited interpretation answer into adjudicable, **citation-anchored**
claims, each carrying a **retrieval-derived** uncertainty marker. Pure: no LLM
calls, no DB, no I/O. The integrity layer stays observable and deterministic —
markers come from retrieval signals (citation presence + the cited source's
reranker score), never self-reported model confidence.

The dual layer is a *presentation split* over the existing pipeline:
- **evidence layer** — the retrieved passages, rendered with provenance (no synthesis);
- **interpretation layer** — the existing LLM answer, segmented into claims (``ai`` mode only).

See ``docs/specs/chunk-2a-dual-interpretation.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from doc_assistant.provenance import WEAK_RETRIEVAL_THRESHOLD, ConfidenceSignals, RetrievedChunk

# Inline citation marker produced by ANSWER_PROMPT ("Cite sources inline
# using [1], [2], ..."). Source N maps to the Nth retrieved chunk (1-based).
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Sentence-ish boundary: terminal punctuation followed by whitespace. Cheap and
# deterministic (no NLP dep); ``edit`` is the escape hatch for coarse splits.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Per-claim markers (persisted on AnswerClaim.marker).
MARKER_OK = "ok"
MARKER_WEAK = "weak"
MARKER_UNSUPPORTED = "unsupported"


@dataclass
class ClaimCitation:
    """One source a claim cites. ``filename``/``page`` are None if the model
    cited a source number that doesn't map to a retrieved chunk."""

    source_number: int
    filename: str | None = None
    page: int | None = None


@dataclass
class Claim:
    """One adjudicable unit of the AI interpretation."""

    claim_index: int
    text: str
    citations: list[ClaimCitation] = field(default_factory=list)
    marker: str = MARKER_OK

    @property
    def source_numbers(self) -> list[int]:
        return [c.source_number for c in self.citations]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentence-ish claim units (deterministic, no NLP dep)."""
    return [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]


def claim_marker(
    citations: list[ClaimCitation],
    sources: list[RetrievedChunk],
    *,
    weak_threshold: float = WEAK_RETRIEVAL_THRESHOLD,
) -> str:
    """Retrieval-derived marker for one claim.

    - Uncited (or only hallucinated source numbers) -> ``unsupported``.
    - Cited a real source but no reranker score available -> ``weak``.
    - Cited; best real reranker score below the weak threshold -> ``weak``.
    - Otherwise -> ``ok``.
    """
    real_scores: list[float] = []
    has_real = False
    for c in citations:
        idx = c.source_number - 1
        if 0 <= idx < len(sources):
            has_real = True
            score = sources[idx].reranker_score
            if score is not None:
                real_scores.append(score)
    if not has_real:
        return MARKER_UNSUPPORTED
    if not real_scores:
        return MARKER_WEAK
    return MARKER_OK if max(real_scores) >= weak_threshold else MARKER_WEAK


def segment_claims(answer: str, sources: list[RetrievedChunk]) -> list[Claim]:
    """Split a cited interpretation answer into citation-anchored claims."""
    claims: list[Claim] = []
    for i, sentence in enumerate(split_sentences(answer)):
        citations: list[ClaimCitation] = []
        seen: set[int] = set()
        for raw in _CITATION_RE.findall(sentence):
            n = int(raw)
            if n in seen:
                continue
            seen.add(n)
            idx = n - 1
            if 0 <= idx < len(sources):
                src = sources[idx]
                citations.append(
                    ClaimCitation(source_number=n, filename=src.filename, page=src.page)
                )
            else:
                citations.append(ClaimCitation(source_number=n))
        claims.append(
            Claim(
                claim_index=i,
                text=sentence,
                citations=citations,
                marker=claim_marker(citations, sources),
            )
        )
    return claims


# ============================================================
# Rendering — markdown for each layer. Kept here (not apps/) so the UI stays a
# thin shell. Quiet on clean claims (UX: inform, don't clutter).
# ============================================================

_MARKER_BADGE = {
    MARKER_OK: "",
    MARKER_WEAK: " ⚠ weakly grounded",
    MARKER_UNSUPPORTED: " ⚠ unsupported",
}


def render_evidence_markdown(sources: list[RetrievedChunk]) -> str:
    """The deterministic evidence layer: retrieved passages + provenance."""
    if not sources:
        return "**Evidence** — no sources retrieved."
    lines = ["**Evidence** — what your sources say:"]
    for i, s in enumerate(sources, start=1):
        loc = s.filename or "unknown source"
        if s.page:
            loc += f", p.{s.page}"
        score = f" · relevance {s.reranker_score:.2f}" if s.reranker_score is not None else ""
        lines.append(f"{i}. *{loc}*{score}")
        if s.chunk_excerpt:
            lines.append(f"   > {s.chunk_excerpt.strip()}")
    return "\n".join(lines)


def render_interpretation_markdown(claims: list[Claim]) -> str:
    """The interpretation layer: AI synthesis, per-claim, markers inline."""
    lines = ["**AI interpretation** — the model's synthesis (review each claim):"]
    for c in claims:
        cites = f" [{', '.join(str(n) for n in c.source_numbers)}]" if c.citations else ""
        lines.append(f"{c.claim_index + 1}. {c.text}{cites}{_MARKER_BADGE.get(c.marker, '')}")
    return "\n".join(lines)


def format_banner(signals: ConfidenceSignals) -> str | None:
    """Whole-answer warning banner, or None when retrieval is clean (stay quiet)."""
    if not signals.any():
        return None
    return "⚠ " + "; ".join(signals.reasons) + " — interpret with caution."
