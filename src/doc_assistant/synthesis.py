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

# Citation-robustness (2026-07-14). LLMs (haiku especially) routinely cite in
# non-canonical-but-unambiguous forms the bare-[n] parser used to drop — [Source 2],
# [Sources 2, 4], [2, 4], [2 and 4] — so a claim that DID cite read as "uncited" and the
# form was scored "malformed". We *parse* these forms (resolve them to source numbers); we
# do NOT rewrite the answer text — surface-don't-mutate still holds (see audit_citations).
# A token resolves iff its interior is an optional "source(s)"/"ref(s)" label followed by a
# ,/;/&/and-separated integer list; truly unparseable attempts ([karp2020dense], (paper.pdf))
# still surface as malformed.
_CITATION_TOKEN_RE = re.compile(
    r"\[\s*(?:sources?|refs?)?\s*(\d+(?:\s*(?:,|;|&|and)\s*\d+)*)\s*\]",
    re.IGNORECASE,
)
_DIGITS_RE = re.compile(r"\d+")


def cited_source_numbers(text: str) -> list[int]:
    """Every source number referenced by a citation token in ``text``, in order, across
    canonical (``[2]``) and common non-canonical (``[Source 2]``, ``[2, 4]``) forms.

    Reads the citations only — the answer's presentation is never altered."""
    nums: list[int] = []
    for m in _CITATION_TOKEN_RE.finditer(text):
        nums.extend(int(x) for x in _DIGITS_RE.findall(m.group(1)))
    return nums


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
        for n in cited_source_numbers(sentence):
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
# Post-hoc citation audit (Integrity Chunk 2a, surfacing layer)
# ============================================================
# A bracket token containing a letter — a candidate citation attempt. audit_citations then
# subtracts the ones that ARE resolvable citations (``[Source 2]`` via _CITATION_TOKEN_RE),
# leaving only the genuinely unparseable forms ([Smith2020], [karp2020dense]) that the model
# meant as a cite but no parser can map — surfacing them tells us the model *tried* to cite.
_MALFORMED_BRACKET_RE = re.compile(r"\[[^\]]*[A-Za-z][^\]]*\]")
# A file name cited inline in parentheses: (paper.pdf), (dpr_kumar_2020.pdf).
_FILENAME_CITE_RE = re.compile(r"\([^()\s]*\.(?:pdf|md|epub|docx?|html?|txt)\)", re.IGNORECASE)


@dataclass
class CitationAudit:
    """Structural audit of an answer's inline citations vs the retrieved sources."""

    valid: list[int]  # distinct in-range source numbers actually cited
    out_of_range: list[int]  # numeric citations outside 1..n_sources (hallucinated)
    malformed: list[str]  # non-numeric citation attempts the [n] parser ignores
    n_sentences: int
    n_uncited_sentences: int

    @property
    def clean(self) -> bool:
        """No hallucinated numbers and no malformed citation attempts."""
        return not self.out_of_range and not self.malformed

    @property
    def reasons(self) -> list[str]:
        out: list[str] = []
        if self.out_of_range:
            out.append(f"out-of-range citations {self.out_of_range}")
        if self.malformed:
            shown = ", ".join(self.malformed[:3])
            out.append(f"{len(self.malformed)} malformed citation(s): {shown}")
        return out

    def note(self) -> str:
        """One-line human summary (for the dev bundle / a UI notice)."""
        base = (
            f"{len(self.valid)} valid citation(s); "
            f"{self.n_uncited_sentences}/{self.n_sentences} sentences uncited"
        )
        return base + (f"; {'; '.join(self.reasons)}" if self.reasons else "")


def audit_citations(answer: str, n_sources: int) -> CitationAudit:
    """Audit an answer's inline citations against the retrieved sources (pure).

    Surfaces — never rewrites (surface-don't-mutate) — what a naive parser can't:
    valid in-range citations (canonical ``[n]`` *and* non-canonical ``[Source 2]`` /
    ``[2, 4]`` forms — see ``cited_source_numbers``), out-of-range numbers, genuinely
    malformed attempts ([Smith2020], (paper.pdf)) that still read as plain uncited text,
    and how many sentences carry no citation. Deterministic."""
    sentences = split_sentences(answer)
    nums = cited_source_numbers(answer)
    valid = sorted({n for n in nums if 1 <= n <= n_sources})
    out_of_range = sorted({n for n in nums if n < 1 or n > n_sources})
    # A bracketed-with-letters token or inline filename that is NOT a resolvable citation
    # ([Source 2] etc. now read as citations, so they no longer count as "malformed").
    candidates = dict.fromkeys(
        _MALFORMED_BRACKET_RE.findall(answer) + _FILENAME_CITE_RE.findall(answer)
    )
    malformed = [tok for tok in candidates if not _CITATION_TOKEN_RE.fullmatch(tok)]
    n_uncited = sum(1 for s in sentences if not cited_source_numbers(s))
    return CitationAudit(
        valid=valid,
        out_of_range=out_of_range,
        malformed=malformed,
        n_sentences=len(sentences),
        n_uncited_sentences=n_uncited,
    )


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
