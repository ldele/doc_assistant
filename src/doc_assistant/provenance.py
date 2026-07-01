"""Provenance card — PR 5 / Phase 5 / Integrity Chunk 1.

Captures everything needed to reproduce or audit one generated answer:
the query, the retrieved chunks (with reranker scores), the model +
config that produced the answer, token cost, latency.

Locked design choices
---------------------

* **Sidecar table.** Lives in the existing SQLite store but never
  touches the chunk store. Follows the Enrichment-Layer Pattern.
* **JSON for retrieved_chunks.** The shape is variable (number of
  chunks, per-chunk metadata) — a separate normalised table would add
  joins for marginal benefit at this scale.
* **`prompt_version`** = stable hash of `(template_hash, top_k,
  use_parent_child, embedding_model)`. Same config = same version
  string. Lets two records be compared apples-to-apples even after the
  prompt template is edited.
* **UUIDs everywhere.** `id` and `session_id` are never auto-increments
  so multi-user later is a non-breaking change.
* **UI-agnostic.** This module returns dataclasses. The provenance card
  and the `/export-record` slash command both read from here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from doc_assistant.db.models import AnswerClaim, AnswerRecord
from doc_assistant.db.session import session_scope

if TYPE_CHECKING:
    from doc_assistant.synthesis import Claim


@dataclass
class RetrievedChunk:
    """One retrieved chunk's per-chunk metadata, suitable for JSON storage."""

    filename: str | None = None
    doc_id: str | None = None
    page: int | None = None
    section: str | None = None
    reranker_score: float | None = None
    chunk_excerpt: str | None = None  # first ~300 chars, for display
    # Wider grounding text the reviewer/judge sees (REVIEWER_EVIDENCE_CHARS). Transient:
    # excluded from the persisted JSON and the UI card — only the reviewer reads it.
    full_text: str | None = None
    # Stable epistemics-format key ``{document_id}:{chunk_index}`` for the 7d marker join
    # (PR-M0 / ADR-2). Populated for flat/baseline chunks; ``None`` for parent-child chunks
    # (the PC→baseline mapping is PR-M1's decision). Transient: excluded from the persisted
    # JSON, like ``full_text`` — a join key, not stored provenance.
    chunk_key: str | None = None


@dataclass
class AnswerProvenance:
    """The full provenance payload for one answer — what gets persisted + displayed."""

    id: str
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    original_query: str | None = None
    model_name: str | None = None
    embedding_model: str | None = None
    prompt_version: str | None = None
    top_k: int | None = None
    use_parent_child: bool | None = None
    token_input: int | None = None
    token_output: int | None = None
    latency_ms: float | None = None
    session_id: str | None = None
    error: str | None = None
    created_at: datetime | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (datetimes → isoformat strings)."""
        d = asdict(self)
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d


# ============================================================
# Stable prompt-version hash
# ============================================================

_PROMPT_VERSION_LEN = 12


def prompt_version_hash(
    *,
    template_hash: str,
    top_k: int,
    use_parent_child: bool,
    embedding_model: str,
) -> str:
    """Stable, reproducible version string for a prompt + retrieval config.

    Two answers with the same config get the same string. Edits to the
    template (passed in as ``template_hash``) or to any retrieval knob
    change the version. The output is the first 12 chars of a sha256 over
    the canonical JSON of the inputs.
    """
    payload = json.dumps(
        {
            "template_hash": template_hash,
            "top_k": top_k,
            "use_parent_child": use_parent_child,
            "embedding_model": embedding_model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_PROMPT_VERSION_LEN]


def template_hash(template: str) -> str:
    """Hash of the prompt template string. First 12 chars of sha256."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:_PROMPT_VERSION_LEN]


# ============================================================
# Persist / read
# ============================================================


def record_answer(
    *,
    query: str,
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
    original_query: str | None = None,
    model_name: str | None = None,
    embedding_model: str | None = None,
    prompt_version: str | None = None,
    top_k: int | None = None,
    use_parent_child: bool | None = None,
    token_input: int | None = None,
    token_output: int | None = None,
    latency_ms: float | None = None,
    session_id: str | None = None,
    error: str | None = None,
) -> str:
    """Persist one answer record. Returns the new ``id``."""
    # Exclude the transient fields — the wide reviewer-only `full_text` and the 7d join key
    # `chunk_key` (ADR-2). The persisted card keeps the compact display excerpt only.
    chunks_json = json.dumps(
        [
            {k: v for k, v in asdict(c).items() if k not in ("full_text", "chunk_key")}
            for c in retrieved_chunks
        ]
    )
    with session_scope() as session:
        record = AnswerRecord(
            session_id=session_id,
            query=query,
            original_query=original_query,
            answer=answer,
            retrieved_chunks_json=chunks_json,
            model_name=model_name,
            embedding_model=embedding_model,
            prompt_version=prompt_version,
            top_k=top_k,
            use_parent_child=use_parent_child,
            token_input=token_input,
            token_output=token_output,
            latency_ms=latency_ms,
            error=error,
        )
        session.add(record)
        session.flush()
        return str(record.id)


# ============================================================
# Chunk 2a — adjudication log (answer_claims sidecar)
# ============================================================

VALID_DECISIONS = ("pending", "accepted", "rejected", "edited")


@dataclass
class PersistedClaim:
    """One adjudication-log row, detached for UI / export (Chunk 3)."""

    id: str
    claim_index: int
    claim_text: str
    citations: list[dict[str, Any]]
    marker: str
    decision: str
    edited_text: str | None


def _claim_citations_json(claim: Claim) -> str:
    return json.dumps(
        [
            {"source_number": c.source_number, "filename": c.filename, "page": c.page}
            for c in claim.citations
        ]
    )


def record_claims(answer_record_id: str, claims: list[Claim]) -> list[str]:
    """Eager-insert the segmented claims as ``pending``. Returns the new ids in order."""
    ids: list[str] = []
    with session_scope() as session:
        for claim in claims:
            row = AnswerClaim(
                answer_record_id=answer_record_id,
                claim_index=claim.claim_index,
                claim_text=claim.text,
                citations_json=_claim_citations_json(claim),
                marker=claim.marker,
                decision="pending",
            )
            session.add(row)
            session.flush()
            ids.append(str(row.id))
    return ids


def adjudicate_claim(claim_id: str, decision: str, edited_text: str | None = None) -> None:
    """Record the user's verdict on one claim. ``edited_text`` only for ``edited``."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}, got {decision!r}")
    with session_scope() as session:
        row = session.get(AnswerClaim, claim_id)
        if row is None:
            raise KeyError(f"no answer_claim with id {claim_id!r}")
        row.decision = decision
        row.edited_text = edited_text if decision == "edited" else None
        row.decided_at = datetime.now(timezone.utc)


def get_claims(answer_record_id: str) -> list[PersistedClaim]:
    """Return the adjudication log for one answer, ordered by claim_index."""
    with session_scope() as session:
        rows = session.scalars(
            select(AnswerClaim)
            .where(AnswerClaim.answer_record_id == answer_record_id)
            .order_by(AnswerClaim.claim_index)
        ).all()
        return [
            PersistedClaim(
                id=str(r.id),
                claim_index=r.claim_index,
                claim_text=r.claim_text,
                citations=json.loads(r.citations_json or "[]"),
                marker=r.marker,
                decision=r.decision,
                edited_text=r.edited_text,
            )
            for r in rows
        ]


def _row_to_provenance(row: AnswerRecord) -> AnswerProvenance:
    chunks_raw = json.loads(row.retrieved_chunks_json or "[]")
    chunks = [RetrievedChunk(**c) for c in chunks_raw]
    return AnswerProvenance(
        id=str(row.id),
        query=row.query,
        answer=row.answer,
        retrieved_chunks=chunks,
        original_query=row.original_query,
        model_name=row.model_name,
        embedding_model=row.embedding_model,
        prompt_version=row.prompt_version,
        top_k=row.top_k,
        use_parent_child=row.use_parent_child,
        token_input=row.token_input,
        token_output=row.token_output,
        latency_ms=row.latency_ms,
        session_id=row.session_id,
        error=row.error,
        created_at=row.created_at,
    )


def get_record(record_id: str) -> AnswerProvenance | None:
    """Look up a record by full id."""
    with session_scope() as session:
        row = session.execute(
            select(AnswerRecord).where(AnswerRecord.id == record_id)
        ).scalar_one_or_none()
        return _row_to_provenance(row) if row else None


def find_record_by_short_id(short_id: str) -> AnswerProvenance | None:
    """Look up a record by an id prefix (first 8+ chars).

    Returns the unique match if exactly one exists, otherwise None.
    """
    with session_scope() as session:
        rows = (
            session.execute(select(AnswerRecord).where(AnswerRecord.id.like(f"{short_id}%")))
            .scalars()
            .all()
        )
        if len(rows) == 1:
            return _row_to_provenance(rows[0])
        return None


def list_recent_records(limit: int = 20) -> list[AnswerProvenance]:
    """Return the most recent answer records."""
    with session_scope() as session:
        rows = (
            session.execute(
                select(AnswerRecord).order_by(AnswerRecord.created_at.desc()).limit(limit)
            )
            .scalars()
            .all()
        )
        return [_row_to_provenance(r) for r in rows]


# ============================================================
# Heuristic confidence signals — PR 5.1
# ============================================================
#
# These derive from the AnswerRecord data we already capture; no extra
# LLM calls. Surfaces the "this answer is probably weak" signal to the
# UI without inventing a confidence score the model didn't actually
# express. See docs/decisions.md → Research Integrity Layer for why we
# avoid self-reported confidence and use retrieval-derived markers
# instead.
#
# Thresholds are sensible defaults for bge-reranker-base (sigmoid scores
# in [0, 1]); tune via the constants below or pass overrides to
# compute_confidence_signals.

WEAK_RETRIEVAL_THRESHOLD = 0.3
"""Max reranker score below which retrieval is considered weak."""

SCORE_CLUSTER_SPAN = 0.05
"""Top-3 span (max - min) below which scores are 'tightly clustered'."""

SCORE_CLUSTER_MAX = 0.7
"""Cluster concern only fires when max is BELOW this — high-confidence
clusters (all scores > 0.7) indicate consensus, not ambiguity."""

SINGLE_SOURCE_MAX_DOCS = 2
"""Single-source flag fires when unique source filenames <= this."""


@dataclass
class ConfidenceSignals:
    """Heuristic confidence flags derived from an AnswerProvenance.

    Each flag is independent. ``any()`` returns True if at least one
    flag fires; consumers (the UI) use that to decide whether to render
    a warning chip. ``reasons`` lists the short labels of fired flags
    for display.
    """

    weak_retrieval: bool = False
    score_cluster_concern: bool = False
    single_source_risk: bool = False
    # Numeric details for the UI / future tuning.
    max_score: float | None = None
    top3_span: float | None = None
    unique_sources: int | None = None

    def any(self) -> bool:
        return self.weak_retrieval or self.score_cluster_concern or self.single_source_risk

    @property
    def reasons(self) -> list[str]:
        flags = []
        if self.weak_retrieval:
            flags.append("weak retrieval")
        if self.score_cluster_concern:
            flags.append("ambiguous top matches")
        if self.single_source_risk:
            flags.append("single-source")
        return flags


def compute_confidence_signals(
    prov: AnswerProvenance,
    *,
    weak_threshold: float = WEAK_RETRIEVAL_THRESHOLD,
    cluster_span: float = SCORE_CLUSTER_SPAN,
    cluster_max: float = SCORE_CLUSTER_MAX,
    single_source_max_docs: int = SINGLE_SOURCE_MAX_DOCS,
) -> ConfidenceSignals:
    """Compute heuristic confidence flags from the retrieved-chunks data.

    Pure function. Operates only on ``prov.retrieved_chunks``; doesn't
    touch the DB. With no chunks, returns all-False (the answer was
    generated without retrieval — caller should treat this as its own
    concern, but it's not what these signals are for).
    """
    chunks = prov.retrieved_chunks
    if not chunks:
        return ConfidenceSignals()

    scores = [c.reranker_score for c in chunks if c.reranker_score is not None]
    if not scores:
        # No score data — can't compute retrieval-quality signals. Only
        # the single-source check is still meaningful.
        unique_sources = len({c.filename for c in chunks if c.filename})
        return ConfidenceSignals(
            single_source_risk=(unique_sources <= single_source_max_docs and unique_sources > 0),
            unique_sources=unique_sources,
        )

    max_score = max(scores)
    top3 = sorted(scores, reverse=True)[:3]
    top3_span = max(top3) - min(top3) if len(top3) >= 2 else 0.0

    weak = max_score < weak_threshold
    # Cluster concern: scores are tightly clustered AND the cluster is in
    # the medium range (not all high — high-and-clustered = consensus).
    clustered = (top3_span < cluster_span) and (max_score < cluster_max) and (not weak)

    unique_sources = len({c.filename for c in chunks if c.filename})
    single_source = unique_sources > 0 and unique_sources <= single_source_max_docs

    return ConfidenceSignals(
        weak_retrieval=weak,
        score_cluster_concern=clustered,
        single_source_risk=single_source,
        max_score=max_score,
        top3_span=top3_span,
        unique_sources=unique_sources,
    )
