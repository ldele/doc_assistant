"""SQLAlchemy models for the doc-assistant library.

Design principles:
- SQLite is the source of truth for document-level metadata.
- Chroma is the source of truth for chunk embeddings.
- Both reference each other via document.id (stable UUID).
- Schema supports Phase 4 (citations) and beyond; unused fields stay NULL.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ============================================================
# Association tables (many-to-many)
# ============================================================

document_folders = Table(
    "document_folders",
    Base.metadata,
    Column(
        "document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    ),
    Column("folder_id", String, ForeignKey("folders.id", ondelete="CASCADE"), primary_key=True),
)

document_tags = Table(
    "document_tags",
    Base.metadata,
    Column(
        "document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    ),
    Column("tag_id", String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)

document_keywords = Table(
    "document_keywords",
    Base.metadata,
    Column(
        "document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    ),
    Column("keyword_id", String, ForeignKey("keywords.id", ondelete="CASCADE"), primary_key=True),
)


# ============================================================
# Folder — hierarchical, but UI starts flat
# ============================================================


class Folder(Base):
    __tablename__ = "folders"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_folder_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("folders.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    parent: Mapped["Folder | None"] = relationship(
        "Folder", remote_side=[id], back_populates="children"
    )
    children: Mapped[list["Folder"]] = relationship("Folder", back_populates="parent")
    documents: Mapped[list["Document"]] = relationship(
        "Document", secondary=document_folders, back_populates="folders"
    )

    __table_args__ = (UniqueConstraint("name", "parent_folder_id", name="uq_folder_name_parent"),)


# ============================================================
# Tag — user-applied organizational labels
# ============================================================


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    color: Mapped[str | None] = mapped_column(String, nullable=True)  # for UI
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    documents: Mapped[list["Document"]] = relationship(
        "Document", secondary=document_tags, back_populates="tags"
    )


# ============================================================
# Keyword — content-derived subject terms
# Distinct from tags: tags are user-applied for organization,
# keywords describe what the document is about.
# ============================================================


class Keyword(Base):
    __tablename__ = "keywords"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    source: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # "author", "extracted", "manual"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    documents: Mapped[list["Document"]] = relationship(
        "Document", secondary=document_keywords, back_populates="keywords"
    )


# ============================================================
# Document
# ============================================================


class Document(Base):
    __tablename__ = "documents"

    # Identity
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(String, nullable=False)
    source_original: Mapped[str] = mapped_column(String, nullable=False)
    source_cache: Mapped[str | None] = mapped_column(String, nullable=True)
    doc_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    format: Mapped[str] = mapped_column(String, nullable=False)

    # User-editable metadata
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    authors: Mapped[str | None] = mapped_column(String, nullable=True)  # JSON list as string
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Extraction & health
    extractor_used: Mapped[str | None] = mapped_column(String, nullable=True)
    extraction_health: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Lifecycle
    added_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    folders: Mapped[list[Folder]] = relationship(
        "Folder", secondary=document_folders, back_populates="documents"
    )
    tags: Mapped[list[Tag]] = relationship(
        "Tag", secondary=document_tags, back_populates="documents"
    )
    keywords: Mapped[list[Keyword]] = relationship(
        "Keyword", secondary=document_keywords, back_populates="documents"
    )
    parts: Mapped[list["DocumentPart"]] = relationship(
        "DocumentPart",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentPart.order_index",
    )
    citations_out: Mapped[list["Citation"]] = relationship(
        "Citation",
        foreign_keys="[Citation.source_document_id]",
        back_populates="source_document",
        cascade="all, delete-orphan",
    )
    citations_in: Mapped[list["Citation"]] = relationship(
        "Citation", foreign_keys="[Citation.target_document_id]", back_populates="target_document"
    )
    ingestion_events: Mapped[list["IngestionEvent"]] = relationship(
        "IngestionEvent",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="IngestionEvent.timestamp.desc()",
    )
    figures: Mapped[list["Figure"]] = relationship(
        "Figure",
        back_populates="document",
        cascade="all, delete-orphan",
    )


# ============================================================
# DocumentPart — sections, chapters, etc.
# ============================================================


class DocumentPart(Base):
    __tablename__ = "document_parts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_part_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("document_parts.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str | None] = mapped_column(String, nullable=True)  # "abstract", "methods", etc.
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    document: Mapped[Document] = relationship("Document", back_populates="parts")
    parent: Mapped["DocumentPart | None"] = relationship(
        "DocumentPart", remote_side=[id], back_populates="children"
    )
    children: Mapped[list["DocumentPart"]] = relationship("DocumentPart", back_populates="parent")


# ============================================================
# Citation — Phase 4 territory, scaffolded now.
# ============================================================


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    source_document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    target_document_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    raw_citation_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_doi: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    target_title: Mapped[str | None] = mapped_column(String, nullable=True)
    target_authors: Mapped[str | None] = mapped_column(String, nullable=True)
    target_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_method: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    source_document: Mapped[Document] = relationship(
        "Document", foreign_keys=[source_document_id], back_populates="citations_out"
    )
    target_document: Mapped["Document | None"] = relationship(
        "Document", foreign_keys=[target_document_id], back_populates="citations_in"
    )

    __table_args__ = (
        Index("idx_citations_source", "source_document_id"),
        Index("idx_citations_target", "target_document_id"),
        Index("idx_citations_target_doi", "target_doi"),
    )


# ============================================================
# DocSimilarity — Phase 4 close-out (sidecar table).
# ============================================================


class DocSimilarity(Base):
    """A directed semantic-similarity edge between two documents.

    Populated by `doc_vectors.py` from mean-pooled chunk embeddings.
    Directed by convention: `(source, target, score)` means "target is in
    source's top-K nearest neighbours under `embedding_model`". The
    relation is symmetric mathematically, but the top-K trim makes the
    persisted edge set asymmetric.
    """

    __tablename__ = "doc_similarities"

    source_document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    target_document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    embedding_model: Mapped[str] = mapped_column(String, primary_key=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_doc_sim_source", "source_document_id", "embedding_model"),
        Index("idx_doc_sim_target", "target_document_id", "embedding_model"),
    )


# ============================================================
# Figure — Phase 6 / Feature 4b (figure detection sidecar).
# ============================================================


class Figure(Base):
    """One detected figure region — a sidecar record, never spliced.

    Populated by `figures.py` / `scripts/extract_figures.py` from PyMuPDF
    geometry (image blocks plus the drawing-bbox union), gated to figure pages by
    `regions.py`. Each row points at a cropped PNG under
    `data/figures/{doc_hash}/`; the caption text stays in the markdown
    (figures are additive, not substituting). Binary artifacts are sidecar
    by the Enrichment-Layer rule — tables are the one text-shaped exception.

    The `vlm_*` columns ship present-but-null: Feature 4c (PR 9) fills them
    with a VLM description and turns each figure into a retrievable chunk.
    A caption-only row (no detectable region) carries `bbox_* = None` and
    `image_path = None` — still a useful 4c baseline.

    `doc_hash` is denormalised onto the row so a content change (which mints
    a new `doc_hash`) makes a stale figure detectable without a join, exactly
    like the citation/table enrichment drift story.
    """

    __tablename__ = "figures"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    doc_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)

    page: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based
    # Region bbox in PDF points; all four NULL for a caption-only figure.
    bbox_x0: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y0: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y1: Mapped[float | None] = mapped_column(Float, nullable=True)

    # The `regions.py` page verdict: "chart" | "photo" | "figure".
    kind: Mapped[str] = mapped_column(String, nullable=False)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # "image_block" | "drawing_union" | "largest_block" | "caption_only".
    extraction_method: Mapped[str | None] = mapped_column(String, nullable=True)

    # Feature 4c (VLM) populates these; present-but-null after 4b.
    vlm_description: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    vlm_call_skipped_reason: Mapped[str | None] = mapped_column(
        String, nullable=True, default=None
    )

    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    document: Mapped[Document] = relationship("Document", back_populates="figures")

    __table_args__ = (
        Index("idx_figures_document", "document_id"),
        Index("idx_figures_doc_hash", "doc_hash"),
    )


# ============================================================
# ChunkEpistemics — Phase 7 / Feature 7d (knowledge-currency sidecar).
# ============================================================


class ChunkEpistemics(Base):
    """Per-chunk projected epistemic weights (Feature 7d) — a regenerable sidecar.

    Written by `epistemics.py` / `scripts/compute_epistemics.py` by projecting the
    concept graph's node corroboration weights onto the chunks whose text mentions
    each concept (structural attribution, never an LLM judgement). Keyed by the stable
    composite `(document_id, chunk_index)` — Chroma's own ids are auto-generated UUIDs,
    unstable across re-ingest. The whole table is replaced on each `compute_epistemics`
    run (regenerable, dropped + rebuilt with the graph); it is never part of retrieval
    and never mutates the chunk store. A chunk with no weighted claim gets no row.

    `coverage_summary` is JSON `{"corroborated": n, "unique": n, "contested": n}`
    (JSON-as-text, like `AnswerRecord.retrieved_chunks_json`). The unique-source rule
    lives upstream in `compute_node_weights`: a sole-source claim is `unique`, never
    contested, so its chunk is never marked.
    """

    __tablename__ = "chunk_epistemics"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    n_claims: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_contested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_superseded_trend: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # JSON: {"corroborated": int, "unique": int, "contested": int}.
    coverage_summary: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # Structural fingerprint of the graph this projection came from (staleness check).
    graph_version: Mapped[str | None] = mapped_column(String, nullable=True)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_chunk_epistemics_document", "document_id"),
        Index("idx_chunk_epistemics_chunk", "document_id", "chunk_index"),
    )


# ============================================================
# Concept graph — REDESIGN (Phase 7 / Feature 7, curated-vocabulary skeleton)
# ============================================================
# The curated-vocabulary + deterministic-skeleton redesign of Feature 7
# (docs/archive/concept-graph-redesign.md; supersedes the open-vocabulary PR-16 core,
# KNOWN_ISSUES KI-7). Two lifecycles are kept deliberately distinct:
#   * Concept / ConceptAlias  — CURATED user data; survive a skeleton rebuild.
#   * ConceptEdge / ConceptPresenceRow — DERIVED sidecar rows; dropped + rebuilt on
#     every `build_concept_skeleton` run (Enrichment-Layer Pattern: regenerable,
#     never mutates the chunk store).
# Producer: `concept_skeleton.py` / `scripts/build_concept_skeleton.py` (Node A is
# deterministic + free; the LLM stance pass, Node B, is deferred).


class Concept(Base):
    """A user-curated concept node — the vocabulary the skeleton is built over.

    CURATED, not derived: the LLM never defines or extends this vocabulary
    (redesign Decision 1). Seeded as *candidates* from `Keyword` rows and promoted
    by the user (`scripts/seed_concepts.py`); survives a skeleton rebuild. `folder_id`
    ships present-but-null for the future projects-as-folders scoping (Decision 9) —
    the first increment builds global (folder-agnostic) presence.
    """

    __tablename__ = "concepts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    label: Mapped[str] = mapped_column(String, nullable=False, index=True)
    folder_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("folders.id", ondelete="SET NULL"), nullable=True
    )
    # "keyword" (promoted from a Keyword candidate) | "manual".
    source: Mapped[str] = mapped_column(String, nullable=False, default="manual")
    # Curated glossary gloss — a short definition of the concept. Optional; feeds the
    # semantic-distance layer (embed the definition, richer than the bare label).
    definition: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    aliases: Mapped[list["ConceptAlias"]] = relationship(
        "ConceptAlias", back_populates="concept", cascade="all, delete-orphan"
    )


class ConceptAlias(Base):
    """A surface form (synonym / abbreviation) for a curated `Concept`.

    CURATED (Decision 1/2): alias coverage is what bounds deterministic presence
    recall (RG-009). Unique per `(concept_id, alias)`; survives a rebuild.
    """

    __tablename__ = "concept_aliases"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    concept_id: Mapped[str] = mapped_column(
        String, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    alias: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    concept: Mapped[Concept] = relationship("Concept", back_populates="aliases")

    __table_args__ = (UniqueConstraint("concept_id", "alias", name="uq_concept_alias"),)


class ConceptEdge(Base):
    """A derived skeleton edge between two curated concepts — a regenerable sidecar.

    Dropped + rebuilt on every `build_concept_skeleton` run (Enrichment-Layer
    Pattern). `provenance_json` is the JSON provenance set ⊆ {cooccurrence, citation,
    similarity, llm_relation}; the edge is KEPT and ranked by `weight`, never dropped
    for lacking an LLM stance (Decision 5). `strength_json` (R4) is the graded per-token
    provenance strength `{token: ratio}` (citation/similarity only; null when no doc-pair
    token applies). `relation` / `stance_json` are the deferred Node-B LLM annotation —
    null after the deterministic Node-A build.
    """

    __tablename__ = "concept_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    source_concept_id: Mapped[str] = mapped_column(
        String, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    target_concept_id: Mapped[str] = mapped_column(
        String, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )
    # JSON list ⊆ ("cooccurrence", "citation", "similarity", "llm_relation").
    provenance_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # R4: JSON {token: strength ratio} for graded doc-pair provenance; null when none.
    strength_json: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    n_cooccurrence_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Node B (deferred): the LLM relation verb + per-document stance.
    relation: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    # JSON list of [document_id, polarity]; null until Node B annotates.
    stance_json: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    graph_version: Mapped[str | None] = mapped_column(String, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_concept_edges_source", "source_concept_id"),
        Index("idx_concept_edges_target", "target_concept_id"),
    )


class ConceptPresenceRow(Base):
    """A derived concept-presence record — which chunks a concept's surface forms hit
    in one document. Regenerable sidecar (dropped + rebuilt each run).

    `chunk_keys_json` is the JSON list of composite chunk keys
    `"{document_id}:p{parent_index}"` (ADR-4) the concept matched in this document.
    """

    __tablename__ = "concept_presence"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    concept_id: Mapped[str] = mapped_column(
        String, ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # JSON list of "{document_id}:p{parent_index}" chunk keys (ADR-4).
    chunk_keys_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    n_mentions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    graph_version: Mapped[str | None] = mapped_column(String, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_concept_presence_concept", "concept_id"),
        Index("idx_concept_presence_document", "document_id"),
    )


# ============================================================
# GapRow — Phase 7, gap-detection layer (deterministic Tier 1 + Tier-2a floor).
# ============================================================
# `docs/decisions/ADR-004-gap-detection-layer.md` + `docs/specs/feature-gap-detection.md`.
# Deterministic rows (tier="t1" / "t2a" with determinism="deterministic") are a
# regenerable sidecar — dropped + rebuilt on every `build_gaps` run, same as
# `ConceptEdge`/`ConceptPresenceRow` (Enrichment-Layer Pattern). Stochastic rows
# (the deferred Tier-2a ceiling / `gap_suggest.py`, out of this sprint's scope)
# persist their `status` across a rebuild — the "compounding arrow" — so the
# rebuild path must delete/replace only `determinism="deterministic"` rows.


class GapRow(Base):
    """One detected (or suggested) corpus gap — see `gaps.Gap` for the pure shape.

    `concept_id` is not a foreign key: for a deterministic gap it is a curated
    `Concept.id`, but a stochastic suggestion's `concept_id` may be a candidate
    label that doesn't exist as a `Concept` yet (that's the point — it's a
    promotion candidate). `evidence_json` holds the deterministic graph-fact ids
    (edge/doc ids, or the contributing `answer_claims` ids) or, for a stochastic
    row, the LLM inputs it was produced from (observability).
    """

    __tablename__ = "gaps"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    concept_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    tier: Mapped[str] = mapped_column(String, nullable=False)  # t1 | t2a | t2b
    determinism: Mapped[str] = mapped_column(String, nullable=False)  # deterministic | stochastic
    kind: Mapped[str] = mapped_column(String, nullable=False)
    # JSON list of fact ids (deterministic) or LLM inputs (stochastic).
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    # surfaced | promoted | dismissed — the curation lifecycle (compounding arrow).
    status: Mapped[str] = mapped_column(String, nullable=False, default="surfaced")
    graph_version: Mapped[str | None] = mapped_column(String, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    __table_args__ = (
        Index("idx_gaps_concept", "concept_id"),
        Index("idx_gaps_determinism", "determinism"),
    )


# ============================================================
# AnswerRecord — Phase 5 / Integrity Chunk 1 (provenance card).
# ============================================================


class AnswerRecord(Base):
    """One record per generated answer — the provenance card backing store.

    Every chat turn that produces an answer writes one row here. Captures
    everything needed to reproduce or audit the answer: the query, the
    retrieved chunks (with scores), the model + prompt config that
    produced the answer, token cost, latency.

    Designed forward-compat for:
    * **Multi-user** — `id` and `session_id` are UUIDs, never auto-increments.
    * **Future threading / sessions** — `session_id` is nullable now; the
      column exists so per-session aggregates don't need a schema migration.
    * **Cost tiering** — `model_name` + `token_input` / `token_output` make
      cost-by-model queries trivial. Phase 6+ reviewer agent can re-target
      the same schema for its own records.
    * **Eval harness reuse** — the eval harness will eventually consume this
      schema; same shape of (query, retrieved_chunks, answer, scores).
    """

    __tablename__ = "answer_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    # The question, post-rewrite (what the pipeline actually retrieved on).
    query: Mapped[str] = mapped_column(Text, nullable=False)
    # The raw question if it was rewritten from history; else None.
    original_query: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The generated answer text, streamed to completion.
    answer: Mapped[str] = mapped_column(Text, nullable=False)

    # JSON: list of {filename, doc_id, page, section, reranker_score, chunk_excerpt}.
    # JSON-as-text avoids a new column type and keeps the schema portable.
    retrieved_chunks_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # Configuration that produced this answer (forward-compat for cost analysis + diffing).
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    use_parent_child: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Cost + latency telemetry.
    token_input: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_output: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Failure mode capture.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


# ============================================================
# AnswerReview — Phase 6 / Integrity Chunk 2b (reviewer agent).
# ============================================================


class AnswerReview(Base):
    """One review of an AnswerRecord — typically by an LLM judge.

    One-to-many with AnswerRecord: an answer may be reviewed multiple
    times (different reviewers, different models, manual re-review).
    Schema is reviewer-kind-agnostic so a future human review path
    can reuse the same row shape.
    """

    __tablename__ = "answer_reviews"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    answer_record_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("answer_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Identifies the review path. Examples: "llm_haiku", "llm_sonnet",
    # "human", "heuristic". Lets future reviewers coexist.
    reviewer_kind: Mapped[str] = mapped_column(String, nullable=False)
    # The specific model id (None for human or heuristic reviews).
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)

    # Rubric — 1-5 integers, nullable so partial reviews (e.g., the
    # parse failed on one dimension) don't lose the dimensions that did succeed.
    faithfulness: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citation_density: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hedging_adequacy: Mapped[int | None] = mapped_column(Integer, nullable=True)
    unsupported_claims_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Chunk 2c — a categorical failure tag from a fixed enum (reviewer.FAILURE_TAGS)
    # alongside the free-text `notes`. The enum is what makes patterns *countable*
    # for the self-improvement loop; "none" / NULL means no dominant fault.
    failure_tag: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


# ============================================================
# AnswerClaim — Phase 6 / Integrity Chunk 2a (dual interpretation).
# ============================================================


class AnswerClaim(Base):
    """One adjudicable claim segmented from an ``ai``-mode interpretation answer.

    Chunk 2a splits the AI interpretation into citation-anchored claims; each
    becomes one row here, eager-inserted as ``pending`` when the answer is
    produced and updated as the user accepts / rejects / edits it. Chunk 3
    (PRISMA-trAIce) reads this as the human-AI adjudication log.

    One-to-many with AnswerRecord (an answer has N claims).
    """

    __tablename__ = "answer_claims"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    answer_record_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("answer_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Order of the claim within the answer (0-based).
    claim_index: Mapped[int] = mapped_column(Integer, nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON list of {source_number, filename, page} the claim cites.
    # Empty list = uncited => an "unsupported" claim.
    citations_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # Retrieval-derived uncertainty marker: "ok" | "weak" | "unsupported".
    marker: Mapped[str] = mapped_column(String, nullable=False, default="ok")

    # Adjudication — "pending" until the user acts; then accepted | rejected | edited.
    decision: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    edited_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


# ============================================================
# IngestionEvent — health audit trail
# ============================================================


class IngestionEvent(Base):
    __tablename__ = "ingestion_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    event_type: Mapped[str] = mapped_column(String)
    extractor: Mapped[str | None] = mapped_column(String, nullable=True)
    chunks_produced: Mapped[int | None] = mapped_column(Integer, nullable=True)
    health_status: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    document: Mapped[Document] = relationship("Document", back_populates="ingestion_events")


class ConversationMeta(Base):
    """Per-conversation management state (pin / archive / soft-delete), keyed by ``session_id``.

    Conversations are *derived* by grouping ``AnswerRecord`` rows (there is no conversation
    entity), so this sidecar holds the small mutable state a user sets on a whole conversation.
    A row exists only once an action has been taken; an **absent** row means the defaults (not
    pinned, not archived, not deleted). Additive — ``create_all`` makes the table.

    **Soft delete:** ``deleted_at`` non-null hides the conversation from the list but retains its
    ``AnswerRecord`` provenance (reversible; a permanent purge is a later, separate action).
    """

    __tablename__ = "conversation_meta"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # A user-set title; when null the list/detail fall back to the derived first-question title.
    title_override: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
