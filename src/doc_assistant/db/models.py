"""SQLAlchemy models for the doc-assistant library.

Design principles:
- SQLite is the source of truth for document-level metadata.
- Chroma is the source of truth for chunk embeddings.
- Both reference each other via document.id (stable UUID).
- Schema supports Phase 4 (citations) and beyond; unused fields stay NULL.
"""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column, ForeignKey, String, Integer, Text, DateTime, Boolean, Float,
    Table, UniqueConstraint, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ============================================================
# Association tables (many-to-many)
# ============================================================

document_folders = Table(
    "document_folders",
    Base.metadata,
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("folder_id", String, ForeignKey("folders.id", ondelete="CASCADE"), primary_key=True),
)

document_tags = Table(
    "document_tags",
    Base.metadata,
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", String, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)

document_keywords = Table(
    "document_keywords",
    Base.metadata,
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    parent: Mapped["Folder | None"] = relationship(
        "Folder", remote_side=[id], back_populates="children"
    )
    children: Mapped[list["Folder"]] = relationship("Folder", back_populates="parent")
    documents: Mapped[list["Document"]] = relationship(
        "Document", secondary=document_folders, back_populates="folders"
    )

    __table_args__ = (
        UniqueConstraint("name", "parent_folder_id", name="uq_folder_name_parent"),
    )


# ============================================================
# Tag — user-applied organizational labels
# ============================================================

class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    color: Mapped[str | None] = mapped_column(String, nullable=True)  # for UI
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
    source: Mapped[str | None] = mapped_column(String, nullable=True)  # "author", "extracted", "manual"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
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
        "DocumentPart", back_populates="document",
        cascade="all, delete-orphan", order_by="DocumentPart.order_index"
    )
    citations_out: Mapped[list["Citation"]] = relationship(
        "Citation", foreign_keys="[Citation.source_document_id]",
        back_populates="source_document", cascade="all, delete-orphan"
    )
    citations_in: Mapped[list["Citation"]] = relationship(
        "Citation", foreign_keys="[Citation.target_document_id]",
        back_populates="target_document"
    )
    ingestion_events: Mapped[list["IngestionEvent"]] = relationship(
        "IngestionEvent", back_populates="document",
        cascade="all, delete-orphan", order_by="IngestionEvent.timestamp.desc()"
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
    children: Mapped[list["DocumentPart"]] = relationship(
        "DocumentPart", back_populates="parent"
    )


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
# IngestionEvent — health audit trail
# ============================================================

class IngestionEvent(Base):
    __tablename__ = "ingestion_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    event_type: Mapped[str] = mapped_column(String)
    extractor: Mapped[str | None] = mapped_column(String, nullable=True)
    chunks_produced: Mapped[int | None] = mapped_column(Integer, nullable=True)
    health_status: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    document: Mapped[Document] = relationship("Document", back_populates="ingestion_events")