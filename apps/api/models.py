"""Pydantic request/response schemas for the desktop API (PR-M2).

Mirror the PR-M0/M1 ``ChatController`` value objects so the frontend renders native JSON
(the pre-rendered markdown blocks ride along as strings — a convenience/fallback, not the
only representation). The ``from_*`` constructors convert the dataclasses → payloads with
the one coercion the dataclasses need: ``Path`` → ``str`` for ``download_path``.

The dataclass types are imported under ``TYPE_CHECKING`` only, so importing this module
does not pull the heavy ``chat_controller`` → ``pipeline`` → torch chain.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from doc_assistant.config import CANDIDATE_K

if TYPE_CHECKING:
    from doc_assistant.chat_controller import (
        ClaimView,
        SourceEpistemics,
        SourceEvalSummary,
        SourceView,
        TurnResult,
        UsageView,
    )
    from doc_assistant.compare import CompareResult, CompareRow, CompareSource
    from doc_assistant.conversations import (
        ConversationDetail,
        ConversationSummary,
        ConversationTurn,
    )
    from doc_assistant.ingest.registry import SourceView as RegistrySourceView
    from doc_assistant.knowledge.concept_graph_view import GraphStaleness, GraphView
    from doc_assistant.knowledge.concept_skeleton import (
        Community,
        ConceptNode,
        ConceptPresence,
        SkeletonEdge,
    )
    from doc_assistant.knowledge.gaps import Gap
    from doc_assistant.knowledge.keyword_families import FamilyProposal
    from doc_assistant.library import (
        DocumentChunkView,
        DocumentSummary,
        FolderSummary,
        KeywordFamily,
        ParentBlock,
    )


# ============================================================
# Requests
# ============================================================


class RagOverrides(BaseModel):
    """Wire model for a session-scoped, non-persistent RAG-sandbox override (ADR-010).
    ``None`` (a field or the whole object) = use the locked default. ``top_k`` is bounded to
    ``[1, CANDIDATE_K]`` — the candidate pool is fixed at pipeline construction, so a top_k
    above it is meaningless; out-of-range is a 422, never a silent clamp.

    ``epistemics_markers_enabled``/``reviewer_evidence_chars`` (U1b, SPRINT-011, ADR-010's
    2026-07-10 amendment) are the two "must revisit" niche knobs. ``reviewer_evidence_chars``
    is bounded ``[200, 6000]``: the floor sits above the ~300-char display excerpt that was
    empirically shown to starve the reviewer into false "unsupported claim" verdicts
    (`config.py`'s own comment on `REVIEWER_EVIDENCE_CHARS`); the ceiling is a generous 4x the
    1500-char default, bounding judge-token cost without being restrictive for experimentation.
    """

    top_k: int | None = Field(default=None, ge=1, le=CANDIDATE_K)
    synthesis_mode: Literal["ai", "human"] | None = None
    use_multi_query: bool | None = None
    epistemics_markers_enabled: bool | None = None
    reviewer_evidence_chars: int | None = Field(default=None, ge=200, le=6000)


class ChatRequest(BaseModel):
    text: str
    session_id: str
    overrides: RagOverrides | None = None
    # ADR-025 F2 — restrict retrieval to one folder for this turn. A sibling of `overrides`,
    # not a field inside it: a scope is a *content* filter (which documents), while
    # `RagOverrides` is ADR-010's governance channel for locked *quality* knobs. Only the id
    # crosses the wire — the backend resolves membership per turn, so a Library edit can never
    # be out of date by the time the answer is produced.
    scope_folder_id: str | None = None


class CompareRequest(BaseModel):
    """A/B-compare (U6): retrieve ``text`` under the locked defaults and under ``overrides``, and
    diff the two source sets. ``overrides=None`` = the session matches defaults (a no-op diff)."""

    text: str
    overrides: RagOverrides | None = None
    # ADR-025 F2 — applied to BOTH sides, so the diff isolates the knob rather than the corpus.
    scope_folder_id: str | None = None


class AdjudicateRequest(BaseModel):
    decision: Literal["accepted", "rejected", "edited"]
    edited_text: str | None = None


class ExportRequest(BaseModel):
    session_id: str
    dev: bool = False


class SettingsUpdate(BaseModel):
    """User-settable settings: the source documents folder, and (ADR-011, U1c) the LLM
    provider/model to switch to. At least one of ``source_dir`` or the ``llm_provider`` +
    ``llm_model`` pair must be present; the two ``llm_*`` fields travel together or not at all —
    both requests reject an otherwise-empty or half-shaped body with a 422, not a silent no-op."""

    source_dir: str | None = None
    llm_provider: Literal["anthropic", "ollama"] | None = None
    llm_model: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_shape(self) -> SettingsUpdate:
        if self.source_dir is None and self.llm_provider is None and self.llm_model is None:
            raise ValueError("at least one of source_dir or llm_provider+llm_model is required")
        if (self.llm_provider is None) != (self.llm_model is None):
            raise ValueError("llm_provider and llm_model must be provided together")
        return self


# ============================================================
# Response payloads (mirror the controller value objects)
# ============================================================


class SourceEpistemicsPayload(BaseModel):
    """ADR-027 D3 — one source's always-on epistemic assessment (mirrors `SourceEpistemics`)."""

    coverage: str | None  # corroborated | unique | contested | null (not assessed)
    superseded: bool
    n_claims: int
    year: int | None

    @classmethod
    def from_view(cls, ev: SourceEpistemics) -> SourceEpistemicsPayload:
        return cls(
            coverage=ev.coverage, superseded=ev.superseded, n_claims=ev.n_claims, year=ev.year
        )


class SourceViewPayload(BaseModel):
    n: int
    citation: str
    excerpt: str
    # The figure *id* (not the server path — no filesystem path crosses the boundary, M2
    # ADR-1); the frontend renders it via GET /api/figures/{figure_id}.
    figure_id: str | None
    chunk_key: str | None
    markers: list[str]
    # ADR-027 D3 — always-on per-source evaluation + the rerank score (strip signals).
    reranker_score: float = 0.0
    evaluation: SourceEpistemicsPayload | None = None

    @classmethod
    def from_view(cls, sv: SourceView) -> SourceViewPayload:
        return cls(
            n=sv.n,
            citation=sv.citation,
            excerpt=sv.excerpt,
            figure_id=sv.figure_id,
            chunk_key=sv.chunk_key,
            markers=list(sv.markers),
            reranker_score=sv.reranker_score,
            evaluation=(
                SourceEpistemicsPayload.from_view(sv.evaluation)
                if sv.evaluation is not None
                else None
            ),
        )


class SourceEvalSummaryPayload(BaseModel):
    """ADR-027 D3 — strip-level freshness (mirrors `SourceEvalSummary`)."""

    graph_version: str | None
    stale: bool

    @classmethod
    def from_view(cls, s: SourceEvalSummary) -> SourceEvalSummaryPayload:
        return cls(graph_version=s.graph_version, stale=s.stale)


class ClaimViewPayload(BaseModel):
    claim_id: str
    n: int
    text: str
    badge: str

    @classmethod
    def from_view(cls, cv: ClaimView) -> ClaimViewPayload:
        return cls(claim_id=cv.claim_id, n=cv.n, text=cv.text, badge=cv.badge)


class UsageViewPayload(BaseModel):
    turn_input: int
    turn_output: int
    session_total: int
    cost_usd: float | None
    is_local: bool

    @classmethod
    def from_view(cls, u: UsageView) -> UsageViewPayload:
        return cls(
            turn_input=u.turn_input,
            turn_output=u.turn_output,
            session_total=u.session_total,
            cost_usd=u.cost_usd,
            is_local=u.is_local,
        )


class ScopePayload(BaseModel):
    """The retrieval scope a turn ran under (ADR-025 F2); absent = the whole library.
    `folder_name` is null when the folder was deleted before the turn ran."""

    folder_id: str
    folder_name: str | None
    doc_count: int


class TurnResultPayload(BaseModel):
    answer: str
    mode: Literal["ai", "human"]
    sources: list[SourceViewPayload]
    flagged_claims: list[ClaimViewPayload]
    usage: UsageViewPayload
    standalone_query: str
    record_id: str | None
    provenance_card_md: str
    claim_review_md: str
    sources_md: str
    usage_md: str
    citation_note_md: str
    download_path: str | None
    scope: ScopePayload | None = None
    # ADR-027 D3 — strip-level freshness for the always-on source-evaluation strip (per-source
    # evaluation rides on each source). null = no epistemics sidecar / 0-doc → no strip.
    source_eval: SourceEvalSummaryPayload | None = None

    @classmethod
    def from_turn_result(cls, r: TurnResult) -> TurnResultPayload:
        return cls(
            answer=r.answer,
            mode=r.mode,
            sources=[SourceViewPayload.from_view(s) for s in r.sources],
            flagged_claims=[ClaimViewPayload.from_view(c) for c in r.flagged_claims],
            usage=UsageViewPayload.from_view(r.usage),
            standalone_query=r.standalone_query,
            record_id=r.record_id,
            provenance_card_md=r.provenance_card_md,
            claim_review_md=r.claim_review_md,
            sources_md=r.sources_md,
            usage_md=r.usage_md,
            citation_note_md=r.citation_note_md,
            download_path=str(r.download_path) if r.download_path is not None else None,
            scope=(
                ScopePayload(
                    folder_id=r.scope.folder_id,
                    folder_name=r.scope.folder_name,
                    doc_count=r.scope.doc_count,
                )
                if r.scope is not None
                else None
            ),
            source_eval=(
                SourceEvalSummaryPayload.from_view(r.source_eval)
                if r.source_eval is not None
                else None
            ),
        )


# ============================================================
# Conversation history (feature-conversation-history.md — read-only)
# ============================================================


def _as_utc(dt: datetime) -> datetime:
    """Tag a naive DB timestamp (``AnswerRecord.created_at`` is naive UTC) as UTC so the ISO wire
    value carries an offset — otherwise a browser ``new Date()`` reads it as *local* time."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class ConversationSummaryPayload(BaseModel):
    """One conversation in the sidebar list."""

    session_id: str
    title: str
    turn_count: int
    started_at: datetime
    last_at: datetime
    pinned: bool = False
    archived: bool = False

    @classmethod
    def from_summary(cls, s: ConversationSummary) -> ConversationSummaryPayload:
        return cls(
            session_id=s.session_id,
            title=s.title,
            turn_count=s.turn_count,
            started_at=_as_utc(s.started_at),
            last_at=_as_utc(s.last_at),
            pinned=s.pinned,
            archived=s.archived,
        )


class ConversationMetaUpdate(BaseModel):
    """PATCH body for a conversation's management flags — only the fields sent are changed.
    ``deleted`` toggles the soft-delete (True hides + retains records; False restores). ``title``
    sets a custom title (blank reverts to the derived first-question title)."""

    pinned: bool | None = None
    archived: bool | None = None
    deleted: bool | None = None
    title: str | None = None


class ConversationSourcePayload(BaseModel):
    n: int
    citation: str
    excerpt: str


class ConversationTurnPayload(BaseModel):
    record_id: str
    question: str
    answer: str
    sources: list[ConversationSourcePayload]
    # ADR-025 F2 — replayed from the record, so a reopened scoped answer still says it was scoped.
    scope: ScopePayload | None = None

    @classmethod
    def from_turn(cls, t: ConversationTurn) -> ConversationTurnPayload:
        return cls(
            record_id=t.record_id,
            question=t.question,
            answer=t.answer,
            sources=[
                ConversationSourcePayload(n=s.n, citation=s.citation, excerpt=s.excerpt)
                for s in t.sources
            ],
            scope=(ScopePayload(**t.retrieval_scope) if t.retrieval_scope is not None else None),
        )


class ConversationDetailPayload(BaseModel):
    """A reopened conversation — its title + ordered turns (read-only transcript)."""

    session_id: str
    title: str
    turns: list[ConversationTurnPayload]

    @classmethod
    def from_detail(cls, d: ConversationDetail) -> ConversationDetailPayload:
        return cls(
            session_id=d.session_id,
            title=d.title,
            turns=[ConversationTurnPayload.from_turn(t) for t in d.turns],
        )


# ============================================================
# Library browser (feature-library-browser.md — read-only, L1)
# ============================================================


class LibraryDocumentPayload(BaseModel):
    """One document in the Library list (mirrors ``library.DocumentSummary``).

    ``title``/``authors``/``year`` are effective values (user override ?? auto); ``customized``
    is True when a user override is in force (ADR-013)."""

    id: str
    filename: str
    title: str | None
    authors: str | None
    year: int | None
    customized: bool
    format: str
    health: str | None
    chunk_count: int | None
    page_count: int | None
    folders: list[str]
    folder_ids: list[str]
    tags: list[str]
    keywords: list[str]
    added_at: datetime | None

    @classmethod
    def from_summary(cls, s: DocumentSummary) -> LibraryDocumentPayload:
        return cls(
            id=s.id,
            filename=s.filename,
            title=s.title,
            authors=s.authors,
            year=s.year,
            customized=s.customized,
            format=s.format,
            health=s.health,
            chunk_count=s.chunk_count,
            page_count=s.page_count,
            folders=list(s.folders),
            folder_ids=list(s.folder_ids),
            tags=list(s.tags),
            keywords=list(s.keywords),
            added_at=_as_utc(s.added_at) if s.added_at is not None else None,
        )


class LibraryDocumentMetaUpdate(BaseModel):
    """PATCH body for a document's user metadata overrides (ADR-013).

    The editor sends the whole small form; each field is the desired *effective* value. A blank
    string (or a value equal to the auto-extracted default) clears that field's override."""

    title: str | None = None
    authors: str | None = None
    year: int | None = None


class DeleteResultPayload(BaseModel):
    """Outcome of a document delete (ADR-014)."""

    filename: str
    trashed_file: bool
    chunks_removed: int


class LibraryChildPayload(BaseModel):
    child_index: int
    text: str
    retrievable: bool


class LibraryParentPayload(BaseModel):
    parent_index: int
    parent_text: str
    children: list[LibraryChildPayload]

    @classmethod
    def from_block(cls, b: ParentBlock) -> LibraryParentPayload:
        return cls(
            parent_index=b.parent_index,
            parent_text=b.parent_text,
            children=[
                LibraryChildPayload(
                    child_index=c.child_index, text=c.text, retrievable=c.retrievable
                )
                for c in b.children
            ],
        )


class LibraryDocumentChunksPayload(BaseModel):
    """A document's header + its chunks grouped into parent blocks (browser detail)."""

    id: str
    filename: str
    format: str
    title: str | None
    authors: str | None
    year: int | None
    chunk_count: int | None
    health: str | None
    parents: list[LibraryParentPayload]
    child_count: int

    @classmethod
    def from_view(cls, v: DocumentChunkView) -> LibraryDocumentChunksPayload:
        return cls(
            id=v.id,
            filename=v.filename,
            format=v.format,
            title=v.title,
            authors=v.authors,
            year=v.year,
            chunk_count=v.chunk_count,
            health=v.health,
            parents=[LibraryParentPayload.from_block(b) for b in v.parents],
            child_count=v.child_count,
        )


# ============================================================
# Folders (docs/specs/feature-corpus-folders.md — ADR-025 F1)
# ============================================================


class LibraryFolderPayload(BaseModel):
    """One folder plus its non-archived member count (mirrors ``library.FolderSummary``).

    ``parent_id`` is always None in F1 — folders are flat until nesting is decided (spec D1);
    the field is on the wire so adding nesting later is not a contract break."""

    id: str
    name: str
    description: str | None
    parent_id: str | None
    doc_count: int

    @classmethod
    def from_folder(cls, f: FolderSummary) -> LibraryFolderPayload:
        return cls(
            id=f.id,
            name=f.name,
            description=f.description,
            parent_id=f.parent_id,
            doc_count=f.doc_count,
        )


class FolderCreate(BaseModel):
    """POST body to create a folder."""

    name: str = Field(min_length=1)
    description: str | None = None


class FolderRename(BaseModel):
    """PATCH body to rename a folder."""

    name: str = Field(min_length=1)


class FolderMembers(BaseModel):
    """POST body to add documents to a folder (bulk; idempotent)."""

    document_ids: list[str] = Field(min_length=1)


# ============================================================
# Tag families (feature-tag-families.md — PR-1)
# ============================================================


class KeywordFamilyPayload(BaseModel):
    """A canonical tag + its member keyword names (mirrors ``library.KeywordFamily``)."""

    id: str
    canonical: str
    aliases: list[str]
    doc_count: int

    @classmethod
    def from_family(cls, f: KeywordFamily) -> KeywordFamilyPayload:
        return cls(id=f.id, canonical=f.canonical, aliases=list(f.aliases), doc_count=f.doc_count)


class KeywordFamilyCreate(BaseModel):
    """POST body to create a family: the canonical label + initial member keywords."""

    canonical: str = Field(min_length=1)
    members: list[str] = Field(default_factory=list)


class KeywordFamilyRename(BaseModel):
    """PATCH body to rename a family's canonical label."""

    canonical: str = Field(min_length=1)


class KeywordFamilyMember(BaseModel):
    """POST body to add a member keyword to a family."""

    keyword: str = Field(min_length=1)


# ============================================================
# Tag families — detection (feature-tag-families.md — PR-2)
# ============================================================


class KeywordFamilyProposalPayload(BaseModel):
    """One deterministic, zero-LLM family proposal (mirrors ``keyword_families.FamilyProposal``).

    Nothing here has been written to the DB — accepting a proposal calls the existing family CRUD
    above (``POST``/``PATCH .../keyword-families``)."""

    canonical: str
    members: list[str]
    tier: Literal["morphological", "embedding"]
    confidence: float

    @classmethod
    def from_proposal(cls, p: FamilyProposal) -> KeywordFamilyProposalPayload:
        return cls(
            canonical=p.canonical, members=list(p.members), tier=p.tier, confidence=p.confidence
        )


# ============================================================
# A/B-compare (feature-ab-compare-sandbox.md — retrieval diff, U6)
# ============================================================


class CompareEffPayload(BaseModel):
    """The effective retrieval knobs on one side of a compare."""

    top_k: int
    use_multi_query: bool


class CompareSourcePayload(BaseModel):
    rank: int
    filename: str
    page: int | None
    section: str | None
    score: float
    excerpt: str
    citation: str
    identity: str

    @classmethod
    def from_source(cls, s: CompareSource) -> CompareSourcePayload:
        return cls(
            rank=s.rank,
            filename=s.filename,
            page=s.page,
            section=s.section,
            score=s.score,
            excerpt=s.excerpt,
            citation=s.citation,
            identity=s.identity,
        )


class CompareRowPayload(BaseModel):
    identity: str
    source_a: CompareSourcePayload | None
    source_b: CompareSourcePayload | None
    status: str  # in_both | only_in_a | only_in_b
    rank_delta: int | None

    @classmethod
    def from_row(cls, r: CompareRow) -> CompareRowPayload:
        return cls(
            identity=r.identity,
            source_a=CompareSourcePayload.from_source(r.source_a) if r.source_a else None,
            source_b=CompareSourcePayload.from_source(r.source_b) if r.source_b else None,
            status=r.status,
            rank_delta=r.rank_delta,
        )


class CompareResultPayload(BaseModel):
    """Both ranked source sets (A = defaults, B = session override) + the diff + honesty note."""

    query: str
    sources_a: list[CompareSourcePayload]
    sources_b: list[CompareSourcePayload]
    rows: list[CompareRowPayload]
    eff_a: CompareEffPayload
    eff_b: CompareEffPayload
    note: str
    # ADR-025 F2 — the folder BOTH sides were retrieved under; null = the whole library.
    scope_label: str | None = None

    @classmethod
    def from_result(cls, r: CompareResult) -> CompareResultPayload:
        return cls(
            query=r.query,
            sources_a=[CompareSourcePayload.from_source(s) for s in r.sources_a],
            sources_b=[CompareSourcePayload.from_source(s) for s in r.sources_b],
            rows=[CompareRowPayload.from_row(x) for x in r.rows],
            eff_a=CompareEffPayload(
                top_k=int(r.eff_a["top_k"]),
                use_multi_query=bool(r.eff_a["use_multi_query"]),
            ),
            eff_b=CompareEffPayload(
                top_k=int(r.eff_b["top_k"]),
                use_multi_query=bool(r.eff_b["use_multi_query"]),
            ),
            note=r.note,
            scope_label=r.scope_label,
        )


# ============================================================
# Selective ingestion (feature-selective-ingestion.md, S1)
# ============================================================


class IngestRequest(BaseModel):
    """Optional POST /api/ingest body. Absent / null ``paths`` = ingest the whole source dir
    (minus standing exclusions); a list = ingest exactly that selection (overriding exclusions)."""

    paths: list[str] | None = None


class SourcePatch(BaseModel):
    """PATCH /api/sources body. v1 sets ``excluded`` only (``doc_type`` is the dormant column)."""

    rel_path: str
    excluded: bool | None = None


class SourceFilePayload(BaseModel):
    """One selective-ingestion registry row — mirrors ``ingest.registry.SourceView``.

    Named ``SourceFile`` (not ``SourceView``) to avoid colliding with the citation-source
    ``SourceView``. ``doc_type`` is always ``null`` in v1 (the dormant column).
    """

    rel_path: str
    format: str
    size: int
    mtime: float
    status: str
    excluded: bool
    doc_type: str | None

    @classmethod
    def from_view(cls, v: RegistrySourceView) -> SourceFilePayload:
        return cls(
            rel_path=v.rel_path,
            format=v.format,
            size=v.size,
            mtime=v.mtime,
            status=v.status,
            excluded=v.excluded,
            doc_type=v.doc_type,
        )


# ============================================================
# Concept graph (PR-G1 — ADR-017 / docs/specs/feature-concept-graph.md)
# ============================================================
#
# Wire id space: concept **UUIDs** everywhere — `ConceptGraphNodePayload.id`,
# `ConceptGraphEdgePayload.source`/`target`, `GapPayload.concept_id` and
# `ConceptCommunityPayload.node_ids` are all `Concept.id`. `label` rides **only** on the node;
# the client joins by id. Mixing ids and labels across this boundary is the bug that caused
# KI-15, so the one id space is a contract, not a convention.


class ConceptGraphNodePayload(BaseModel):
    """One concept node. `degree` and `community` are precomputed layout signal."""

    id: str
    label: str
    doc_ids: list[str]
    degree: int
    community: int

    @classmethod
    def from_node(cls, n: ConceptNode) -> ConceptGraphNodePayload:
        return cls(
            id=n.id,
            label=n.label,
            doc_ids=list(n.doc_ids),
            degree=n.degree,
            community=n.community,
        )


class ConceptGraphEdgePayload(BaseModel):
    """An undirected concept-concept edge, typed by its provenance set.

    `relation`/`stance` are the deferred Node-B annotation and are empty on every edge until
    that pass runs — a renderer must not imply agreement/disagreement it does not have.
    """

    source: str
    target: str
    provenance: list[str]
    weight: float
    n_cooccurrence_chunks: int
    relation: str | None = None

    @classmethod
    def from_edge(cls, e: SkeletonEdge) -> ConceptGraphEdgePayload:
        return cls(
            source=e.source_concept_id,
            target=e.target_concept_id,
            provenance=sorted(e.provenance),
            weight=e.weight,
            n_cooccurrence_chunks=e.n_cooccurrence_chunks,
            relation=e.relation,
        )


class ConceptCommunityPayload(BaseModel):
    """A Louvain community. `id` is POSITIONAL, not identity — it renumbers when the
    vocabulary changes, so a client must never persist a preference against it."""

    id: int
    label: str
    node_ids: list[str]
    size: int

    @classmethod
    def from_community(cls, c: Community) -> ConceptCommunityPayload:
        return cls(id=c.id, label=c.label, node_ids=list(c.node_ids), size=c.size)


class GapPayload(BaseModel):
    """One detected corpus gap (ADR-004), anchored to a concept.

    `rating` is `None` for every deterministic gap (a raw graph fact carries no confidence).
    `status` is the row's own value; per ADR-017 C1 a user's triage lives in its own override
    sidecar (deterministic rows are delete-and-replace), so it is not yet resolved here.
    """

    concept_id: str
    tier: str
    determinism: str
    kind: str
    fact_ids: list[str]
    rating: float | None = None
    status: str

    @classmethod
    def from_gap(cls, g: Gap) -> GapPayload:
        return cls(
            concept_id=g.concept_id,
            tier=g.tier,
            determinism=g.determinism,
            kind=g.kind,
            fact_ids=list(g.evidence.fact_ids),
            rating=g.rating,
            status=g.status,
        )


class GraphStalenessPayload(BaseModel):
    """How far the built graph has drifted from the live vocabulary.

    The skeleton is a build artifact and the Manage-keywords view writes `Concept` rows live,
    so drift is structural, not a defect: the UI reports it and offers a rebuild.
    """

    stale: bool
    n_concepts_in_db: int
    n_concepts_in_skeleton: int
    added_labels: list[str]
    removed_ids: list[str]

    @classmethod
    def from_staleness(cls, s: GraphStaleness) -> GraphStalenessPayload:
        return cls(
            stale=s.stale,
            n_concepts_in_db=s.n_concepts_in_db,
            n_concepts_in_skeleton=s.n_concepts_in_skeleton,
            added_labels=list(s.added_labels),
            removed_ids=list(s.removed_ids),
        )


class ConceptGraphPayload(BaseModel):
    """The whole read model for one render of the concept-graph view."""

    graph_version: str
    nodes: list[ConceptGraphNodePayload]
    edges: list[ConceptGraphEdgePayload]
    communities: list[ConceptCommunityPayload]
    gaps: list[GapPayload]
    staleness: GraphStalenessPayload

    @classmethod
    def from_view(cls, v: GraphView) -> ConceptGraphPayload:
        return cls(
            graph_version=str(v.skeleton.meta.get("graph_version", "")),
            nodes=[ConceptGraphNodePayload.from_node(n) for n in v.skeleton.nodes],
            edges=[ConceptGraphEdgePayload.from_edge(e) for e in v.skeleton.edges],
            communities=[
                ConceptCommunityPayload.from_community(c) for c in v.skeleton.communities
            ],
            gaps=[GapPayload.from_gap(g) for g in v.gaps],
            staleness=GraphStalenessPayload.from_staleness(v.staleness),
        )


class ConceptPresencePayload(BaseModel):
    """Where one concept appears in one document.

    `chunk_keys` are ADR-4 composite `"{document_id}:p{parent_index}"` — the navigation
    payload that takes the ego view from a concept down to the chunks that mention it.
    """

    document_id: str
    chunk_keys: list[str]
    n_mentions: int

    @classmethod
    def from_presence(cls, p: ConceptPresence) -> ConceptPresencePayload:
        return cls(
            document_id=p.document_id,
            chunk_keys=list(p.chunk_keys),
            n_mentions=p.n_mentions,
        )
