"""Pydantic request/response schemas for the desktop API (PR-M2).

Mirror the PR-M0/M1 ``ChatController`` value objects so the frontend renders native JSON
(the pre-rendered markdown blocks ride along as strings — a convenience/fallback, not the
only representation). The ``from_*`` constructors convert the dataclasses → payloads with
the one coercion the dataclasses need: ``Path`` → ``str`` for ``download_path``.

The dataclass types are imported under ``TYPE_CHECKING`` only, so importing this package
does not pull the heavy ``chat_controller`` → ``pipeline`` → torch chain.

**Layout.** One module per domain, named to match ``apps/api/routers/`` so a wire change and
its route are the same word on both sides:

``chat`` · ``compare`` · ``conversations`` · ``library`` · ``connections`` · ``references`` ·
``folders`` · ``keywords`` · ``sources`` · ``concepts`` · ``taxonomy`` · ``settings``
(+ ``_common``).

Prefer importing from the domain module (``from apps.api.models.folders import FolderCreate``)
— the import line then says which domain the route belongs to. The flat re-export below is kept
so ``from apps.api.models import X`` still resolves for anything not yet updated.
"""

from __future__ import annotations

from apps.api.models.chat import (
    AdjudicateRequest,
    ChatRequest,
    ClaimViewPayload,
    ExportRequest,
    RagOverrides,
    ScopePayload,
    SourceEpistemicsPayload,
    SourceEvalSummaryPayload,
    SourceViewPayload,
    TurnResultPayload,
    UsageViewPayload,
)
from apps.api.models.compare import (
    CompareEffPayload,
    CompareRequest,
    CompareResultPayload,
    CompareRowPayload,
    CompareSourcePayload,
)
from apps.api.models.concepts import (
    ConceptCommunityPayload,
    ConceptGraphEdgePayload,
    ConceptGraphNodePayload,
    ConceptGraphPayload,
    ConceptPresencePayload,
    GapListItemPayload,
    GapPayload,
    GapTriageRequest,
    GraphStalenessPayload,
)
from apps.api.models.connections import (
    CitedByPayload,
    DocConnectionsPayload,
    RelatedDocPayload,
)
from apps.api.models.conversations import (
    ConversationBulkResult,
    ConversationBulkUpdate,
    ConversationDetailPayload,
    ConversationMetaUpdate,
    ConversationSourcePayload,
    ConversationSummaryPayload,
    ConversationTurnPayload,
)
from apps.api.models.folders import (
    FolderCreate,
    FolderMembers,
    FolderRename,
    LibraryFolderPayload,
)
from apps.api.models.keywords import (
    KeywordFamilyCreate,
    KeywordFamilyMember,
    KeywordFamilyPatch,
    KeywordFamilyPayload,
    KeywordFamilyProposalPayload,
)
from apps.api.models.library import (
    DeleteResultPayload,
    LibraryChildPayload,
    LibraryDocumentChunksPayload,
    LibraryDocumentMetaUpdate,
    LibraryDocumentPayload,
    LibraryParentPayload,
    ReingestOptionsPayload,
    ReingestOutcomePayload,
    ReingestPartPayload,
    ReingestRequest,
)
from apps.api.models.references import DocReferencesPayload, ReferencePayload
from apps.api.models.settings import SettingsUpdate
from apps.api.models.sources import (
    FileVerdictPayload,
    IngestRequest,
    InspectRequest,
    InspectResponse,
    SourceFilePayload,
    SourcePatch,
)
from apps.api.models.taxonomy import (
    FieldDetailPayload,
    FieldMemberPayload,
    HierarchyEdgeRequest,
    TaxonomyFieldPayload,
    TaxonomyViewPayload,
)

__all__ = [
    "AdjudicateRequest",
    "ChatRequest",
    "CitedByPayload",
    "ClaimViewPayload",
    "CompareEffPayload",
    "CompareRequest",
    "CompareResultPayload",
    "CompareRowPayload",
    "CompareSourcePayload",
    "ConceptCommunityPayload",
    "ConceptGraphEdgePayload",
    "ConceptGraphNodePayload",
    "ConceptGraphPayload",
    "ConceptPresencePayload",
    "ConversationBulkResult",
    "ConversationBulkUpdate",
    "ConversationDetailPayload",
    "ConversationMetaUpdate",
    "ConversationSourcePayload",
    "ConversationSummaryPayload",
    "ConversationTurnPayload",
    "DeleteResultPayload",
    "DocConnectionsPayload",
    "DocReferencesPayload",
    "ExportRequest",
    "FieldDetailPayload",
    "FieldMemberPayload",
    "FileVerdictPayload",
    "FolderCreate",
    "FolderMembers",
    "FolderRename",
    "GapListItemPayload",
    "GapPayload",
    "GapTriageRequest",
    "GraphStalenessPayload",
    "HierarchyEdgeRequest",
    "IngestRequest",
    "InspectRequest",
    "InspectResponse",
    "KeywordFamilyCreate",
    "KeywordFamilyMember",
    "KeywordFamilyPatch",
    "KeywordFamilyPayload",
    "KeywordFamilyProposalPayload",
    "LibraryChildPayload",
    "LibraryDocumentChunksPayload",
    "LibraryDocumentMetaUpdate",
    "LibraryDocumentPayload",
    "LibraryFolderPayload",
    "LibraryParentPayload",
    "RagOverrides",
    "ReferencePayload",
    "ReingestOptionsPayload",
    "ReingestOutcomePayload",
    "ReingestPartPayload",
    "ReingestRequest",
    "RelatedDocPayload",
    "ScopePayload",
    "SettingsUpdate",
    "SourceEpistemicsPayload",
    "SourceEvalSummaryPayload",
    "SourceFilePayload",
    "SourcePatch",
    "SourceViewPayload",
    "TaxonomyFieldPayload",
    "TaxonomyViewPayload",
    "TurnResultPayload",
    "UsageViewPayload",
]
