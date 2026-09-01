"""Data access layer for the document library.

A stable Python API over the SQLite store. The UI calls into this package rather than touching
SQLAlchemy directly, so swapping the UI or the storage backend doesn't require coordinated changes.
All functions return plain dataclasses, not SQLAlchemy models, which keeps callers free of session
lifecycle concerns.

**Layout.** One module per sub-domain, named to match `apps/api/routers/library/` and
`apps/api/models/` where they overlap:

``models`` (the returned dataclasses) · ``documents`` (queries + ADR-013 overrides + ADR-014
delete) · ``pins`` (demo-corpus removal) · ``folders`` (ADR-025 F1) · ``keywords`` (tag families,
ADR-015) · ``chunks`` (the L1 browser) · ``figures`` (L1b, per-document figures) · ``citations`` ·
``similarity``.

Prefer importing the sub-module when you know it (``from doc_assistant.library.folders import
create_folder``) — the import line then names the sub-domain. The flat re-export below keeps
``from doc_assistant.library import X`` working for the existing call sites.

**Monkeypatching:** patch a helper on the module that *owns* it
(``doc_assistant.library.documents._reveal_in_file_manager``), not on this package — a name
re-exported here is a separate binding, so patching it here will not affect the real caller.
"""

from __future__ import annotations

from doc_assistant.library.chunks import (
    ChunkChild,
    ChunkContext,
    DocumentChunkView,
    ParentBlock,
    get_chunk_context,
    get_document_chunks,
    group_children,
)
from doc_assistant.library.citations import (
    CitationEdge,
    CitationGraph,
    DocumentReference,
    DocumentReferences,
    GraphEdge,
    GraphNode,
    _row_to_edge,
    cited_by,
    cites_out,
    document_references,
    graph_subgraph,
)
from doc_assistant.library.documents import (
    DeleteResult,
    DocumentPrefixError,
    DocumentRef,
    _dedup_override,
    _reveal_in_file_manager,
    clear_document_meta,
    count_documents,
    delete_document,
    document_years,
    get_document_details,
    list_documents,
    resolve_document_prefix,
    resolve_source_path,
    reveal_document_source,
    set_document_meta,
)
from doc_assistant.library.figures import (
    DocumentFigureView,
    FigureView,
    list_document_figures,
)
from doc_assistant.library.folders import (
    FolderSummary,
    _build_folder,
    _edit_membership,
    _find_by_name,
    _folder_doc_count,
    add_documents_to_folder,
    create_folder,
    delete_folder,
    folder_doc_hashes,
    folder_document_ids,
    get_folder,
    list_folders,
    remove_documents_from_folder,
    rename_folder,
)
from doc_assistant.library.keywords import (
    KeywordFamily,
    KeywordFamilyExists,
    _all_keyword_names,
    _build_family,
    _family_doc_count,
    add_family_member,
    create_keyword_family,
    delete_keyword_family,
    detect_family_candidates,
    get_keyword_family,
    list_keyword_families,
    remove_family_member,
    rename_keyword_family,
    set_family_graph_include,
)
from doc_assistant.library.models import (
    DocumentDetails,
    DocumentSummary,
    LibrarySummary,
)
from doc_assistant.library.pins import (
    SourceMatch,
    SourcePin,
    SourceRemoval,
    _file_sha256,
    find_document_by_short_id,
    library_summary,
    match_pinned_sources,
    remove_pinned_sources,
)
from doc_assistant.library.similarity import (
    CitedByDoc,
    DocConnections,
    SimilarDoc,
    document_connections,
    similar_docs,
)
from doc_assistant.library.source_view import (
    ChunkLocation,
    PageUnavailable,
    SourceDocumentView,
    clamp_dpi,
    get_source_view,
    locate_chunk,
    page_for_chunk,
    page_for_offset,
    render_page,
)

__all__ = [
    "ChunkChild",
    "ChunkContext",
    "ChunkLocation",
    "CitationEdge",
    "CitationGraph",
    "CitedByDoc",
    "DeleteResult",
    "DocConnections",
    "DocumentChunkView",
    "DocumentDetails",
    "DocumentFigureView",
    "DocumentPrefixError",
    "DocumentRef",
    "DocumentReference",
    "DocumentReferences",
    "DocumentSummary",
    "FigureView",
    "FolderSummary",
    "GraphEdge",
    "GraphNode",
    "KeywordFamily",
    "KeywordFamilyExists",
    "LibrarySummary",
    "PageUnavailable",
    "ParentBlock",
    "SimilarDoc",
    "SourceDocumentView",
    "SourceMatch",
    "SourcePin",
    "SourceRemoval",
    "_all_keyword_names",
    "_build_family",
    "_build_folder",
    "_dedup_override",
    "_edit_membership",
    "_family_doc_count",
    "_file_sha256",
    "_find_by_name",
    "_folder_doc_count",
    "_reveal_in_file_manager",
    "_row_to_edge",
    "add_documents_to_folder",
    "add_family_member",
    "cited_by",
    "cites_out",
    "clamp_dpi",
    "clear_document_meta",
    "count_documents",
    "create_folder",
    "create_keyword_family",
    "delete_document",
    "delete_folder",
    "delete_keyword_family",
    "detect_family_candidates",
    "document_connections",
    "document_references",
    "document_years",
    "find_document_by_short_id",
    "folder_doc_hashes",
    "folder_document_ids",
    "get_chunk_context",
    "get_document_chunks",
    "get_document_details",
    "get_folder",
    "get_keyword_family",
    "get_source_view",
    "graph_subgraph",
    "group_children",
    "library_summary",
    "list_document_figures",
    "list_documents",
    "list_folders",
    "list_keyword_families",
    "locate_chunk",
    "match_pinned_sources",
    "page_for_chunk",
    "page_for_offset",
    "remove_documents_from_folder",
    "remove_family_member",
    "remove_pinned_sources",
    "rename_folder",
    "rename_keyword_family",
    "render_page",
    "resolve_document_prefix",
    "resolve_source_path",
    "reveal_document_source",
    "set_document_meta",
    "set_family_graph_include",
    "similar_docs",
]
