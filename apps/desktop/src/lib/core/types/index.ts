// Wire-contract barrel — re-exports every domain module so `from '../core/types'`
// keeps resolving. Prefer importing the domain module directly when you know it
// (`from '../core/types/folders'`): the import line then names the domain, and a wire
// change shows up as a one-file diff against apps/api/models/<domain>.py.
//
// Domains mirror apps/api/models/ exactly. See docs/architecture.md,
// section "apps/ — the domain spine".

export type { ClaimView, Decision, RagOverrides, SourceEpistemics, SourceEvalSummary, SourceView, TurnResult, TurnScope, UsageView } from './chat'
export type { Health } from './health'
export type { ConversationDetail, ConversationSource, ConversationSummary, ConversationTurn } from './conversations'
export type { ProviderOption, Settings } from './settings'
export type { ApiKeyResult, ProviderReadiness, SetupState, SetupStep } from './setup'
export type { IngestStatus, SourceFile } from './sources'
export type { LibraryChild, LibraryDocument, LibraryDocumentChunks, LibraryDocumentFigures, LibraryFigure, LibraryParent } from './library'
export type { LibraryFolder } from './folders'
export type { KeywordFamily, KeywordFamilyProposal } from './keywords'
export type { CitedByDoc, DocConnections, RelatedDoc } from './connections'
export type { DocReferences, DocumentReference } from './references'
export type { CompareEff, CompareResult, CompareRow, CompareSource } from './compare'
export type { ConceptCommunity, ConceptGraph, ConceptGraphEdge, ConceptGraphNode, ConceptPresence, Gap, GapKind, GapListItem, GapStatus, GraphRebuildStatus, GraphStaleness } from './concepts'
export type { FieldDetail, FieldMember, HierarchyEdgeRequest, LabelledOption, TaxonomyField, TaxonomyView } from './taxonomy'
export type { UpdateState, UpdateStatus } from './updates'
