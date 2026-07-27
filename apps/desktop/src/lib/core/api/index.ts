// API-client barrel — re-exports every domain client so `from '../core/api'` keeps
// resolving. Prefer the domain module when you know it (`from '../core/api/folders'`):
// the import line then names the domain, and a route change is a one-file diff against
// apps/api/routers/<domain>.py.

export { streamChat, adjudicate, figureUrl } from './chat'
export type { SSEvent } from './chat'
export { getHealth } from './health'
export { listConversations, getConversation, updateConversationMeta, exportConversation } from './conversations'
export { listLibraryDocuments, getLibraryDocument, updateDocumentMeta, resetDocumentMeta, revealDocument, deleteDocument } from './library'
export type { DeleteResult } from './library'
export { getDocConnections } from './connections'
export { listFolders, createFolder, renameFolder, deleteFolder, addDocumentsToFolder, removeDocumentFromFolder } from './folders'
export { listKeywordFamilies, createKeywordFamily, renameKeywordFamily, addFamilyMember, removeFamilyMember, deleteKeywordFamily, detectKeywordFamilies } from './keywords'
export { compareRetrieval } from './compare'
export { getSettings, setSourceDir, setLlmProvider, setMarkersEnabled } from './settings'
export { startIngest, getSources, patchSource, getIngestStatus } from './sources'
export { getConceptGraph, getGapList, triageGap, getConceptPresence, rebuildConceptGraph, getGraphRebuildStatus } from './concepts'
export { getTaxonomy, getFieldDetail, addHierarchyEdge, removeHierarchyEdge, attachDocumentField, detachDocumentField } from './taxonomy'
