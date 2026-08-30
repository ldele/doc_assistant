// API-client barrel — re-exports every domain client so `from '../core/api'` keeps
// resolving. Prefer the domain module when you know it (`from '../core/api/folders'`):
// the import line then names the domain, and a route change is a one-file diff against
// apps/api/routers/<domain>.py.

export { streamChat, adjudicate, figureUrl } from './chat'
export type { SSEvent } from './chat'
export { getHealth } from './health'
export { listConversations, getConversation, updateConversationMeta, exportConversation, bulkUpdateConversations, exportAllConversations } from './conversations'
export { deleteDocument, getDocumentFigures, getDocumentReferences, getLibraryDocument, getReingestOptions, getReingestStatus, listLibraryDocuments, resetDocumentMeta, revealDocument, startReingest, updateDocumentMeta } from './library'
export type { DeleteResult } from './library'
export { getDocConnections } from './connections'
export { addDocuments, indexPaths, inspectDocuments, undoAddDocuments } from './documents'
export type { AddMode, AddOutcome, AddResult } from './documents'
export { listFolders, createFolder, renameFolder, deleteFolder, addDocumentsToFolder, removeDocumentFromFolder } from './folders'
export { listKeywordFamilies, createKeywordFamily, renameKeywordFamily, setFamilyGraphInclude, addFamilyMember, removeFamilyMember, deleteKeywordFamily, detectKeywordFamilies } from './keywords'
export { compareRetrieval } from './compare'
export {
  getSettings,
  setSourceDir,
  setLlmProvider,
  setMarkersEnabled,
  reindexKeywords,
} from './settings'
export { getSetup, saveAnthropicKey, clearAnthropicKey } from './setup'
export { startIngest, getSources, patchSource, getIngestStatus } from './sources'
export { getConceptGraph, getGapList, triageGap, getConceptPresence, rebuildConceptGraph, getGraphRebuildStatus } from './concepts'
export { getTaxonomy, getFieldDetail, addHierarchyEdge, removeHierarchyEdge, attachDocumentField, detachDocumentField } from './taxonomy'
export { getUpdateStatus, checkForUpdate, setAutoUpdateCheck } from './updates'
