// Thin client for the desktop API (PR-M2). No business logic — fetch + SSE parsing only.
//
// In dev, Vite proxies `/api` → 127.0.0.1:8001 (same-origin, no CORS). In the packaged
// Tauri build the frontend is served from the asset/tauri origin, so it hits the absolute
// backend URL (the API's CORS allowlist includes `tauri://localhost`).
import type {
  CompareResult,
  ConversationDetail,
  ConversationSummary,
  Decision,
  Health,
  IngestStatus,
  KeywordFamily,
  LibraryDocument,
  LibraryDocumentChunks,
  RagOverrides,
  Settings,
  SourceFile,
  TurnResult,
} from './types'

const API_BASE: string = import.meta.env.DEV ? '' : 'http://127.0.0.1:8001'

export interface SSEvent {
  event: string
  data: string
}

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${API_BASE}/api/health`)
  if (!r.ok) throw new Error(`health failed: ${r.status}`)
  return (await r.json()) as Health
}

/** List past conversations for the history sidebar (feature-conversation-history.md). */
export async function listConversations(): Promise<ConversationSummary[]> {
  const r = await fetch(`${API_BASE}/api/conversations`)
  if (!r.ok) throw new Error(`conversations failed: ${r.status}`)
  return (await r.json()) as ConversationSummary[]
}

/** Rehydrate one conversation as a read-only transcript. */
export async function getConversation(sessionId: string): Promise<ConversationDetail> {
  const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}`)
  if (!r.ok) throw new Error(`conversation failed: ${r.status}`)
  return (await r.json()) as ConversationDetail
}

/** Set a conversation's management flags (pin / archive / soft-delete). Only the fields passed
 *  change. `deleted: true` hides it (records retained); `deleted: false` restores it. */
export async function updateConversationMeta(
  sessionId: string,
  patch: { pinned?: boolean; archived?: boolean; deleted?: boolean; title?: string },
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!r.ok) throw new Error(`update conversation failed: ${r.status}`)
}

/** List ingested documents for the Library browser (feature-library-browser.md, read-only). */
export async function listLibraryDocuments(): Promise<LibraryDocument[]> {
  const r = await fetch(`${API_BASE}/api/library/documents`)
  if (!r.ok) throw new Error(`library documents failed: ${r.status}`)
  return (await r.json()) as LibraryDocument[]
}

/** One document's chunks grouped into parent blocks (each expandable to its children). 404 unknown. */
export async function getLibraryDocument(docId: string): Promise<LibraryDocumentChunks> {
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}`)
  if (!r.ok) throw new Error(`library document failed: ${r.status}`)
  return (await r.json()) as LibraryDocumentChunks
}

/** Set a document's user metadata overrides (title/authors/year). The editor sends the whole form;
 *  each effective value blank/equal-to-default clears that field's override (ADR-013). */
export async function updateDocumentMeta(
  docId: string,
  patch: { title?: string; authors?: string; year?: number | null },
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!r.ok) throw new Error(`update document metadata failed: ${r.status}`)
}

/** Reset a document's metadata to the auto-extracted defaults (drop the override). */
export async function resetDocumentMeta(docId: string): Promise<void> {
  const r = await fetch(
    `${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/reset-metadata`,
    { method: 'POST' },
  )
  if (!r.ok) throw new Error(`reset document metadata failed: ${r.status}`)
}

/** Reveal a document's source file in the OS file manager (local desktop action). 404 if the file
 *  can't be located (moved/deleted since ingest). */
export async function revealDocument(docId: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/reveal`, {
    method: 'POST',
  })
  if (!r.ok) throw new Error(`reveal document failed: ${r.status}`)
}

export interface DeleteResult {
  filename: string
  trashed_file: boolean
  chunks_removed: number
}

/** Safe-delete a document: source file → Recycle Bin, then drop its DB row + index chunks
 *  (ADR-014). Throws on failure (e.g. 409 when the file couldn't be moved to the Recycle Bin). */
export async function deleteDocument(docId: string): Promise<DeleteResult> {
  const r = await fetch(`${API_BASE}/api/library/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(`delete document failed: ${r.status}`)
  return (await r.json()) as DeleteResult
}

// Tag families (feature-tag-families.md, PR-1). A family collapses near-duplicate keywords
// (`llm`/`llms`) into one filterable entry — errors carry the backend's `detail` (400/404).

export async function listKeywordFamilies(): Promise<KeywordFamily[]> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families`)
  if (!r.ok) throw new Error(await errorDetail(r, 'keyword families'))
  return (await r.json()) as KeywordFamily[]
}

export async function createKeywordFamily(
  canonical: string,
  members: string[] = [],
): Promise<KeywordFamily> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canonical, members }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'create keyword family'))
  return (await r.json()) as KeywordFamily
}

export async function renameKeywordFamily(
  familyId: string,
  canonical: string,
): Promise<KeywordFamily> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canonical }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'rename keyword family'))
  return (await r.json()) as KeywordFamily
}

/** Assign a keyword to a family, moving it off any other family it belonged to (ADR-015). */
export async function addFamilyMember(familyId: string, keyword: string): Promise<KeywordFamily> {
  const r = await fetch(
    `${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}/members`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keyword }),
    },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'add family member'))
  return (await r.json()) as KeywordFamily
}

export async function removeFamilyMember(
  familyId: string,
  keyword: string,
): Promise<KeywordFamily> {
  const r = await fetch(
    `${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}/members/${encodeURIComponent(keyword)}`,
    { method: 'DELETE' },
  )
  if (!r.ok) throw new Error(await errorDetail(r, 'remove family member'))
  return (await r.json()) as KeywordFamily
}

export async function deleteKeywordFamily(familyId: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/library/keyword-families/${encodeURIComponent(familyId)}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'delete keyword family'))
}

/** A/B-compare retrieval (U6): the query under the locked defaults vs the session override.
 *  $0 — retrieval only, no generation. `overrides` rides this one request. */
export async function compareRetrieval(
  text: string,
  overrides?: RagOverrides,
): Promise<CompareResult> {
  const r = await fetch(`${API_BASE}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, overrides: overrides ?? null }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'compare'))
  return (await r.json()) as CompareResult
}

/** Stream a chat turn. `/api/chat` is POST-SSE, so we parse the body stream by hand
 *  (EventSource is GET-only). Yields `{event, data}` for token / step / result / done.
 *  `overrides` (ADR-010) rides this one request only — never persisted, never a default. */
export async function* streamChat(
  text: string,
  sessionId: string,
  overrides?: RagOverrides,
  signal?: AbortSignal,
): AsyncGenerator<SSEvent> {
  const r = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, session_id: sessionId, overrides: overrides ?? null }),
    signal,
  })
  if (!r.ok || !r.body) throw new Error(`chat failed: ${r.status}`)

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const ev = parseEventBlock(block)
      if (ev) yield ev
    }
  }
}

function parseEventBlock(block: string): SSEvent | null {
  let event = 'message'
  const data: string[] = []
  for (const line of block.split('\n')) {
    if (line.startsWith(':')) continue // comment / heartbeat
    if (line.startsWith('event:')) event = line.slice(6).trimStart()
    else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
  }
  if (event === 'message' && data.length === 0) return null
  return { event, data: data.join('\n') }
}

export async function adjudicate(
  claimId: string,
  decision: Decision,
  editedText?: string,
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/claims/${encodeURIComponent(claimId)}/adjudicate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision, edited_text: editedText ?? null }),
  })
  if (!r.ok) throw new Error(`adjudicate failed: ${r.status}`)
}

export function figureUrl(figureId: string): string {
  return `${API_BASE}/api/figures/${encodeURIComponent(figureId)}`
}

// ── Settings + first-run ingest (PR-M4 data-home flow) ──────────────────────────
// The backend owns validation + persistence (apps/api/main.py); these are fetch-only.

export async function getSettings(): Promise<Settings> {
  const r = await fetch(`${API_BASE}/api/settings`)
  if (!r.ok) throw new Error(`settings failed: ${r.status}`)
  return (await r.json()) as Settings
}

/** Persist the source documents folder. 400 carries the backend's reason (e.g. not a directory). */
export async function setSourceDir(sourceDir: string): Promise<Settings> {
  const r = await fetch(`${API_BASE}/api/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_dir: sourceDir }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'save settings'))
  return (await r.json()) as Settings
}

/** Switch the live LLM provider/model (ADR-011, U1c) — takes effect on the next turn, no
 *  restart. 400 carries the backend's reason (unknown provider, or one with no key configured). */
export async function setLlmProvider(provider: string, model: string): Promise<Settings> {
  const r = await fetch(`${API_BASE}/api/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ llm_provider: provider, llm_model: model }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'switch provider'))
  return (await r.json()) as Settings
}

/** Kick off a background re-index. No `paths` = the whole saved source folder (minus exclusions);
 *  a `paths` list (rel_paths) = ingest exactly that selection. 409 if one is already running,
 *  400 if any path is invalid (the detail names the offenders). */
export async function startIngest(paths?: string[]): Promise<IngestStatus> {
  const init: RequestInit =
    paths === undefined
      ? { method: 'POST' }
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ paths }),
        }
  const r = await fetch(`${API_BASE}/api/ingest`, init)
  if (!r.ok) throw new Error(await errorDetail(r, 'start ingest'))
  return (await r.json()) as IngestStatus
}

/** Selective ingestion (S2): stat-only scan of the source folder → every file with a derived
 *  ingest status + its `excluded` flag. Cheap ($0/offline); safe to call on every panel open. */
export async function getSources(): Promise<SourceFile[]> {
  const r = await fetch(`${API_BASE}/api/sources`)
  if (!r.ok) throw new Error(`sources failed: ${r.status}`)
  return (await r.json()) as SourceFile[]
}

/** Set a file's `excluded` flag (an excluded file is skipped by a whole-folder index; an explicit
 *  selection still overrides it). Returns the updated row. 404 if the rel_path is unknown. */
export async function patchSource(relPath: string, excluded: boolean): Promise<SourceFile> {
  const r = await fetch(`${API_BASE}/api/sources`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rel_path: relPath, excluded }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'update source'))
  return (await r.json()) as SourceFile
}

export async function getIngestStatus(): Promise<IngestStatus> {
  const r = await fetch(`${API_BASE}/api/ingest/status`)
  if (!r.ok) throw new Error(`ingest status failed: ${r.status}`)
  return (await r.json()) as IngestStatus
}

/** Pull a human message out of a FastAPI error body, falling back to the status code. `detail` is
 *  usually a string, but the selective-ingest 400 sends `{error, offenders}` — render that too. */
async function errorDetail(r: Response, what: string): Promise<string> {
  try {
    const detail = ((await r.json()) as { detail?: unknown }).detail
    if (typeof detail === 'string') return detail
    if (detail && typeof detail === 'object') {
      const d = detail as { error?: string; offenders?: Record<string, string[]> }
      if (d.offenders) {
        const parts = Object.entries(d.offenders)
          .filter(([, v]) => v.length)
          .map(([k, v]) => `${k}: ${v.join(', ')}`)
        return `${d.error ?? 'invalid selection'} — ${parts.join('; ')}`
      }
      return JSON.stringify(detail)
    }
  } catch {
    // non-JSON body — fall through to the status code
  }
  return `${what} failed: ${r.status}`
}

export async function exportConversation(sessionId: string, dev: boolean): Promise<void> {
  const r = await fetch(`${API_BASE}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, dev }),
  })
  if (!r.ok) throw new Error(`export failed: ${r.status}`)
  const blob = await r.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `doc_assistant-${sessionId}-${dev ? 'debug' : 'transcript'}.md`
  a.click()
  URL.revokeObjectURL(url)
}

export type { TurnResult }
