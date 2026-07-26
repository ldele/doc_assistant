// Thin `chat` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/chat.py and apps/api/models/chat.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE } from './_base'
import type {
  Decision,
  RagOverrides,
  Settings,
} from '../types'

export interface SSEvent {
  event: string
  data: string
}
/** Stream a chat turn. `/api/chat` is POST-SSE, so we parse the body stream by hand
 *  (EventSource is GET-only). Yields `{event, data}` for token / step / result / done.
 *  `overrides` (ADR-010) rides this one request only — never persisted, never a default. */
export async function* streamChat(
  text: string,
  sessionId: string,
  overrides?: RagOverrides,
  signal?: AbortSignal,
  scopeFolderId?: string | null,
): AsyncGenerator<SSEvent> {
  const r = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      session_id: sessionId,
      overrides: overrides ?? null,
      // ADR-025 F2 — only the folder id travels; the backend resolves membership per turn.
      scope_folder_id: scopeFolderId ?? null,
    }),
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
