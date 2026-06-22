// Thin client for the desktop API (PR-M2). No business logic — fetch + SSE parsing only.
//
// In dev, Vite proxies `/api` → 127.0.0.1:8001 (same-origin, no CORS). In the packaged
// Tauri build the frontend is served from the asset/tauri origin, so it hits the absolute
// backend URL (the API's CORS allowlist includes `tauri://localhost`).
import type { Decision, Health, TurnResult } from './types'

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

/** Stream a chat turn. `/api/chat` is POST-SSE, so we parse the body stream by hand
 *  (EventSource is GET-only). Yields `{event, data}` for token / step / result / done. */
export async function* streamChat(
  text: string,
  sessionId: string,
  signal?: AbortSignal,
): AsyncGenerator<SSEvent> {
  const r = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, session_id: sessionId }),
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
