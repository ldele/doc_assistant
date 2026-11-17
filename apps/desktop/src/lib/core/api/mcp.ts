// Thin `mcp` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/mcp.py and apps/api/models/mcp.py (ADR-066, A4).

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { McpSetup, McpStatus } from '../types/mcp'

async function post<T>(path: string, body: unknown, what: string): Promise<T> {
  const r = await apiFetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await errorDetail(r, what))
  return (await r.json()) as T
}

/** Whether the endpoint is on, what it offers, and the latest calls. No secret in it. */
export async function getMcpStatus(): Promise<McpStatus> {
  const r = await apiFetch(`${API_BASE}/api/mcp`)
  if (!r.ok) throw new Error(await errorDetail(r, 'read the MCP settings'))
  return (await r.json()) as McpStatus
}

/** Turn the endpoint, or its conversation tools, on or off. */
export function setMcpSettings(change: { enabled?: boolean; conversations?: boolean }): Promise<McpStatus> {
  return post('/api/mcp/settings', change, 'change the MCP settings')
}

/** The token and what to paste into a client. */
export function getMcpSetup(): Promise<McpSetup> {
  return post('/api/mcp/setup', {}, 'read the MCP setup')
}

/** A new token; every client holding the old one is cut off. */
export function replaceMcpToken(): Promise<McpSetup> {
  return post('/api/mcp/token/replace', {}, 'replace the MCP token')
}
