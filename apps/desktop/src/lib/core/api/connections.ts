// Thin `connections` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/connections.py and apps/api/models/connections.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE } from './_base'
import type {
  DocConnections,
} from '../types'

/** One document's exploration bundle (ADR-027 D1, E4): related papers + citation edges +
 *  extracted external references. Pure sidecar read, $0. 404 unknown. */
export async function getDocConnections(docId: string): Promise<DocConnections> {
  const r = await fetch(
    `${API_BASE}/api/library/documents/${encodeURIComponent(docId)}/connections`,
  )
  if (!r.ok) throw new Error(`document connections failed: ${r.status}`)
  return (await r.json()) as DocConnections
}
