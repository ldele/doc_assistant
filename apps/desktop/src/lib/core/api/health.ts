// Thin `health` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/health.py and apps/api/models/health.py; see
// docs/architecture.md, section "apps/ — the domain spine".

import { API_BASE } from './_base'
import type {
  Health,
} from '../types'

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${API_BASE}/api/health`)
  if (!r.ok) throw new Error(`health failed: ${r.status}`)
  return (await r.json()) as Health
}
