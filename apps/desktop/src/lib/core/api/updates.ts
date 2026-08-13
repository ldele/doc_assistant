// Thin `updates` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/updates.py and apps/api/models/updates.py (ADR-044).

import { API_BASE, errorDetail } from './_base'
import type { UpdateStatus } from '../types'

/** The cached answer. Costs a network request only if the user opted into automatic checks and
 *  one is due (24 h), so calling this on mount is free in the default configuration. */
export async function getUpdateStatus(): Promise<UpdateStatus> {
  const r = await fetch(`${API_BASE}/api/updates`)
  if (!r.ok) throw new Error(`update status failed: ${r.status}`)
  return (await r.json()) as UpdateStatus
}

/** Check now, regardless of the automatic-check toggle — an explicit press is its own consent
 *  (ADR-044). Never throws for an offline machine: that comes back as state 'unknown'. */
export async function checkForUpdate(): Promise<UpdateStatus> {
  const r = await fetch(`${API_BASE}/api/updates/check`, { method: 'POST' })
  if (!r.ok) throw new Error(await errorDetail(r, 'check for updates'))
  return (await r.json()) as UpdateStatus
}

/** Turn the automatic daily check on or off. */
export async function setAutoUpdateCheck(enabled: boolean): Promise<UpdateStatus> {
  const r = await fetch(`${API_BASE}/api/updates/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ auto_check_enabled: enabled }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'save update setting'))
  return (await r.json()) as UpdateStatus
}
