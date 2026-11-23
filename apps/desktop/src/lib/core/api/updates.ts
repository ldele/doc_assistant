// Thin `updates` client for the desktop API — fetch + parsing only, no business logic.
// Pairs with apps/api/routers/updates.py and apps/api/models/updates.py (ADR-044).

import { API_BASE, apiFetch, errorDetail } from './_base'
import type { UpdateStatus } from '../types'

/** The cached answer, after the automatic check when the user opted in and one is due (24 h);
 *  free on mount in the default configuration. A POST since S-17: the check calls out and writes
 *  settings, which a GET may not. */
export async function getUpdateStatus(): Promise<UpdateStatus> {
  const r = await apiFetch(`${API_BASE}/api/updates/auto`, { method: 'POST' })
  if (!r.ok) throw new Error(`update status failed: ${r.status}`)
  return (await r.json()) as UpdateStatus
}

/** Check now, regardless of the automatic-check toggle — an explicit press is its own consent
 *  (ADR-044). Never throws for an offline machine: that comes back as state 'unknown'. */
export async function checkForUpdate(): Promise<UpdateStatus> {
  const r = await apiFetch(`${API_BASE}/api/updates/check`, { method: 'POST' })
  if (!r.ok) throw new Error(await errorDetail(r, 'check for updates'))
  return (await r.json()) as UpdateStatus
}

/** Turn the automatic daily check on or off. */
export async function setAutoUpdateCheck(enabled: boolean): Promise<UpdateStatus> {
  const r = await apiFetch(`${API_BASE}/api/updates/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ auto_check_enabled: enabled }),
  })
  if (!r.ok) throw new Error(await errorDetail(r, 'save update setting'))
  return (await r.json()) as UpdateStatus
}
