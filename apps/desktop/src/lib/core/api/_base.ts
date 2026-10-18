// Shared base URL for every domain client.
//
// In dev, Vite proxies `/api` → 127.0.0.1:8001 (same-origin, no CORS). In the packaged Tauri
// build the frontend is served from the asset/tauri origin, so it hits the absolute backend URL
// (the API's CORS allowlist includes `tauri://localhost`).

import { needsToken, withToken } from '../launchToken'
import { launchToken } from '../tauri'

export const API_BASE: string = import.meta.env.DEV ? '' : 'http://127.0.0.1:8001'

/** `fetch` for this API: a request that can change something carries the launch token
 *  (security S-6), which the API requires when the shell started it. Every domain client calls the
 *  API through this, never `fetch` itself. */
export async function apiFetch(url: string, init: RequestInit = {}): Promise<Response> {
  return fetch(url, needsToken(init.method) ? withToken(init, await launchToken()) : init)
}

// Shared error-detail unwrapper. FastAPI puts the message in `detail`, which is a string for
// a plain HTTPException and an object for the structured 4xx bodies (e.g. a folder selection
// with offenders). Exported because every domain client needs it.
export async function errorDetail(r: Response, what: string): Promise<string> {
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
