// Security S-6 (docs/security.md S1b) — pure helpers, no fetch and no Tauri, so `node --test`
// can reach them.
//
// The desktop shell mints a token at every launch and gives it to the API sidecar and to this
// window; a request that can change something carries it in `X-Doc-Api-Token`, and the API answers
// 401 without it. Reads never carry it: a custom header on a cross-origin GET costs a preflight
// per request, and the API does not check reads. `_base.ts` fetches the token and applies this.

export const TOKEN_HEADER = 'X-Doc-Api-Token'

const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])

/** True for a method the API checks: anything but GET, HEAD and OPTIONS (fetch's default is GET). */
export function needsToken(method: string | undefined): boolean {
  return !SAFE_METHODS.has((method ?? 'GET').toUpperCase())
}

/** `init` with the token header added when the method needs it and there is a token; else `init`
 *  itself. The caller's own headers are kept. */
export function withToken(init: RequestInit, token: string | null): RequestInit {
  if (!token || !needsToken(init.method)) return init
  const headers = new Headers(init.headers)
  headers.set(TOKEN_HEADER, token)
  return { ...init, headers }
}
