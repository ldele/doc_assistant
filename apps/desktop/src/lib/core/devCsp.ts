/**
 * The dev server's Content-Security-Policy, derived from the production one (docs/security.md S2).
 *
 * Tauri injects `app.security.csp` only into the assets it serves itself — the built app. In
 * `tauri dev` on desktop the webview loads the Vite server at `devUrl` directly, so no policy
 * reached the dev loop, and `devCsp` would not change that (it is read on the same injection path;
 * checked in the Tauri 2.11 source, `manager/mod.rs` `csp()` → `get_asset`). The dev server sends
 * the header itself instead, built from the production string so the two cannot drift.
 *
 * Dev adds only what the dev loop cannot run without, and nothing that lets injected script run:
 * - `style-src 'unsafe-inline'` — Vite injects each stylesheet as a `<style>` element;
 * - the HMR websocket on the dev port;
 * - Tauri's IPC endpoints, which its own CSP injection adds in the built app.
 * `script-src` is never loosened: an injected `<img onerror>` stays blocked in dev as in the build.
 */
export function devCsp(productionCsp: string, devPort: number): string {
  const directives = new Map<string, string[]>()
  for (const part of productionCsp.split(';')) {
    const [name, ...values] = part.trim().split(/\s+/)
    if (name) directives.set(name, values)
  }
  const base = directives.get('default-src') ?? ["'self'"]
  const add = (name: string, ...values: string[]): void => {
    const current = directives.get(name) ?? [...base]
    for (const v of values) if (!current.includes(v)) current.push(v)
    directives.set(name, current)
  }
  add('style-src', "'unsafe-inline'")
  add('connect-src', `ws://localhost:${devPort}`, 'ipc:', 'http://ipc.localhost')
  return [...directives].map(([name, values]) => [name, ...values].join(' ')).join('; ')
}
