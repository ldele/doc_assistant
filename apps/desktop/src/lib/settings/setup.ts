// Pure helpers for the first-run setup panel (ADR-034). Plain `.ts`, not `.svelte.ts`, so
// `npm test` can run it (apps/desktop/CLAUDE.md — the extension is the marker).
//
// Everything here is presentation logic over the `/api/setup` payload. The *judgements* (ready,
// what to do next) come from the backend so the docs, the API and the UI cannot drift apart.

import type { ProviderReadiness, SetupState } from '../core/types/setup'

/** The chat model to offer first when Ollama has models installed.
 *
 * Preference order: the model this project documents as its local default, then anything from the
 * same family, then the first installed model. Never "none" when the list is non-empty — an empty
 * pre-selection would make the Use button look broken. */
export function suggestOllamaModel(models: string[]): string | null {
  if (models.length === 0) return null
  const preferred = ['llama3.1:8b', 'llama3.1', 'llama3']
  for (const p of preferred) {
    const hit = models.find((m) => m === p || m.startsWith(`${p}:`))
    if (hit) return hit
  }
  return models[0]
}

/** The Claude model offered when Anthropic is picked without one (the app's documented default). */
export const DEFAULT_CLAUDE_MODEL = 'claude-haiku-4-5-20251001'

const isClaude = (model: string): boolean => /^claude/i.test(model.trim())

/** The model to put in the box when the provider changes: the current one if that provider can
 *  serve it, else one it can. A switch that kept the old model left the 0.7.0 walkthrough's
 *  sandbox on Ollama with a Claude model, which can never answer. `installed` is what an Ollama
 *  probe found; empty means unknown, so a non-Claude name is kept. */
export function modelForProvider(provider: string, current: string, installed: string[] = []): string {
  const model = current.trim()
  if (provider === 'anthropic') return isClaude(model) ? model : DEFAULT_CLAUDE_MODEL
  if (provider === 'ollama') {
    const fits = model !== '' && !isClaude(model) && (installed.length === 0 || installed.includes(model))
    return fits ? model : (suggestOllamaModel(installed) ?? 'llama3.1:8b')
  }
  return model
}

/** Whether the Ollama verdict is still to come: a first draw from local state (`probe=false`)
 *  answers before the network probe, which can take seconds when nothing is listening. */
export function ollamaPending(state: SetupState | null, probing: boolean): boolean {
  const o = state?.providers.find((p) => p.id === 'ollama')
  return probing && !!o && o.reachable === null
}

/** The steps as shown: while the Ollama probe is out and Ollama is the active engine, its step
 *  says it is being checked rather than the unprobed "done" the first draw carries. */
export function displaySteps(state: SetupState | null, probing: boolean): SetupState['steps'] {
  const steps = state?.steps ?? []
  if (!state || state.active_provider !== 'ollama' || !ollamaPending(state, probing)) return steps
  return steps.map((s) =>
    s.id === 'provider'
      ? { ...s, title: 'Answer engine', detail: 'Checking whether Ollama is answering…', done: false, action: null }
      : s,
  )
}

/** What a refused key reads as, next to the key field: what happened, then Anthropic's reason. */
export function keyRefusal(reason: string): string {
  return `Anthropic refused this key, so nothing was saved. ${reason}`.trim()
}

/** Steps the user still has to do, in the backend's order. */
export function outstandingSteps(state: SetupState | null): SetupState['steps'] {
  return (state?.steps ?? []).filter((s) => !s.done)
}

/** One-line status for the header: how much is left. */
export function setupSummary(state: SetupState | null): string {
  if (!state) return 'Checking…'
  const left = outstandingSteps(state).length
  if (left === 0) return 'Ready to answer questions'
  return left === 1 ? '1 step left' : `${left} steps left`
}

/** Cost wording for a provider — informational, never a gate. */
export function costLabel(p: ProviderReadiness): string {
  return p.paid ? 'metered by the provider' : 'runs locally, free'
}

/** Why a key the user just saved may not be the one in use.
 *
 * Returns `null` unless a `.env` key is shadowing the app's own — the only case where the app
 * would otherwise show a key it is not sending (the honesty requirement in ADR-034 D2). */
export function keyPrecedenceNote(p: ProviderReadiness, savedInApp: boolean): string | null {
  if (p.key_source === 'env' && savedInApp) {
    return 'A key in your .env file takes precedence, so that one is being used. Remove it there to use the key saved here.'
  }
  return null
}

/** Whether the provider picker should offer this provider as a working choice.
 *
 * Unready providers stay selectable (inform-don't-block: Ollama may just not be started yet), so
 * this only drives the label, never a `disabled` attribute. */
export function providerStatusLabel(p: ProviderReadiness, probing = false): string {
  if (probing && p.id === 'ollama' && p.reachable === null) return 'checking…'
  if (p.ready) return 'ready'
  if (p.reachable === false) return 'not running'
  if (!p.configured) return 'needs a key'
  return 'not ready'
}
