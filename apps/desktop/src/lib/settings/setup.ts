// Pure helpers for the first-run setup panel (ADR-034). Plain `.ts`, not `.svelte.ts`, so
// `npm test` can run it (apps/desktop/CLAUDE.md — the extension is the marker).
//
// Everything here is presentation logic over the `/api/setup` payload. The *judgements* (ready,
// what to do next) come from the backend so the docs, the API and the UI cannot drift apart.

import type { ProviderReadiness, SetupState, SetupStep } from '../core/types/setup'

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
  // A compatible server names its models its own way (a router serves `anthropic/claude-…`), so
  // a typed name is kept; an empty box takes the first one the server listed.
  return model || installed[0] || ''
}

/** What the form says under the model box when the typed model is not in the server's list
 *  (ADR-060, choice 8): named, and still saved. `null` when it is listed or there is no list. */
export function unlistedModel(model: string, listed: string[]): string | null {
  const name = model.trim()
  if (!name || listed.length === 0 || listed.includes(name)) return null
  return `“${name}” is not in the list this server gave. It will be saved as typed.`
}

/** The warning before a server's address changes while it holds a key (choice 3): the key is
 *  sent only to the address it was saved with. */
export function addressChangeNote(saved: string, typed: string, hasKey: boolean): string | null {
  const same = (a: string): string => a.trim().replace(/\/+$/, '').toLowerCase()
  if (!hasKey || !saved || !typed.trim() || same(saved) === same(typed)) return null
  return 'Changing the address removes the saved key. Enter it again for the new address.'
}

/** `14:02`, in this computer's time, from an ISO timestamp; `''` when there is none. */
export function clockTime(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

/** Whether the Ollama verdict is still to come: a first draw from local state (`probe=false`)
 *  answers before the network probe, which can take seconds when nothing is listening. */
export function ollamaPending(state: SetupState | null, probing: boolean): boolean {
  const o = state?.providers.find((p) => p.id === 'ollama')
  return probing && !!o && o.reachable === null
}

/** How a step's row is coloured: green when done, red when the chosen engine does not answer, and
 *  plain when it has not been done yet. The backend decides which; this names the colour. */
export type Tone = 'ok' | 'down' | 'todo'

export function stepTone(step: SetupStep): Tone {
  if (step.done) return 'ok'
  return step.offline ? 'down' : 'todo'
}

/** What the *Answering with* card shows. `model` is `null` when no model can answer yet, and
 *  `line` says where the text goes once one can, or what stands in the way until then. `billing`
 *  is who bills, a fact of its own (ADR-060's amendment), and empty when nothing can answer. */
export interface AnsweringWith {
  model: string | null
  provider: string
  status: string
  tone: Tone
  line: string
  billing: string
  action: string | null
}

export function answeringWith(state: SetupState, probing: boolean): AnsweringWith {
  const step = state.steps.find((s) => s.id === 'provider')
  const active = state.providers.find((p) => p.id === state.active_provider)
  // Worked out per model: an Ollama cloud model sends the text to ollama.com from localhost.
  const line = state.active_destination || active?.destination || ''
  const billing = state.active_billing || active?.billing || ''
  const provider = state.active_name || active?.name || state.active_provider
  const answering = { model: state.active_model, provider, line, billing, action: null }
  // The first draw comes before the probe, so its "done" means nobody has asked Ollama yet.
  if (state.active_provider === 'ollama' && ollamaPending(state, probing)) {
    return { ...answering, status: 'Checking…', tone: 'todo' }
  }
  if (!step || step.done) {
    return { ...answering, status: 'Ready', tone: 'ok' }
  }
  return {
    model: step.offline ? state.active_model : null,
    provider,
    status: step.status,
    tone: stepTone(step),
    line: step.detail,
    billing: step.offline ? billing : '',
    action: step.action,
  }
}

// Providers are listed by where the text goes, nearest first. A provider joins a list by its
// `group` from the backend, and a list with nobody in it is not drawn.
const PROVIDER_GROUPS: readonly { id: ProviderReadiness['group']; label: string }[] = [
  { id: 'local', label: 'On this computer' },
  { id: 'cloud', label: 'Cloud, with your own key' },
  { id: 'server', label: 'Your own server' },
]

export function providerGroups(
  providers: ProviderReadiness[],
): { id: string; label: string; providers: ProviderReadiness[] }[] {
  return PROVIDER_GROUPS.map((g) => ({ ...g, providers: providers.filter((p) => p.group === g.id) })).filter(
    (g) => g.providers.length > 0,
  )
}

/** What a refused key reads as, next to the key field: what happened, then the server's reason
 *  when it adds one. */
export function keyRefusal(name: string, reason: string): string {
  const said = reason.trim()
  const lead = `${name} refused this key, so nothing was saved.`
  return said && !said.startsWith(`${name} refused this key`) ? `${lead} ${said}` : lead
}

/** Steps the user still has to do, in the backend's order. */
export function outstandingSteps(state: SetupState | null): SetupState['steps'] {
  return (state?.steps ?? []).filter((s) => !s.done)
}

/** Getting started's heading: how much is left. */
export function setupSummary(state: SetupState | null): string {
  if (!state) return 'Checking…'
  const left = outstandingSteps(state).length
  if (left === 0) return 'Provenote is ready'
  return `${left} thing${left === 1 ? '' : 's'} left before Provenote can answer`
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
  // A probe asks every server on this computer at once, so each of them is pending until it ends.
  const probed = p.id === 'ollama' || (p.kind !== 'claude' && p.reach === 'this_computer')
  if (probing && probed && p.reachable === null) return 'Checking…'
  if (p.key_status === 'refused') return 'Key refused'
  if (p.ready) return 'Ready'
  if (p.reachable === false) return p.reach === 'this_computer' ? 'Not running' : 'Not reached'
  if (!p.configured) return 'Needs a key'
  return 'Not ready'
}
