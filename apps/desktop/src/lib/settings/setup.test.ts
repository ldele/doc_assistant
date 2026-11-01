// Tests for the first-run setup panel's pure helpers (ADR-034).
// Runner: node's built-in `node:test` with native TypeScript stripping — no new dependency.
// `npm test` from apps/desktop.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  DEFAULT_CLAUDE_MODEL,
  answeringWith,
  keyPrecedenceNote,
  keyRefusal,
  modelForProvider,
  ollamaPending,
  outstandingSteps,
  providerGroups,
  providerStatusLabel,
  setupSummary,
  stepTone,
  suggestOllamaModel,
} from './setup.ts'
import type { ProviderReadiness, SetupState, SetupStep } from '../core/types/setup.ts'

const provider = (over: Partial<ProviderReadiness> = {}): ProviderReadiness => ({
  id: 'ollama',
  paid: false,
  configured: true,
  reachable: true,
  ready: true,
  detail: 'fine',
  action: null,
  key_source: null,
  key_hint: null,
  models: [],
  name: 'Ollama',
  group: 'local',
  destination: 'Your questions and the passages found for them stay on this computer.',
  ...over,
})

const step = (over: Partial<SetupStep> = {}): SetupStep => ({
  id: 'provider',
  title: 'Model',
  detail: 'ok',
  done: true,
  status: 'Engine online',
  action: null,
  offline: false,
  ...over,
})

const state = (over: Partial<SetupState> = {}): SetupState => ({
  providers: [provider()],
  active_provider: 'ollama',
  active_model: 'llama3.1:8b',
  active_ready: true,
  chunk_count: 10,
  document_count: 2,
  ollama_host: 'http://localhost:11434',
  steps: [step(), step({ id: 'documents', title: 'Documents', status: 'Indexed' })],
  ready: true,
  ...over,
})

test('suggestOllamaModel prefers the documented local default', () => {
  assert.equal(suggestOllamaModel(['gemma:2b', 'llama3.1:8b', 'phi3']), 'llama3.1:8b')
})

test('suggestOllamaModel falls back within the family, then to the first model', () => {
  assert.equal(suggestOllamaModel(['gemma:2b', 'llama3.1:70b']), 'llama3.1:70b')
  assert.equal(suggestOllamaModel(['gemma:2b', 'phi3']), 'gemma:2b')
})

test('suggestOllamaModel returns null only when nothing is installed', () => {
  // An empty pre-selection with models present would make the Use button look broken.
  assert.equal(suggestOllamaModel([]), null)
})

test('outstandingSteps keeps only unfinished steps, in the backend order', () => {
  const s = state({
    steps: [step({ done: false, action: 'do it' }), step({ id: 'documents' })],
  })
  assert.deepEqual(
    outstandingSteps(s).map((x) => x.id),
    ['provider'],
  )
})

test('setupSummary counts what is left and never guesses while loading', () => {
  assert.equal(setupSummary(null), 'Checking…')
  assert.equal(setupSummary(state()), 'Provenote is ready')
  assert.equal(
    setupSummary(state({ steps: [step({ done: false })] })),
    '1 thing left before Provenote can answer',
  )
  assert.equal(
    setupSummary(state({ steps: [step({ done: false }), step({ id: 'documents', done: false })] })),
    '2 things left before Provenote can answer',
  )
})

test('a row is green when done, red when the engine is offline, plain until then', () => {
  assert.equal(stepTone(step()), 'ok')
  assert.equal(stepTone(step({ done: false })), 'todo')
  assert.equal(stepTone(step({ done: false, offline: true })), 'down')
})

test('the card names the model and where the text goes when it can answer', () => {
  const card = answeringWith(state(), false)
  assert.equal(card.model, 'llama3.1:8b')
  assert.equal(card.status, 'Ready')
  assert.equal(card.tone, 'ok')
  assert.match(card.line, /stay on this computer/)
  assert.equal(card.action, null)
})

test('the card says what stands in the way, in the backend\'s words, when it cannot', () => {
  const offline = answeringWith(
    state({
      active_ready: false,
      steps: [
        step({
          done: false,
          offline: true,
          status: 'Engine offline',
          detail: 'No Ollama server answering at x.',
          action: 'Run `ollama serve`.',
        }),
      ],
    }),
    false,
  )
  assert.equal(offline.model, 'llama3.1:8b') // still the chosen model, and it is the one down
  assert.equal(offline.status, 'Engine offline')
  assert.equal(offline.tone, 'down')
  assert.equal(offline.line, 'No Ollama server answering at x.')
  assert.equal(offline.action, 'Run `ollama serve`.')

  const notSet = answeringWith(
    state({
      active_provider: 'anthropic',
      active_ready: false,
      steps: [step({ done: false, status: 'Not set', detail: 'No model chosen yet.' })],
    }),
    false,
  )
  assert.equal(notSet.model, null) // a default nobody can use is not a model answering
  assert.equal(notSet.tone, 'todo')
  assert.equal(notSet.line, 'No model chosen yet.')
})

test('providers sit in the list of where their text goes, and an empty list is not drawn', () => {
  const claude = provider({ id: 'anthropic', name: 'Claude', group: 'cloud' })
  assert.deepEqual(
    providerGroups([claude, provider()]).map((g) => [g.label, g.providers.map((p) => p.name)]),
    [
      ['On this computer', ['Ollama']],
      ['Cloud, with your own key', ['Claude']],
    ],
  )
  // An Ollama on another machine is not on this computer, whatever its name suggests.
  assert.deepEqual(
    providerGroups([provider({ group: 'server' })]).map((g) => g.label),
    ['Your own server'],
  )
})

test('keyPrecedenceNote warns only when .env shadows the key saved in the app', () => {
  const envKey = provider({ id: 'anthropic', paid: true, key_source: 'env' })
  assert.match(keyPrecedenceNote(envKey, true) ?? '', /\.env/)
  assert.equal(keyPrecedenceNote(envKey, false), null)
  assert.equal(keyPrecedenceNote(provider({ key_source: 'app' }), true), null)
})

test('providerStatusLabel separates "not running" from "needs a key"', () => {
  // The two states have different fixes, so one label for both would send the user down the
  // wrong path.
  assert.equal(providerStatusLabel(provider()), 'Ready')
  assert.equal(providerStatusLabel(provider({ ready: false, reachable: false })), 'Not running')
  assert.equal(
    providerStatusLabel(provider({ id: 'anthropic', ready: false, configured: false })),
    'Needs a key',
  )
})

test('switching provider keeps a model that provider serves, and replaces one it cannot', () => {
  // The walkthrough's stuck sandbox: Ollama picked, the Claude model kept.
  assert.equal(modelForProvider('ollama', 'claude-haiku-4-5-20251001', ['qwen2.5:7b', 'llama3.1:8b']), 'llama3.1:8b')
  assert.equal(modelForProvider('ollama', 'claude-haiku-4-5-20251001', []), 'llama3.1:8b')
  assert.equal(modelForProvider('ollama', 'qwen2.5:7b', ['qwen2.5:7b']), 'qwen2.5:7b')
  assert.equal(modelForProvider('ollama', 'mistral:7b', []), 'mistral:7b') // unknown list: kept
  assert.equal(modelForProvider('ollama', 'mistral:7b', ['llama3.1:8b']), 'llama3.1:8b')
  assert.equal(modelForProvider('anthropic', 'llama3.1:8b'), DEFAULT_CLAUDE_MODEL)
  assert.equal(modelForProvider('anthropic', 'claude-sonnet-4-5'), 'claude-sonnet-4-5')
})

test('the first draw says Ollama is being checked, not that it is done', () => {
  const unprobed = state({
    providers: [provider({ reachable: null, ready: true, detail: 'Local server (not checked).' })],
  })
  assert.equal(ollamaPending(unprobed, true), true)
  assert.equal(ollamaPending(unprobed, false), false)
  assert.equal(answeringWith(unprobed, true).status, 'Checking…')
  assert.equal(providerStatusLabel(unprobed.providers[0], true), 'Checking…')
  // Once probed, the backend's verdict is shown as it is.
  assert.equal(answeringWith(state(), true).status, 'Ready')
  // Another active engine is not held back by the Ollama probe.
  assert.equal(answeringWith({ ...unprobed, active_provider: 'anthropic' }, true).status, 'Ready')
})

test('a refused key says nothing was saved, then why', () => {
  assert.equal(keyRefusal('invalid x-api-key'), 'Anthropic refused this key, so nothing was saved. invalid x-api-key')
})
