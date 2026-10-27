// Tests for the first-run setup panel's pure helpers (ADR-034).
// Runner: node's built-in `node:test` with native TypeScript stripping — no new dependency.
// `npm test` from apps/desktop.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  DEFAULT_CLAUDE_MODEL,
  costLabel,
  displaySteps,
  keyPrecedenceNote,
  keyRefusal,
  modelForProvider,
  ollamaPending,
  outstandingSteps,
  providerStatusLabel,
  setupSummary,
  suggestOllamaModel,
} from './setup.ts'
import type { ProviderReadiness, SetupState } from '../core/types/setup.ts'

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
  steps: [
    { id: 'provider', title: 'Answer engine', detail: 'ok', done: true, action: null },
    { id: 'documents', title: 'Your documents', detail: 'ok', done: true, action: null },
  ],
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
    steps: [
      { id: 'provider', title: 'p', detail: 'd', done: false, action: 'do it' },
      { id: 'documents', title: 'd', detail: 'd', done: true, action: null },
    ],
  })
  assert.deepEqual(
    outstandingSteps(s).map((x) => x.id),
    ['provider'],
  )
})

test('setupSummary counts what is left and never guesses while loading', () => {
  assert.equal(setupSummary(null), 'Checking…')
  assert.equal(setupSummary(state()), 'Ready to answer questions')
  assert.equal(
    setupSummary(
      state({
        steps: [{ id: 'provider', title: 'p', detail: 'd', done: false, action: null }],
      }),
    ),
    '1 step left',
  )
})

test('costLabel names the metered path without blocking it', () => {
  assert.match(costLabel(provider({ paid: true })), /metered/)
  assert.match(costLabel(provider({ paid: false })), /free/)
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
  assert.equal(providerStatusLabel(provider()), 'ready')
  assert.equal(providerStatusLabel(provider({ ready: false, reachable: false })), 'not running')
  assert.equal(
    providerStatusLabel(provider({ id: 'anthropic', ready: false, configured: false })),
    'needs a key',
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
  const step = displaySteps(unprobed, true).find((s) => s.id === 'provider')
  assert.equal(step?.done, false)
  assert.match(step?.detail ?? '', /Checking whether Ollama is answering/)
  assert.equal(providerStatusLabel(unprobed.providers[0], true), 'checking…')
  // Once probed, the backend's verdict is shown as it is.
  assert.deepEqual(displaySteps(state(), true), state().steps)
  // Another active engine is not held back by the Ollama probe.
  assert.deepEqual(displaySteps({ ...unprobed, active_provider: 'anthropic' }, true), unprobed.steps)
})

test('a refused key says nothing was saved, then why', () => {
  assert.equal(keyRefusal('invalid x-api-key'), 'Anthropic refused this key, so nothing was saved. invalid x-api-key')
})
