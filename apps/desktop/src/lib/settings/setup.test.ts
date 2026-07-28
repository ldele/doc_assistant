// Tests for the first-run setup panel's pure helpers (ADR-034).
// Runner: node's built-in `node:test` with native TypeScript stripping — no new dependency.
// `npm test` from apps/desktop.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  costLabel,
  keyPrecedenceNote,
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
