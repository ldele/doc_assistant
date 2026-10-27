import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  MAX_SNAPSHOT_CHARS,
  type SnapshotStorage,
  loadSnapshot,
  saveSnapshot,
  storageKey,
} from './lastSession.ts'

function memory(): SnapshotStorage & { map: Map<string, string> } {
  const map = new Map<string, string>()
  return {
    map,
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => void map.set(k, v),
    removeItem: (k) => void map.delete(k),
  }
}

test('a saved list comes back at the next launch, with when it was saved', () => {
  const store = memory()
  const docs = [{ id: 'a', title: 'A paper' }]
  saveSnapshot(store, 'documents', docs, new Date('2026-10-27T10:00:00Z'))
  assert.deepEqual(loadSnapshot(store, 'documents'), { data: docs, savedAt: '2026-10-27T10:00:00.000Z' })
  assert.equal(loadSnapshot(store, 'folders'), null) // each list is kept apart
})

test('an empty library is a real state and is kept as one', () => {
  const store = memory()
  saveSnapshot(store, 'documents', [])
  assert.deepEqual(loadSnapshot(store, 'documents')?.data, [])
})

test('a copy past the cap is not kept, and an older one is removed rather than left behind', () => {
  const store = memory()
  saveSnapshot(store, 'documents', [{ id: 'small' }])
  saveSnapshot(store, 'documents', ['x'.repeat(MAX_SNAPSHOT_CHARS)])
  assert.equal(loadSnapshot(store, 'documents'), null)
})

test('no storage, a refused write, or an unreadable copy never throws', () => {
  assert.equal(loadSnapshot(null, 'graph'), null)
  saveSnapshot(null, 'graph', { nodes: [] })
  const refusing: SnapshotStorage = {
    getItem: () => null,
    setItem: () => {
      throw new Error('quota exceeded')
    },
    removeItem: () => {},
  }
  saveSnapshot(refusing, 'graph', { nodes: [] })
  const store = memory()
  store.map.set(storageKey('graph'), '{not json')
  assert.equal(loadSnapshot(store, 'graph'), null)
  store.map.set(storageKey('graph'), JSON.stringify({ v: 999, savedAt: 'x', data: {} }))
  assert.equal(loadSnapshot(store, 'graph'), null) // another format version
})
