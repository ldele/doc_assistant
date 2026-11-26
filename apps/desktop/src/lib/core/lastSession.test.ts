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

test('a copy written by another app version is never read back (KI-99)', () => {
  const store = memory()
  saveSnapshot(store, 'documents', [{ id: 'a' }], new Date('2026-11-26T10:00:00Z'), '0.8.0')
  assert.deepEqual(loadSnapshot(store, 'documents', '0.8.0')?.data, [{ id: 'a' }])
  assert.equal(loadSnapshot(store, 'documents', '0.8.1'), null) // an upgrade
  assert.equal(loadSnapshot(store, 'documents', '0.7.3'), null) // a downgrade
})

test("0.7.3's chat list, which lacks the rows' labels, is not drawn by a later version (KI-99)", () => {
  // The shape 0.7.3 kept, as found in a real window's storage on the 0.8.0 upgrade: format 1, no
  // app version, chat rows without `labels`, `mode_name` or `folders`. Drawn, its first row threw
  // at mount and the window stayed blank on every launch.
  const store = memory()
  const row = {
    session_id: 's1',
    title: 'A chat',
    started_at: '2026-11-20T09:00:00Z',
    last_at: '2026-11-20T09:05:00Z',
    turn_count: 2,
    pinned: false,
    archived: false,
  }
  store.map.set(
    storageKey('conversations'),
    JSON.stringify({ v: 1, savedAt: '2026-11-25T09:04:00.334Z', data: [row] }),
  )
  assert.equal(loadSnapshot(store, 'conversations', '0.8.0'), null)
  assert.equal(loadSnapshot(store, 'conversations'), null) // the default version too
})
