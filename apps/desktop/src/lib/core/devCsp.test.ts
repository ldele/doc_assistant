import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { devCsp } from './devCsp.ts'

const conf = JSON.parse(
  readFileSync(new URL('../../../src-tauri/tauri.conf.json', import.meta.url), 'utf-8'),
)
const PRODUCTION: string = conf.app.security.csp

const directives = (csp: string): Map<string, string[]> =>
  new Map(
    csp
      .split(';')
      .map((p) => p.trim().split(/\s+/))
      .filter((t) => t[0])
      .map(([name, ...values]) => [name, values]),
  )

test('the dev policy keeps every production directive and value', () => {
  const dev = directives(devCsp(PRODUCTION, 1420))
  for (const [name, values] of directives(PRODUCTION)) {
    for (const v of values) assert.ok(dev.get(name)?.includes(v), `${name} lost ${v}`)
  }
})

test('dev never lets injected script run', () => {
  const dev = directives(devCsp(PRODUCTION, 1420))
  // No script-src means script falls back to default-src — which must stay 'self' only.
  assert.equal(dev.get('script-src'), undefined)
  assert.deepEqual(dev.get('default-src'), ["'self'"])
  assert.ok(!devCsp(PRODUCTION, 1420).includes("'unsafe-eval'"))
})

test('dev adds only inline styles, the HMR socket and Tauri IPC', () => {
  const dev = directives(devCsp(PRODUCTION, 1420))
  assert.deepEqual(dev.get('style-src'), ["'self'", "'unsafe-inline'"])
  const connect = dev.get('connect-src') ?? []
  assert.ok(connect.includes('ws://localhost:1420'))
  assert.ok(connect.includes('ipc:') && connect.includes('http://ipc.localhost'))
})
