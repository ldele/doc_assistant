import assert from 'node:assert/strict'
import { test } from 'node:test'

import { TOKEN_HEADER, needsToken, withToken } from './launchToken.ts'

test('a method that can change something needs the token; a read does not', () => {
  for (const m of ['POST', 'PUT', 'PATCH', 'DELETE', 'post', 'delete']) assert.equal(needsToken(m), true, m)
  for (const m of ['GET', 'HEAD', 'OPTIONS', 'get', undefined]) assert.equal(needsToken(m), false, String(m))
})

test('the header is added to a write and the caller headers are kept', () => {
  const init = withToken(
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' },
    'tok',
  )
  const headers = new Headers(init.headers)
  assert.equal(headers.get(TOKEN_HEADER), 'tok')
  assert.equal(headers.get('Content-Type'), 'application/json')
  assert.equal(init.body, '{}')
  assert.equal(init.method, 'POST')
})

test('a write with no headers of its own gets the header too', () => {
  const headers = new Headers(withToken({ method: 'DELETE' }, 'tok').headers)
  assert.equal(headers.get(TOKEN_HEADER), 'tok')
})

test('a read, or no token, leaves the request as it was', () => {
  const read: RequestInit = { headers: { Accept: 'application/json' } }
  assert.equal(withToken(read, 'tok'), read)
  const write: RequestInit = { method: 'POST' }
  assert.equal(withToken(write, null), write)
  assert.equal(withToken(write, ''), write)
})
