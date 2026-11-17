import { test } from 'node:test'
import assert from 'node:assert/strict'

import { callLine, toolsLine } from './mcp.ts'

test('a call line names the tool, its items and its time, never its question', () => {
  const line = callLine({ tool: 'search_library', items: 5, milliseconds: 120, at: '2026-11-17T14:02:00Z' }, 'en-GB')
  assert.match(line, /^search_library · 5 items · 120 ms · \d\d:\d\d$/)
  assert.equal(callLine({ tool: 'get_document', items: 1, milliseconds: 4, at: 'bad' }), 'get_document · 1 item · 4 ms')
})

test('the tools line says whether conversations are offered', () => {
  assert.equal(toolsLine(['a', 'b'], false), '2 tools. Past conversations are not offered.')
  assert.equal(toolsLine(['a', 'b', 'c'], true), '3 tools, past conversations included.')
})
