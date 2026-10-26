import { test } from 'node:test'
import assert from 'node:assert/strict'

import { emphasisMarks, withoutEmphasis } from './emphasis.ts'

test('a citation the extractor wrote bold-italic reads as plain text', () => {
  // The Definition card showed this verbatim on the 0.7.0 candidate (2026-10-26).
  assert.equal(
    withoutEmphasis('brain targets _**Jakobs et al., 2019**_ .'),
    'brain targets Jakobs et al., 2019 .',
  )
  assert.equal(
    withoutEmphasis('( _**Bullmore and Sporns, 2009**_ ; _**Sporns et al., 2004**_ ).'),
    '( Bullmore and Sporns, 2009 ; Sporns et al., 2004 ).',
  )
})

test('italic and bold lose their markers', () => {
  assert.equal(
    withoutEmphasis('_Functional specialisation_ considers **large** regions'),
    'Functional specialisation considers large regions',
  )
})

test('an underscore inside a word and a single star are text', () => {
  assert.equal(withoutEmphasis('the snake_case name x_1 at p < 0.05*'), 'the snake_case name x_1 at p < 0.05*')
  assert.deepEqual([...emphasisMarks('plain text')], [])
})

test('the marks are positions in the original text, so highlight spans stay valid', () => {
  const text = '_**Moyle et al.**_'
  assert.deepEqual([...emphasisMarks(text)].sort((a, b) => a - b), [0, 1, 2, 15, 16, 17])
})
