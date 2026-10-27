import assert from 'node:assert/strict'
import { test } from 'node:test'

import { hiddenLine, selectedRows, tickAll, tickable } from './addselection.ts'

const add = (path: string) => ({ path, selected_by_default: true })
const ruledOut = (path: string) => ({ path, selected_by_default: false })

test('only a row the server starts ticked carries a checkbox', () => {
  assert.equal(tickable(add('a.pdf')), true)
  assert.equal(tickable(ruledOut('dup.pdf')), false)
})

test('untouched, the selection is exactly what the server ticked, in its order', () => {
  const files = [ruledOut('dup.pdf'), add('a.pdf'), add('b.pdf')]
  assert.deepEqual(
    selectedRows(files, new Set()).map((f) => f.path),
    ['a.pdf', 'b.pdf'],
  )
})

test('an unticked row is left out, the others stay', () => {
  const files = [add('a.pdf'), add('b.pdf'), add('c.pdf')]
  assert.deepEqual(
    selectedRows(files, new Set(['b.pdf'])).map((f) => f.path),
    ['a.pdf', 'c.pdf'],
  )
})

test('a tick can never reach a file the server ruled out', () => {
  // Even if a ruled-out path never appears in the unticked set, it is not selected.
  const files = [ruledOut('dup.pdf'), ruledOut('notes.xyz')]
  assert.deepEqual(selectedRows(files, new Set()), [])
})

test('Select none unticks every tickable row, Select all clears the set', () => {
  const files = [add('a.pdf'), ruledOut('dup.pdf'), add('b.pdf')]
  const none = tickAll(files, false)
  assert.deepEqual([...none].sort(), ['a.pdf', 'b.pdf'])
  assert.equal(selectedRows(files, none).length, 0)
  assert.equal(tickAll(files, true).size, 0)
})

test('the line under a capped list counts what is still ticked', () => {
  const hidden = [add('x.pdf'), add('y.pdf'), add('z.pdf')]
  assert.equal(hiddenLine([], new Set()), '')
  assert.equal(hiddenLine(hidden, new Set()), 'and 3 more, all ready to add')
  assert.equal(hiddenLine(hidden, new Set(['y.pdf'])), 'and 3 more, 2 ticked')
  assert.equal(hiddenLine(hidden, tickAll(hidden, false)), 'and 3 more, none ticked')
})
