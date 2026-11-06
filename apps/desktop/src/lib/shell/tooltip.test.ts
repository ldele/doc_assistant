// The shared tooltip's pure parts (tooltip.ts): what text it shows and where it goes.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import { EDGE, GAP, hoverText, placeTooltip } from './tooltip.ts'

const VIEW = { width: 1280, height: 800 }
const TIP = { width: 200, height: 40 }

test('hoverText: no text is null, whitespace collapses', () => {
  assert.equal(hoverText(undefined), null)
  assert.equal(hoverText(null), null)
  assert.equal(hoverText(''), null)
  assert.equal(hoverText('   \n  '), null)
  assert.equal(hoverText(42), null)
  assert.equal(hoverText('  Leave\n   select   mode '), 'Leave select mode')
})

test('placeTooltip: centred under the button when it fits', () => {
  const p = placeTooltip({ left: 500, top: 100, width: 80, height: 30 }, TIP, VIEW)
  assert.equal(p.side, 'below')
  assert.equal(p.top, 100 + 30 + GAP)
  assert.equal(p.left, 500 + 40 - 100)
})

test('placeTooltip: above the button near the bottom edge', () => {
  const p = placeTooltip({ left: 500, top: 760, width: 80, height: 30 }, TIP, VIEW)
  assert.equal(p.side, 'above')
  assert.equal(p.top, 760 - GAP - 40)
})

test('placeTooltip: below when neither side fits and below has more room', () => {
  const tall = { width: 200, height: 500 }
  const p = placeTooltip({ left: 500, top: 200, width: 80, height: 30 }, tall, { width: 1280, height: 600 })
  assert.equal(p.side, 'below')
  // Kept inside the window, even though that means covering part of the button's row.
  assert.equal(p.top, 600 - 500 - EDGE)
})

test('placeTooltip: kept EDGE px inside the window at 375 px', () => {
  const narrow = { width: 375, height: 812 }
  const left = placeTooltip({ left: 4, top: 10, width: 32, height: 32 }, TIP, narrow)
  assert.equal(left.left, EDGE)
  const right = placeTooltip({ left: 340, top: 10, width: 32, height: 32 }, TIP, narrow)
  assert.equal(right.left, 375 - 200 - EDGE)
  const wide = placeTooltip({ left: 100, top: 10, width: 32, height: 32 }, { width: 500, height: 40 }, narrow)
  assert.equal(wide.left, EDGE)
})
