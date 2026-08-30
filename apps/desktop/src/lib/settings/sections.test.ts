// Tests for the settings drawer's category model.
//
// Two things here are worth guarding rather than eyeballing. The landing category is the one
// decision that can strand a first-run user (ADR-034: on a fresh install the setup checklist is
// the only category that matters), and the icon names are the one field whose typo `svelte-check`
// catches only if the literal types survive `as const`.

import { test } from 'node:test'
import assert from 'node:assert/strict'

import { SETTINGS_SECTIONS, initialSection, sectionBadge } from './sections.ts'

test('a fresh install lands on the setup checklist', () => {
  assert.equal(initialSection(2), 'setup')
  assert.equal(initialSection(1), 'setup')
})

test('a finished install lands on Documents, not on a checklist with nothing left to do', () => {
  assert.equal(initialSection(0), 'documents')
})

test('the setup badge counts outstanding steps and disappears when there are none', () => {
  assert.equal(sectionBadge('setup', 3), '3')
  assert.equal(sectionBadge('setup', 0), null)
})

test('no other category carries a badge', () => {
  for (const s of SETTINGS_SECTIONS) {
    if (s.id === 'setup') continue
    assert.equal(sectionBadge(s.id, 3), null, `${s.id} should not badge`)
  }
})

test('every category is complete, unique, and reachable as a landing page', () => {
  const ids = SETTINGS_SECTIONS.map((s) => s.id)
  assert.equal(new Set(ids).size, ids.length, 'duplicate category id')
  for (const s of SETTINGS_SECTIONS) {
    assert.ok(s.label.length > 0, `${s.id} has no label`)
    assert.ok(s.icon.length > 0, `${s.id} has no icon`)
    // The blurb is the category's own explanation of itself; an empty one silently turns the
    // rail back into a list of bare words, which is the thing the categories replaced.
    assert.ok(s.blurb.length > 0, `${s.id} has no blurb`)
  }
  // Both landing choices must name a category that actually exists.
  assert.ok(ids.includes(initialSection(0)))
  assert.ok(ids.includes(initialSection(1)))
})
