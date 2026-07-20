// Tests for the family grouping layer (PR-2.5). Until now this layer was **entirely
// unexercised**: `familyCanonicalMap` / `familyUnitsOf` / `facetFilter` composition is what makes
// a family behave as one facet unit, and D5 (a live selection stranded by a family write) lived
// squarely inside it.
//
// Runner: node's built-in `node:test` with native TypeScript stripping — no new dependency, no
// config. `npm test` from apps/desktop.
import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  facetFilter,
  familyCanonicalMap,
  familyUnitsOf,
  keywordFacets,
  orderedUnits,
  remapSelection,
} from './library.ts'
import type { KeywordFamily, LibraryDocument } from './types.ts'

const family = (canonical: string, aliases: string[], doc_count = 0): KeywordFamily =>
  ({ id: canonical, canonical, aliases, doc_count }) as KeywordFamily

const doc = (id: string, keywords: string[]): LibraryDocument =>
  ({ id, filename: `${id}.pdf`, keywords }) as unknown as LibraryDocument

const DOCS = [
  doc('a', ['llm', 'retrieval']),
  doc('b', ['llms']),
  doc('c', ['llm', 'llms']),
  doc('d', ['retrieval']),
]
const FAMILIES = [family('Large language model', ['llm', 'llms'])]

// --- the grouping layer ---------------------------------------------------------------------- //

test('familyCanonicalMap maps every member *and* the canonical, case-insensitively', () => {
  const map = familyCanonicalMap(FAMILIES)
  assert.equal(map.get('llm'), 'Large language model')
  assert.equal(map.get('llms'), 'Large language model')
  assert.equal(map.get('large language model'), 'Large language model')
  assert.equal(map.get('retrieval'), undefined)
})

test('familyUnitsOf collapses members to one unit and dedupes', () => {
  const unitsOf = familyUnitsOf(familyCanonicalMap(FAMILIES))
  assert.deepEqual(unitsOf(DOCS[2]!).sort(), ['Large language model'])
  assert.deepEqual(unitsOf(DOCS[0]!).sort(), ['Large language model', 'retrieval'])
})

test('an empty family map leaves the raw keywords untouched (the no-families path)', () => {
  const unitsOf = familyUnitsOf(familyCanonicalMap([]))
  for (const d of DOCS) assert.deepEqual(unitsOf(d).sort(), [...d.keywords].sort())
})

test('facetFilter over family units returns the union of the members', () => {
  const unitsOf = familyUnitsOf(familyCanonicalMap(FAMILIES))
  const hits = facetFilter(DOCS, ['Large language model'], unitsOf)
  assert.deepEqual(
    hits.map((d) => d.id),
    ['a', 'b', 'c'],
  )
})

test('AND semantics still narrow across a family and a plain keyword', () => {
  const unitsOf = familyUnitsOf(familyCanonicalMap(FAMILIES))
  const hits = facetFilter(DOCS, ['Large language model', 'retrieval'], unitsOf)
  assert.deepEqual(
    hits.map((d) => d.id),
    ['a'],
  )
})

test('the family is one chip whose count is the union, not one chip per member', () => {
  const unitsOf = familyUnitsOf(familyCanonicalMap(FAMILIES))
  const facets = keywordFacets(DOCS, [], unitsOf)
  assert.deepEqual(
    facets.map((f) => f.value).sort(),
    ['Large language model', 'retrieval'],
  )
  assert.equal(facets.find((f) => f.value === 'Large language model')?.count, 3)
})

// --- D5: a live selection survives a family write ---------------------------------------------- //

test('D5: grouping a selected keyword re-points the selection at the new family', () => {
  const after = remapSelection(['llm'], familyCanonicalMap(FAMILIES), DOCS)
  assert.deepEqual(after, ['Large language model'])
  assert.equal(facetFilter(DOCS, after, familyUnitsOf(familyCanonicalMap(FAMILIES))).length, 3)
})

test('D5: selecting two members of the same family collapses to one unit, not a dead AND', () => {
  const after = remapSelection(['llm', 'llms'], familyCanonicalMap(FAMILIES), DOCS)
  assert.deepEqual(after, ['Large language model'])
})

test('D5: deleting the family drops the now-nonexistent canonical from the selection', () => {
  const after = remapSelection(['Large language model'], familyCanonicalMap([]), DOCS)
  assert.deepEqual(after, [])
})

test('D5: an unaffected selection is left exactly as it was', () => {
  assert.deepEqual(remapSelection(['retrieval'], familyCanonicalMap(FAMILIES), DOCS), ['retrieval'])
  assert.deepEqual(remapSelection([], familyCanonicalMap(FAMILIES), DOCS), [])
})

// --- PR-2.6 / D6: tiles render family *units*, so a family selection can actually match --------- //

test('D6: a family selection matches nothing against raw keywords — the shipped bug', () => {
  const tile = doc('c', ['llm', 'llms'])
  const selection = ['Large language model']

  // What LibraryGrid did: `activeKeywords.includes(rawKeyword)`.
  assert.equal(
    tile.keywords.some((k) => selection.includes(k)),
    false,
  )
  // Same tile through the family units the grid now renders.
  const units = familyUnitsOf(familyCanonicalMap(FAMILIES))(tile)
  assert.deepEqual(units, ['Large language model'])
  assert.equal(
    units.some((u) => selection.includes(u)),
    true,
  )
})

test('D6: the selected unit is floated to the front so the `+N` cap cannot hide it', () => {
  const units = ['alpha', 'beta', 'Large language model', 'gamma']
  assert.deepEqual(orderedUnits(units, ['Large language model']), [
    'Large language model',
    'alpha',
    'beta',
    'gamma',
  ])
})

test('orderedUnits is a no-op without a selection, or when nothing matches', () => {
  const units = ['alpha', 'beta']
  assert.equal(orderedUnits(units, []), units, 'the no-facets path returns the same array')
  assert.deepEqual(orderedUnits(units, ['gamma']), units)
})

test('orderedUnits keeps every unit — it orders, it never filters', () => {
  const units = ['alpha', 'beta', 'gamma']
  assert.deepEqual([...orderedUnits(units, ['beta'])].sort(), [...units].sort())
})

test('a family collapses several raw keywords into one chip, freeing the tile chip budget', () => {
  const unitsOf = familyUnitsOf(familyCanonicalMap([family('Connectome', ['connectomes', 'connectomics'])]))
  const tile = doc('x', ['connectome', 'connectomes', 'connectomics', 'mouse'])
  assert.deepEqual(unitsOf(tile).sort(), ['Connectome', 'mouse'])
})
