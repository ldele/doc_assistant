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
  filterByQuery,
  keywordFacets,
  orderedUnits,
  remapSelection,
  splitInheritedFamilies,
  splitRareFacets,
  unitDocCounts,
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

// --- PR-2.7: the rare tail (F4), inherited vocabulary (F3), picker search (F3) ------------------ //

const facet = (value: string, selected = false) => ({ value, count: 0, selected, available: true })

test('unitDocCounts counts documents per unit, not occurrences', () => {
  const counts = unitDocCounts(DOCS)
  assert.equal(counts.get('llm'), 2)
  assert.equal(counts.get('retrieval'), 2)
  assert.equal(counts.get('llms'), 2)
})

test('unitDocCounts is family-aware when given the family accessor', () => {
  const counts = unitDocCounts(DOCS, familyUnitsOf(familyCanonicalMap(FAMILIES)))
  assert.equal(counts.get('Large language model'), 3, 'the union of llm + llms')
  assert.equal(counts.get('llm'), undefined)
})

test('F4: a 1-doc keyword is demoted, a 2-doc keyword is not', () => {
  const counts = new Map([
    ['common', 5],
    ['pair', 2],
    ['mathrm', 1],
    ['va1v', 1],
  ])
  const { common, rare } = splitRareFacets(
    [facet('common'), facet('pair'), facet('mathrm'), facet('va1v')],
    counts,
  )
  assert.deepEqual(
    common.map((f) => f.value),
    ['common', 'pair'],
  )
  assert.deepEqual(
    rare.map((f) => f.value),
    ['mathrm', 'va1v'],
  )
})

test('F4: a selected facet is never demoted — it has to stay unselectable-from', () => {
  const counts = new Map([['rare-but-selected', 1], ['common', 4]])
  const { common, rare } = splitRareFacets(
    [facet('rare-but-selected', true), facet('common')],
    counts,
  )
  assert.deepEqual(
    common.map((f) => f.value),
    ['rare-but-selected', 'common'],
  )
  assert.deepEqual(rare, [])
})

test('F4: when every facet is rare, nothing is demoted (honest-empty)', () => {
  const counts = new Map([['a', 1], ['b', 1]])
  const { common, rare } = splitRareFacets([facet('a'), facet('b')], counts)
  assert.equal(common.length, 2, 'a small collection must not collapse to an empty list')
  assert.deepEqual(rare, [])
})

test('F4: an unknown unit counts as 0 docs and is demoted', () => {
  const { rare } = splitRareFacets([facet('ghost'), facet('real')], new Map([['real', 9]]))
  assert.deepEqual(
    rare.map((f) => f.value),
    ['ghost'],
  )
})

test('F3: a 0-member 0-doc concept is inherited vocabulary, not a family', () => {
  const { real, inherited } = splitInheritedFamilies([
    family('Large language model', ['llm', 'llms'], 3),
    family('BERT', [], 0),
    family('Cited but memberless', [], 4),
  ])
  assert.deepEqual(
    real.map((f) => f.canonical),
    ['Large language model', 'Cited but memberless'],
    'a memberless concept that still matches documents still partitions the grid — measured on the real corpus: 12 collapse synonyms, 10 are single-label with docs, only 4 are inert',
  )
  assert.deepEqual(
    inherited.map((f) => f.canonical),
    ['BERT'],
  )
})

test('F3: filterByQuery is a case-insensitive substring match, empty query = identity', () => {
  const items = ['ImageNet', 'imagenette', 'BM25']
  assert.equal(filterByQuery(items, '', (s) => s), items)
  assert.deepEqual(filterByQuery(items, 'imagenet', (s) => s), ['ImageNet', 'imagenette'])
  assert.deepEqual(filterByQuery(items, 'bm', (s) => s), ['BM25'])
  assert.deepEqual(filterByQuery(items, 'zzz', (s) => s), [])
})
