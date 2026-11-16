import { test } from 'node:test'
import assert from 'node:assert/strict'

import { isWebLink, linkAllowed, normaliseLink } from './links.ts'

test('a link from a passage stays clickable', () => {
  assert.equal(linkAllowed('https://github.com/org/repo', ['https://github.com/org/repo']), true)
})

test('a link no passage carries is not clickable', () => {
  assert.equal(linkAllowed('http://unlock.example/verify', ['https://github.com/org/repo']), false)
})

test('while an answer streams, no web link is clickable', () => {
  assert.equal(linkAllowed('https://a.test', []), false)
})

test('text the app wrote keeps its links', () => {
  assert.equal(linkAllowed('https://a.test', null), true)
})

test('a non-web href is left to the sanitiser', () => {
  assert.equal(isWebLink('mailto:x@y.test'), false)
  assert.equal(linkAllowed('#section', []), true)
})

test('sentence punctuation does not make a sourced link look new', () => {
  assert.equal(normaliseLink('https://a.test/x.'), 'https://a.test/x')
  assert.equal(linkAllowed('https://a.test/x.', ['https://a.test/x']), true)
})
