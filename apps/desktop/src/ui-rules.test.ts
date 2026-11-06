// The style rules of ADR-063, held by three checks (its *Enforced by*), each with an allow-list in
// `ui-rules.allow.ts` whose every entry gives its reason:
//   1. a component defines its own `.primary`, `.ghost` or `.danger` (the roles live in app.css);
//   2. a `<button>` has no hover text (`use:tooltip`, lib/shell/tooltip.ts);
//   3. a literal colour (hex, rgb(), hsl()) appears in a component's styles (tokens live in app.css).
// Plus one with no allow-list: a `<button>` never carries the native `title` (the tooltip replaces
// it, WCAG 2.2 SC 1.4.13). Reads the source tree as text; it imports nothing from the app.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, join, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  type Allowed,
  BUTTONS_WITHOUT_HOVER_TEXT,
  LITERAL_COLOURS,
  LOCAL_BUTTON_CLASSES,
} from './ui-rules.allow.ts'

const SRC = dirname(fileURLToPath(import.meta.url))

const components: string[] = (readdirSync(SRC, { recursive: true }) as string[])
  .map((f) => f.split(sep).join('/'))
  .filter((f) => f.endsWith('.svelte'))
  .sort()

const source = new Map(components.map((f) => [f, readFileSync(join(SRC, f), 'utf-8')]))

/** Same length, same line breaks, nothing else: blanking keeps every later index and line. */
const blank = (s: string): string => s.replace(/[^\n]/g, ' ')

/** The CSS of a component's `<style>` blocks, comments removed. */
export function styleOf(text: string): string {
  return [...text.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/g)]
    .map((m) => m[1].replace(/\/\*[\s\S]*?\*\//g, ''))
    .join('\n')
}

/** The markup: scripts, styles and HTML comments blanked out. */
export function markupOf(text: string): string {
  return text
    .replace(/<script\b[\s\S]*?<\/script>/g, blank)
    .replace(/<style\b[\s\S]*?<\/style>/g, blank)
    .replace(/<!--[\s\S]*?-->/g, blank)
}

/**
 * Index of the `>` that closes the tag opened at `from`, skipping `>` inside quoted attribute
 * values and `{…}` expressions (an arrow function's `=>`, a comparison), or -1.
 */
export function tagEnd(src: string, from: number): number {
  const stack: string[] = ['tag']
  for (let i = from; i < src.length; i++) {
    const c = src[i]
    const top = stack[stack.length - 1]
    if (top === 'tag') {
      if (c === '>') return i
      if (c === '"' || c === "'") stack.push('a' + c)
      else if (c === '{') stack.push('{')
    } else if (top === 'a"' || top === "a'") {
      if (c === top[1]) stack.pop()
      else if (c === '{') stack.push('{')
    } else if (top === '{') {
      if (c === '}') stack.pop()
      else if (c === '{') stack.push('{')
      else if (c === '"' || c === "'" || c === '`') stack.push('s' + c)
    } else if (c === '\\') {
      i++
    } else if (c === top[1]) {
      stack.pop()
    } else if (top === 's`' && c === '$' && src[i + 1] === '{') {
      stack.push('{')
      i++
    }
  }
  return -1
}

/**
 * The attribute names of one tag's source (`<button …>`). Values are dropped, so a `title` inside
 * an expression is not an attribute; `{name}` shorthand counts as the attribute `name`.
 */
export function attributeNames(tag: string): string[] {
  let flat = ''
  let i = tag.search(/\s/)
  if (i < 0) return []
  while (i < tag.length) {
    const c = tag[i]
    if (c === '{') {
      const close = closeOf(tag, i)
      const inner = tag.slice(i + 1, close)
      const shorthand = /^\s*[A-Za-z_$][\w$]*\s*$/.test(inner) && /\s$/.test(flat)
      flat += shorthand ? `${inner.trim()} ` : '""'
      i = close + 1
    } else if (c === '"' || c === "'") {
      i = quoteEnd(tag, i) + 1
      flat += '""'
    } else {
      flat += c
      i++
    }
  }
  return flat
    .replace(/\/?>\s*$/, '')
    .split(/\s+/)
    .map((t) => t.split('=')[0])
    .filter((t) => t !== '' && t !== '""')
}

/** Index of the quote closing the attribute value opened at `open`; `{…}` inside it is skipped. */
function quoteEnd(src: string, open: number): number {
  for (let i = open + 1; i < src.length; i++) {
    if (src[i] === '{') i = closeOf(src, i)
    else if (src[i] === src[open]) return i
  }
  return src.length - 1
}

/** Index of the `}` matching the `{` at `open`, skipping strings inside the expression. */
function closeOf(src: string, open: number): number {
  const stack: string[] = ['{']
  for (let i = open + 1; i < src.length; i++) {
    const c = src[i]
    const top = stack[stack.length - 1]
    if (top === '{') {
      if (c === '}') {
        stack.pop()
        if (stack.length === 0) return i
      } else if (c === '{') stack.push('{')
      else if (c === '"' || c === "'" || c === '`') stack.push('s' + c)
    } else if (c === '\\') {
      i++
    } else if (c === top[1]) {
      stack.pop()
    } else if (top === 's`' && c === '$' && src[i + 1] === '{') {
      stack.push('{')
      i++
    }
  }
  return src.length - 1
}

interface ButtonTag {
  line: number
  attrs: string[]
}

/** Every `<button …>` opening tag in a component's markup, with its line and attribute names. */
export function buttonsOf(text: string): ButtonTag[] {
  const markup = markupOf(text)
  const out: ButtonTag[] = []
  for (const m of markup.matchAll(/<button(?=[\s>/])/g)) {
    const end = tagEnd(markup, m.index + 1)
    const tag = markup.slice(m.index, end < 0 ? undefined : end + 1)
    out.push({ line: markup.slice(0, m.index).split('\n').length, attrs: attributeNames(tag) })
  }
  return out
}

const LOCAL_CLASS = /\.(primary|ghost|danger)(?![\w-])/g
const LITERAL_COLOUR = /#[0-9a-fA-F]{3,8}(?![\w-])|\b(?:rgba?|hsla?)\(/g
/** The named colours a component is likely to reach for (CSS has 148; these are the plausible
 *  ones). Matched only in a declaration's value and never inside a name: `var(--lavender)` is a
 *  token, not a colour. */
const NAMED_COLOUR =
  /(?<![\w-])(?:black|white|gr[ae]y|silver|red|maroon|crimson|tomato|coral|salmon|orange|gold|yellow|olive|khaki|beige|ivory|tan|brown|green|lime|teal|turquoise|aqua|cyan|blue|navy|indigo|purple|violet|lavender|fuchsia|magenta|pink)(?![\w-])/g

function literalColours(text: string): number {
  const css = styleOf(text)
  const values = [...css.matchAll(/:([^;{}]*)/g)].map((m) => m[1]).join(';')
  return [...css.matchAll(LITERAL_COLOUR)].length + [...values.matchAll(NAMED_COLOUR)].length
}

const measure = {
  localButtonClasses: (text: string): number =>
    new Set([...styleOf(text).matchAll(LOCAL_CLASS)].map((m) => m[1])).size,
  buttonsWithoutHoverText: (text: string): number =>
    buttonsOf(text).filter((b) => !b.attrs.includes('use:tooltip')).length,
  literalColours,
}

function counts(of: (text: string) => number): Map<string, number> {
  const out = new Map<string, number>()
  for (const [f, text] of source) {
    const n = of(text)
    if (n > 0) out.set(f, n)
  }
  return out
}

/**
 * A ratchet: no file may break the rule more often than its allow-list entry says, and an entry
 * that is now too high must come down, so the list stays the truth.
 */
function holds(rule: string, measured: Map<string, number>, allowed: Record<string, Allowed>): void {
  const over = [...measured]
    .filter(([f, n]) => n > (allowed[f]?.count ?? 0))
    .map(([f, n]) => `${f}: ${n}, allowed ${allowed[f]?.count ?? 0}`)
  assert.deepEqual(over, [], `${rule}: more than its allow-list in ui-rules.allow.ts`)
  const stale = Object.entries(allowed)
    .filter(([f, a]) => (measured.get(f) ?? 0) < a.count)
    .map(([f, a]) => `${f}: ${measured.get(f) ?? 0}, listed ${a.count}`)
  assert.deepEqual(stale, [], `${rule}: fewer than listed; lower (or remove) the entry`)
}

function wellFormed(allowed: Record<string, Allowed>): void {
  for (const [f, a] of Object.entries(allowed)) {
    assert.ok(existsSync(join(SRC, f)), `allow-list entry for a file that does not exist: ${f}`)
    assert.ok(a.why.trim().length > 0, `allow-list entry without a reason: ${f}`)
    assert.ok(Number.isInteger(a.count) && a.count > 0, `allow-list count must be at least 1: ${f}`)
  }
}

test('1. no component defines its own .primary / .ghost / .danger (beyond its allow-list)', () => {
  wellFormed(LOCAL_BUTTON_CLASSES)
  holds('local button classes', counts(measure.localButtonClasses), LOCAL_BUTTON_CLASSES)
})

test('2. every <button> has hover text, use:tooltip (beyond its allow-list)', () => {
  wellFormed(BUTTONS_WITHOUT_HOVER_TEXT)
  holds('buttons without hover text', counts(measure.buttonsWithoutHoverText), BUTTONS_WITHOUT_HOVER_TEXT)
})

test('3. no literal colour in a component style (beyond its allow-list)', () => {
  wellFormed(LITERAL_COLOURS)
  holds('literal colours', counts(measure.literalColours), LITERAL_COLOURS)
})

test('a <button> never carries the native title (the tooltip replaces it)', () => {
  const bad = [...source].flatMap(([f, text]) =>
    buttonsOf(text)
      .filter((b) => b.attrs.includes('title'))
      .map((b) => `${f}:${b.line}`),
  )
  assert.deepEqual(bad, [])
})

test('the scanner reads a tag past an arrow function and a quoted ">"', () => {
  const text = `<button class="a" onclick={() => go(1 > 0)} aria-label="x > y" use:tooltip={'t'} {disabled}>+</button>`
  const [b] = buttonsOf(text)
  assert.deepEqual(b.attrs, ['class', 'onclick', 'aria-label', 'use:tooltip', 'disabled'])
  assert.deepEqual(
    buttonsOf('<script>const s = "<button title=x>"</script>\n<!-- <button> -->\n<button\n  title="t">x</button>'),
    [{ line: 3, attrs: ['title'] }],
  )
  assert.equal(measure.literalColours('<style>/* #fff */ a { color: #fff; background: rgb(0 0 0); }</style>'), 2)
  assert.equal(
    measure.literalColours('<style>a:hover { color: var(--lavender); background: color-mix(in srgb, green 25%, transparent); }</style>'),
    1,
  )
  assert.equal(measure.localButtonClasses('<style>.ghost, .ghost.sm, .btn-danger, .primary-x {}</style>'), 1)
})
