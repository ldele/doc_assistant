// The highlight token against each theme's page and text, read from app.css itself, so a change
// of value is judged by its ratio and not by eye (0.8.0 walk, item 6: a weak claim and the cited
// passage were near invisible in the dark theme, --warn-bg at 1.2:1 off the page).

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { contrastRatio } from './lib/core/contrast.ts'

const css = readFileSync(new URL('./app.css', import.meta.url), 'utf-8')

/** The tokens of the first block whose selector starts with `selector`. */
function tokens(selector: string): Record<string, string> {
  const at = css.indexOf(selector)
  assert.ok(at >= 0, `no ${selector} block in app.css`)
  const body = css.slice(css.indexOf('{', at) + 1, css.indexOf('}', at))
  const out: Record<string, string> = {}
  for (const m of body.matchAll(/(--[\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)) out[m[1]] = m[2]
  return out
}

const THEMES = [
  { name: 'light (default)', selector: ':root {', dark: false },
  { name: 'light (chosen)', selector: ":root[data-theme='light']", dark: false },
  { name: 'dark (chosen)', selector: ":root[data-theme='dark']", dark: true },
  { name: 'dark (from the system)', selector: ':root:not([data-theme])', dark: true },
]

for (const t of THEMES) {
  test(`${t.name}: text on a highlight stays readable`, () => {
    const v = tokens(t.selector)
    assert.ok(v['--mark-bg'] && v['--fg'], 'the block defines --mark-bg and --fg')
    assert.ok(contrastRatio(v['--fg'], v['--mark-bg']) >= 7, `--fg on --mark-bg below 7:1`)
  })
}

// On a dark page the eye reads a fill by its lightness alone; on white a pale amber carries its
// hue, so the off-the-page bar is asked of the dark themes only.
for (const t of THEMES.filter((x) => x.dark)) {
  test(`${t.name}: a highlight stands off the page`, () => {
    const v = tokens(t.selector)
    assert.ok(contrastRatio(v['--mark-bg'], v['--bg']) >= 1.8, `--mark-bg on --bg below 1.8:1`)
  })
}

test('contrastRatio matches the WCAG end points', () => {
  assert.equal(Math.round(contrastRatio('#000000', '#ffffff')), 21)
  assert.equal(contrastRatio('#777777', '#777777'), 1)
})
