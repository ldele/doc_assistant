// The folder layout of ADR-064, held by a test: components live in `screens/<screen>/` or `shell/`,
// the domain spine (rune modules, pure tested `.ts`, `core/`) in `lib/`, and imports point one way.
// Reads the source tree as text; it imports nothing from the app.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, join, normalize, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const SRC = dirname(fileURLToPath(import.meta.url))
const SPEC = /(?:\bfrom\s*|\bimport\s*\(?\s*)(['"])(\.{1,2}\/[^'"]+)\1/g
const SHARED_FROM_SHELL = new Set(['shell/Icon.svelte', 'shell/Dialog.svelte', 'shell/ConfirmDialog.svelte'])

const posix = (p: string): string => p.split(sep).join('/')

const files: string[] = (readdirSync(SRC, { recursive: true }) as string[])
  .map(posix)
  .filter((f) => /\.(svelte|ts)$/.test(f))

/** Each relative import of `file`, as the path (from src/) of the file it loads. */
function importsOf(file: string): string[] {
  const text = readFileSync(join(SRC, file), 'utf-8')
  const out: string[] = []
  for (const m of text.matchAll(SPEC)) {
    const lineStart = text.lastIndexOf('\n', m.index) + 1
    if (/^\s*(\/\/|\*|\/\*)/.test(text.slice(lineStart, m.index))) continue
    const base = normalize(join(SRC, dirname(file), m[2]))
    const hit = ['', '.ts', '/index.ts'].map((ext) => base + ext).find((p) => existsSync(p))
    out.push(posix(relative(SRC, hit ?? base)))
  }
  return out
}

const under = (dir: string) => (f: string) => f.startsWith(dir + '/')

test('components live in screens/ or shell/, and those folders hold only components', () => {
  assert.deepEqual(files.filter((f) => f.endsWith('.svelte') && under('lib')(f)), [])
  assert.deepEqual(
    files.filter((f) => f.endsWith('.ts') && (under('screens')(f) || under('shell')(f))),
    [],
  )
})

test('shell/ and lib/ never import from screens/', () => {
  const bad = files
    .filter((f) => under('shell')(f) || under('lib')(f))
    .flatMap((f) => importsOf(f).filter(under('screens')).map((t) => `${f} -> ${t}`))
  assert.deepEqual(bad, [])
})

test('lib/ never imports a component', () => {
  const bad = files
    .filter(under('lib'))
    .flatMap((f) => importsOf(f).filter((t) => t.endsWith('.svelte')).map((t) => `${f} -> ${t}`))
  assert.deepEqual(bad, [])
})

test('a screen takes only Icon, Dialog and ConfirmDialog from shell/', () => {
  const bad = files
    .filter(under('screens'))
    .flatMap((f) =>
      importsOf(f)
        .filter((t) => under('shell')(t) && t.endsWith('.svelte') && !SHARED_FROM_SHELL.has(t))
        .map((t) => `${f} -> ${t}`),
    )
  assert.deepEqual(bad, [])
})

test('every relative import resolves to a file', () => {
  const bad = files.flatMap((f) =>
    importsOf(f)
      .filter((t) => !existsSync(join(SRC, t)))
      .map((t) => `${f} -> ${t}`),
  )
  assert.deepEqual(bad, [])
})
