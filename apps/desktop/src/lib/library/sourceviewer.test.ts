// Tests for the source viewer's decisions (ROADMAP 18, ADR-050).
//
// `sourceMode` is the load-bearing one. Three different causes stop the pane rendering pages —
// no document, no file, no pages — and each is a different sentence to the reader. Collapsing
// them into one "unavailable" is precisely the failure the row exists to prevent, so the tests
// keep the arms apart rather than checking that something-went-wrong was detected.

import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  BASE_DPI,
  DPI_LADDER,
  MAX_ZOOM,
  MIN_ZOOM,
  canStep,
  clampPage,
  clampSplit,
  clampZoom,
  fitPageZoom,
  openingPage,
  pageLabel,
  pageWidthPt,
  renderDpi,
  sourceMode,
  unavailableMessage,
  zoomLabel,
  zoomStep,
} from './sourceviewer.ts'
import type { SourceDocumentView } from '../core/types/library.ts'

const view = (over: Partial<SourceDocumentView> = {}): SourceDocumentView => ({
  document_id: 'doc-1',
  filename: 'paper.pdf',
  format: 'pdf',
  page_count: 12,
  available: true,
  pageable: true,
  path: 'C:/papers/paper.pdf',
  reason: null,
  ...over,
})

// --- sourceMode: the four causes stay four ----------------------------------------------------

test('a reachable pdf renders pages', () => {
  assert.equal(sourceMode(view()), 'pages')
})

test('nothing fetched yet is loading, not unknown', () => {
  assert.equal(sourceMode(undefined), 'loading')
})

test('a 404 is unknown — a different thing from a missing file', () => {
  assert.equal(sourceMode(null), 'unknown')
})

test('a document whose file moved is unavailable, not unknown', () => {
  assert.equal(sourceMode(view({ available: false, path: null })), 'unavailable')
})

test('a format without pages is text-only, not a failure', () => {
  assert.equal(sourceMode(view({ format: 'epub', pageable: false })), 'text-only')
})

test('unavailable beats text-only — the file is the first thing missing', () => {
  const v = view({ available: false, pageable: false, format: 'epub' })
  assert.equal(sourceMode(v), 'unavailable')
})

// --- the sentences ----------------------------------------------------------------------------

test("an unavailable document shows the server's own reason, verbatim", () => {
  const reason = 'The drive holding this document is not connected (E:\\papers).'
  assert.equal(unavailableMessage(view({ available: false, reason })), reason)
})

test('a missing reason still yields a sentence rather than an empty pane', () => {
  const msg = unavailableMessage(view({ available: false, reason: null }))
  assert.ok(msg.length > 0)
})

test('a text-only document names its format, without claiming a direction', () => {
  const msg = unavailableMessage(view({ format: 'epub', pageable: false }))
  assert.match(msg, /EPUB/)
  assert.match(msg, /no pages to show/)
})

test('a renderable document has no message to show', () => {
  assert.equal(unavailableMessage(view()), '')
})

// --- page arithmetic --------------------------------------------------------------------------

test('a page inside the document is left alone', () => {
  assert.equal(clampPage(5, 12), 5)
})

test('page 0 and below floor at 1', () => {
  assert.equal(clampPage(0, 12), 1)
  assert.equal(clampPage(-4, 12), 1)
})

test('a page past the end clamps to the last page', () => {
  assert.equal(clampPage(99, 12), 12)
})

test('an unknown page count clamps at the floor only — no invented ceiling', () => {
  assert.equal(clampPage(99, null), 99)
  assert.equal(clampPage(0, null), 1)
})

test('a non-integer page is floored, not rejected', () => {
  assert.equal(clampPage(3.7, 12), 3)
  assert.equal(clampPage(Number.NaN, 12), 1)
})

test('stepping stops at both ends', () => {
  assert.equal(canStep(1, 12, -1), false)
  assert.equal(canStep(1, 12, 1), true)
  assert.equal(canStep(12, 12, 1), false)
  assert.equal(canStep(12, 12, -1), true)
})

test('an unknown page count never blocks forward stepping', () => {
  assert.equal(canStep(400, null, 1), true)
  assert.equal(canStep(1, null, -1), false)
})

test('the label omits a total it does not have', () => {
  assert.equal(pageLabel(4, 12), 'Page 4 of 12')
  assert.equal(pageLabel(4, null), 'Page 4')
  assert.equal(pageLabel(1, 0), 'Page 1')
})

// --- opening from a citation ------------------------------------------------------------------

test('a located citation opens on its page and says so', () => {
  assert.deepEqual(openingPage(7, 12), { page: 7, located: true })
})

test('an unplaceable citation opens at page 1 WITHOUT claiming to be located', () => {
  // The distinction the flag exists for: this is visually identical to a real page 1.
  assert.deepEqual(openingPage(null, 12), { page: 1, located: false })
})

test('a cited page past the end is clamped but still counts as located', () => {
  assert.deepEqual(openingPage(99, 12), { page: 12, located: true })
})

// --- zoom ---------------------------------------------------------------------------------

test('zoom is clamped to a usable range', () => {
  assert.equal(clampZoom(0.01), MIN_ZOOM)
  assert.equal(clampZoom(99), MAX_ZOOM)
  assert.equal(clampZoom(1.5), 1.5)
})

test('a nonsense zoom falls back to 1 rather than NaN-ing the layout', () => {
  // Infinity is not a big zoom, it is a corrupted one — most likely a bad persisted value. The
  // safe reading is the default, not the maximum, which would open the pane at 600%.
  assert.equal(clampZoom(Number.NaN), 1)
  assert.equal(clampZoom(Number.POSITIVE_INFINITY), 1)
  assert.equal(clampZoom(Number.NEGATIVE_INFINITY), 1)
})

test('steps are geometric, so zooming feels the same at every scale', () => {
  const a = zoomStep(1, 1) / 1
  const b = zoomStep(4, 1) / 4
  assert.ok(Math.abs(a - b) < 0.01, `${a} vs ${b}`)
})

test('stepping in then out returns to where it started', () => {
  assert.equal(zoomStep(zoomStep(1, 1), -1), 1)
})

test('stepping stops at the ends instead of running away', () => {
  assert.equal(zoomStep(MAX_ZOOM, 1), MAX_ZOOM)
  assert.equal(zoomStep(MIN_ZOOM, -1), MIN_ZOOM)
})

test('a step never produces a float the label would render badly', () => {
  let z = 1
  for (let i = 0; i < 8; i++) z = zoomStep(z, 1)
  assert.equal(z, Math.round(z * 100) / 100)
})

// --- fit page -----------------------------------------------------------------------------

test('fit-page zooms out enough to show a tall page whole', () => {
  // A 1.57 page in a 443x519 box — the measured Cajal case.
  const z = fitPageZoom(443, 519, 1.57)
  assert.ok(z < 1, `expected to zoom out, got ${z}`)
  assert.ok(Math.abs(443 * z * 1.57 - 519) < 1, 'the fitted height should equal the box height')
})

test('fit-page never magnifies a page that already fits', () => {
  // A wide, short page in a tall box: fitting by arithmetic alone would blow it up past the pane.
  assert.equal(fitPageZoom(400, 2000, 1.29), 1)
})

test('fit-page is 1 until the box and aspect are known', () => {
  assert.equal(fitPageZoom(0, 500, 1.29), 1)
  assert.equal(fitPageZoom(400, 0, 1.29), 1)
  assert.equal(fitPageZoom(400, 500, 0), 1)
})

// --- resolution ---------------------------------------------------------------------------

test('a page width in points is recovered exactly from a render of known dpi', () => {
  // US Letter is 612pt wide; at 110 dpi that is 935 px.
  assert.ok(Math.abs(pageWidthPt(935, 110) - 612) < 1)
})

test('an unknown render tells us nothing rather than a wrong number', () => {
  assert.equal(pageWidthPt(0, 110), 0)
  assert.equal(pageWidthPt(935, 0), 0)
})

test('the requested dpi climbs the ladder as the page is drawn bigger', () => {
  const letter = 612
  const small = renderDpi(400, letter, 1)
  const large = renderDpi(1600, letter, 1)
  assert.ok(large > small, `${large} should exceed ${small}`)
  assert.ok(DPI_LADDER.includes(small as never) && DPI_LADDER.includes(large as never))
})

test('the dpi snaps UP — never below what is needed, which would be blurrier', () => {
  const letter = 612
  const dpi = renderDpi(500, letter, 1)
  assert.ok((letter * dpi) / 72 >= 500, 'the render must be at least as wide as it is drawn')
})

test('a 2x display asks for more resolution once the page is drawn large', () => {
  const letter = 612
  assert.ok(renderDpi(900, letter, 2) > renderDpi(900, letter, 1))
})

test('...but NOT at the size the pane actually opens at — the default already covers 2x', () => {
  // 433 CSS px is the measured pane width. This pins the claim the code comment makes, because
  // an earlier draft of that comment asserted the opposite and nothing would have caught it.
  const letter = 612
  assert.equal(renderDpi(433, letter, 1), BASE_DPI)
  assert.equal(renderDpi(433, letter, 2), BASE_DPI)
})

test('the ladder is the ceiling — an enormous zoom does not ask for an enormous render', () => {
  assert.equal(renderDpi(100000, 612, 3), DPI_LADDER[DPI_LADDER.length - 1])
})

test('nothing known yet means the default dpi, not a division by zero', () => {
  assert.equal(renderDpi(0, 612, 1), BASE_DPI)
  assert.equal(renderDpi(400, 0, 1), BASE_DPI)
})

test('the zoom label is rounded for a person', () => {
  assert.equal(zoomLabel(1), '100%')
  assert.equal(zoomLabel(0.4297), '43%')
})

// --- the split ----------------------------------------------------------------------------

test('the split cannot be dragged to hide either side', () => {
  assert.equal(clampSplit(0), 0.25)
  assert.equal(clampSplit(1), 0.75)
  assert.equal(clampSplit(0.5), 0.5)
})

test('a nonsense split falls back to the default rather than collapsing the pane', () => {
  assert.equal(clampSplit(Number.NaN), 0.46)
})
