// Source-viewer logic (ROADMAP 18, ADR-050) — pure, so it is testable under `node:test`.
//
// The component owns the pane, the fetch and the `<img>`; everything here is the two decisions
// worth guarding: *what the pane should show at all*, and *which page it is on*.
//
// The first exists because "cannot show this" has three different causes with three different
// sentences — no file, no pages, no document — and collapsing them into one "unavailable" is
// exactly the dishonest degradation the row was written to avoid.

import type { SourceDocumentView } from '../core/types'

/** What the pane renders. One arm per cause, because each says something different to the user. */
export type SourceMode =
  | 'loading' // nothing asked yet
  | 'pages' // a reachable, pageable document: render page images
  | 'text-only' // a reachable document whose format has no pages (ADR-050 D3)
  | 'unavailable' // the app knows the document; the bytes are not reachable (D4)
  | 'unknown' // no such document

/** Which arm applies. `null` is the 404 the API returns for an unknown document — kept distinct
 *  from `available: false`, which is a document the app knows everything else about. */
export function sourceMode(view: SourceDocumentView | null | undefined): SourceMode {
  if (view === undefined) return 'loading'
  if (view === null) return 'unknown'
  if (!view.available) return 'unavailable'
  return view.pageable ? 'pages' : 'text-only'
}

/** The sentence the pane shows when it cannot render pages.
 *
 * For `unavailable` this is the server's own `reason`, never a re-phrasing: it names the path, and
 * whether the *drive* or the *file* went missing is a distinction only the server can draw. */
export function unavailableMessage(view: SourceDocumentView | null | undefined): string {
  const mode = sourceMode(view)
  if (mode === 'unknown') return 'This document is not in the library.'
  if (mode === 'unavailable') {
    return view?.reason ?? 'The file for this document is not reachable right now.'
  }
  if (mode === 'text-only') {
    const fmt = (view?.format ?? '').toUpperCase()
    // No "below" or "beside": Chunks is beside this pane in the split layout and above it in
    // the stacked one, so any direction here is wrong half the time.
    return `${fmt || 'This format'} has no pages to show.`
  }
  return ''
}

/** Clamp a page into the document, 1-based.
 *
 * `page_count` can be null (never extracted, or a format without pages), which is not the same as
 * zero: with no known ceiling the only honest clamp is the floor, so the pane shows page 1 rather
 * than refusing. A document that genuinely has 0 pages also floors at 1 — asking the server for it
 * yields a 404 with the real range, which is a better error than one invented here. */
export function clampPage(page: number, pageCount: number | null): number {
  const n = Number.isFinite(page) ? Math.floor(page) : 1
  if (n < 1) return 1
  if (pageCount !== null && pageCount > 0 && n > pageCount) return pageCount
  return n
}

/** Whether the pane can step in a direction — what greys out the arrows. */
export function canStep(page: number, pageCount: number | null, delta: number): boolean {
  const target = page + delta
  if (target < 1) return false
  if (pageCount !== null && pageCount > 0 && target > pageCount) return false
  return true
}

/** "Page 4 of 12", or "Page 4" when the count was never recorded.
 *  Never "Page 4 of null", and never a guessed total. */
export function pageLabel(page: number, pageCount: number | null): string {
  return pageCount !== null && pageCount > 0 ? `Page ${page} of ${pageCount}` : `Page ${page}`
}

/** Which page to open on, given what a citation could say about itself.
 *
 * `null` from `/chunk-page` means the passage could not be placed — an unmarked cache, or a chunk
 * with no recorded span. The viewer then opens at page 1 **and the caller shows no page claim**,
 * which is why this returns the flag rather than just the number: "page 1" and "page 1, and we
 * know that" look identical otherwise. */
export function openingPage(
  citedPage: number | null,
  pageCount: number | null,
): { page: number; located: boolean } {
  if (citedPage === null) return { page: 1, located: false }
  return { page: clampPage(citedPage, pageCount), located: true }
}

// ---------------------------------------------------------------------------------------------
// Zoom (ROADMAP 18, second pass)
// ---------------------------------------------------------------------------------------------
//
// Zoom is expressed as a multiple of the pane's own width: 1 means the page exactly fills the
// pane, 2 means twice that. That choice is what keeps the arithmetic honest — the alternative,
// a percentage of "actual size", has no meaning for a page that was never on paper here, and it
// would change under the reader whenever the pane is resized.
//
// The second half is resolution. A page fetched at one dpi and then magnified is just blur, so
// the viewer re-asks the server for a sharper render as it zooms. That request has to be
// *quantised*, or every drag of the zoom control would fetch a new image.

/** The dpi the pane asks for at rest, matching the server's own default. */
export const BASE_DPI = 110

/** Resolutions the viewer will ask for. Snapping to a ladder is what makes zoom cheap: without
 *  it, a continuous zoom issues a fresh render per frame. The top rung is the server's ceiling. */
export const DPI_LADDER = [110, 150, 200, 260, 330, 400] as const

export const MIN_ZOOM = 0.25
export const MAX_ZOOM = 6

/** How far one press of +/- moves. A geometric step keeps the *proportional* change constant,
 *  so zooming feels the same at 0.5x as at 4x — a fixed +0.25 does not. */
const ZOOM_FACTOR = 1.25

export function clampZoom(zoom: number): number {
  if (!Number.isFinite(zoom)) return 1
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom))
}

/** One step in or out, snapped to 2 decimals so the label does not show 1.5625000000000002. */
export function zoomStep(zoom: number, direction: 1 | -1): number {
  const next = direction === 1 ? zoom * ZOOM_FACTOR : zoom / ZOOM_FACTOR
  return clampZoom(Math.round(next * 100) / 100)
}

/** The zoom at which the whole page is visible in a box, given the page's aspect (height/width).
 *
 * Never magnifies: a page shorter than its box fits at 1 rather than being blown up to touch the
 * top and bottom, which is what "fit page" means to a reader and not what the arithmetic alone
 * would give. Returns 1 for a box or aspect that is not yet known, so the first paint is sane. */
export function fitPageZoom(boxWidth: number, boxHeight: number, aspect: number): number {
  if (!(boxWidth > 0) || !(boxHeight > 0) || !(aspect > 0)) return 1
  return clampZoom(Math.min(1, boxHeight / (boxWidth * aspect)))
}

/** The page's width in PostScript points, recovered from any render whose dpi we know.
 *
 * The viewer never learns a page's true size from the API — it infers it from the picture it
 * asked for. That is exact (dpi is points-to-pixels by definition) and it costs no extra field
 * on the wire. */
export function pageWidthPt(naturalWidth: number, dpi: number): number {
  if (!(naturalWidth > 0) || !(dpi > 0)) return 0
  return (naturalWidth * 72) / dpi
}

/** The dpi to request so a page drawn `cssWidth` CSS-pixels wide is sharp on this display.
 *
 * `devicePixelRatio` belongs in the sum, but it does **not** mean the default is soft at rest —
 * a first draft of this comment claimed that and a test disproved it. At the measured pane width
 * (433 CSS px for a 612pt page) the render needs 51 dpi at 1x and 102 at 2x, both under the 110
 * served. It starts to bite once zoomed (153 dpi at 1.5x on a 2x display), on a pane dragged
 * wide (212 dpi at 900 px on 2x), or at 3x. Snaps UP the ladder — rounding down would ask for an
 * image blurrier than the one it replaces. */
export function renderDpi(cssWidth: number, widthPt: number, devicePixelRatio = 1): number {
  if (!(cssWidth > 0) || !(widthPt > 0)) return BASE_DPI
  const needed = (cssWidth * Math.max(1, devicePixelRatio) * 72) / widthPt
  return DPI_LADDER.find((rung) => rung >= needed) ?? DPI_LADDER[DPI_LADDER.length - 1]
}

/** "140%" — what the control shows. Rounded, because a reader does not want 139.7%. */
export function zoomLabel(zoom: number): string {
  return `${Math.round(zoom * 100)}%`
}

/** Where a drag puts the split between the document and the source pane, as the pane's share.
 *
 * Clamped so neither side can be dragged away entirely: a pane at 0 is indistinguishable from a
 * closed one but leaves the close button lying, and a document column below ~25% cannot hold a
 * line of text. */
export const MIN_SPLIT = 0.25
export const MAX_SPLIT = 0.75

export function clampSplit(fraction: number): number {
  if (!Number.isFinite(fraction)) return 0.46
  return Math.min(MAX_SPLIT, Math.max(MIN_SPLIT, fraction))
}
