// Library view preferences — grid/list, sort order, and how the source pane fits a page.
//
// Client-only, localStorage-backed, in the same class as the theme toggle and the sidebar width:
// never a backend setting. Every read/write is wrapped because localStorage throws in private
// modes and on a blocked origin — a preference failing to persist must never break the Library.

import type { LibrarySort } from './library'
import { clampSplit } from './sourceviewer'

export const LIB_SORTS: { key: LibrarySort; label: string }[] = [
  { key: 'title-az', label: 'Title (A–Z)' },
  { key: 'author-az', label: 'Author (A–Z)' },
  { key: 'pub-desc', label: 'Publication date (newest)' },
  { key: 'added-desc', label: 'Added date (newest)' },
]

function loadView(): 'grid' | 'list' {
  try {
    const v = localStorage.getItem('libraryView')
    if (v === 'grid' || v === 'list') return v
  } catch {
    /* ignore — fall back to default */
  }
  return 'grid'
}

/** How the source pane sizes a page image (ROADMAP 18).
 *
 * `page` fits the whole page in the pane; `width` fills the pane's width and scrolls. */
export type SourceFit = 'page' | 'width'

function loadSourceFit(): SourceFit {
  try {
    const v = localStorage.getItem('sourceFit')
    if (v === 'page' || v === 'width') return v
  } catch {
    /* ignore — fall back to default */
  }
  // `page` by default, because the pane's job is *where did this come from*, not reading — the
  // extracted text is the reading surface (ADR-050 D1). Measured 2026-09-01: at fit-width no page
  // in the corpus fits, not even US Letter (94% visible); the tallest shows 67%. Fitting costs
  // little width in exchange — 405px against 433px on a Letter page in a 1280x720 window.
  return 'page'
}

function loadSourceSplit(): number {
  try {
    const v = Number(localStorage.getItem('sourceSplit'))
    if (Number.isFinite(v) && v > 0) return clampSplit(v)
  } catch {
    /* ignore — fall back to default */
  }
  return 0.46
}

function loadSort(): LibrarySort {
  try {
    const v = localStorage.getItem('librarySort')
    if (LIB_SORTS.some((s) => s.key === v)) return v as LibrarySort
  } catch {
    /* ignore — fall back to default */
  }
  return 'title-az'
}

export const libPrefs = $state({
  view: loadView(),
  sort: loadSort(),
  sourceFit: loadSourceFit(),
  /** The source pane's share of the split, as a fraction (ROADMAP 18). */
  sourceSplit: loadSourceSplit(),
  /** The sort dropdown's open flag — transient UI, not persisted. */
  sortOpen: false,
})

export function setLibraryView(v: 'grid' | 'list'): void {
  libPrefs.view = v
  try {
    localStorage.setItem('libraryView', v)
  } catch {
    /* ignore — view just won't persist */
  }
}

export function setLibrarySort(v: LibrarySort): void {
  libPrefs.sort = v
  libPrefs.sortOpen = false
  try {
    localStorage.setItem('librarySort', v)
  } catch {
    /* ignore — just won't persist */
  }
}

export function setSourceFit(v: SourceFit): void {
  libPrefs.sourceFit = v
  try {
    localStorage.setItem('sourceFit', v)
  } catch {
    /* ignore — just won't persist */
  }
}

export function setSourceSplit(v: number): void {
  libPrefs.sourceSplit = clampSplit(v)
  try {
    localStorage.setItem('sourceSplit', String(libPrefs.sourceSplit))
  } catch {
    /* ignore — just won't persist */
  }
}
