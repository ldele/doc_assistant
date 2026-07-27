// Library view preferences — grid/list and sort order.
//
// Client-only, localStorage-backed, in the same class as the theme toggle and the sidebar width:
// never a backend setting. Every read/write is wrapped because localStorage throws in private
// modes and on a blocked origin — a preference failing to persist must never break the Library.

import type { LibrarySort } from './library'

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
