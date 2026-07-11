// Manual System/Light/Dark theme (U1, docs/specs/feature-phase8-ui-upgrade.md). Pure DOM +
// storage, no framework dependency — callable from main.ts before mount() so the theme
// attribute is set before first paint (no flash-of-wrong-theme). Client-only: never a
// backend setting (ADR-010's non-persistence wall is for the retrieval-quality-governed
// knobs; theme is a cosmetic rendering preference and was never part of that governance).

export type Theme = 'system' | 'light' | 'dark'

const STORAGE_KEY = 'theme'

export function getTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY)
  return stored === 'light' || stored === 'dark' ? stored : 'system'
}

export function setTheme(theme: Theme): void {
  localStorage.setItem(STORAGE_KEY, theme)
}

export function applyTheme(theme: Theme): void {
  if (theme === 'system') {
    delete document.documentElement.dataset.theme
  } else {
    document.documentElement.dataset.theme = theme
  }
}
