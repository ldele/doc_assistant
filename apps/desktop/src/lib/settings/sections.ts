// Settings categories — the drawer's navigation model (2026-08-30).
//
// The panel used to be one flat scroll of ten `<section>`s. That was legible at five and stopped
// being legible before it stopped growing: everything a user might want was equidistant from
// everything else, and the only way to find a control was to read all of them. Categories fix the
// reading order, and they are the seam that lets the panel take new settings without the same
// decay — a new control joins a category, and the category tells you where it belongs.
//
// Pure and data-only on purpose: it lives beside `Settings.svelte` rather than inside it so it is
// runnable under `node:test` (apps/desktop/CLAUDE.md — `.ts` is tested, `.svelte.ts` is not). The
// component owns which sections render; this owns what the categories *are*.

export type SettingsSectionId = 'setup' | 'documents' | 'models' | 'answers' | 'general'

export interface SettingsSection {
  readonly id: SettingsSectionId
  /** Rail label. Short — the rail is a fixed column, not a paragraph. */
  readonly label: string
  /** Icon name from `shell/Icon.svelte`. Literal-typed via `as const` below, so a name that
   *  icon set does not have fails `svelte-check` at the `<Icon>` call rather than at runtime. */
  readonly icon: string
  /** One line under the category heading: what this category is *for*, so a user who guessed
   *  wrong finds out here instead of by reading every control in it. */
  readonly blurb: string
}

export const SETTINGS_SECTIONS = [
  {
    id: 'setup',
    label: 'Getting started',
    icon: 'square-check-big',
    blurb: 'The two things Provenote needs before it can answer: somewhere to get answers from, and documents to ground them in.',
  },
  {
    id: 'documents',
    label: 'Documents',
    icon: 'library',
    blurb: 'Where your library lives, what is in it, and which files are indexed.',
  },
  {
    id: 'models',
    label: 'Provider & model',
    icon: 'message-square',
    blurb: 'Which model answers your questions, and what it is allowed to say about your sources.',
  },
  {
    id: 'answers',
    label: 'Retrieval',
    icon: 'search',
    blurb: 'How an answer is assembled from your library. Session-only experiments, plus the locked defaults they start from.',
  },
  {
    id: 'general',
    label: 'General',
    icon: 'settings',
    blurb: 'Appearance, your chat history, and update checks.',
  },
] as const satisfies readonly SettingsSection[]

/** Which category the drawer opens on.
 *
 * Setup wins whenever it has something outstanding: on a fresh install it is the only category
 * that matters (ADR-034), and landing anywhere else would bury the one thing standing between the
 * user and a working app. It is decided from state the shell has *already* loaded, so the panel
 * opens on the right category rather than opening on one and jumping to another. */
export function initialSection(outstandingSetupSteps: number): SettingsSectionId {
  return outstandingSetupSteps > 0 ? 'setup' : 'documents'
}

/** The rail badge for a category, or `null` for no badge.
 *
 * Only setup carries one, and only while it is incomplete — a badge that is always there is
 * decoration, and a user learns to stop seeing it. */
export function sectionBadge(
  id: SettingsSectionId,
  outstandingSetupSteps: number,
): string | null {
  if (id !== 'setup' || outstandingSetupSteps <= 0) return null
  return String(outstandingSetupSteps)
}
