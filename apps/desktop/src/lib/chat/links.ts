// Which links in an answer may stay clickable (A2, docs/security.md S5). Pure and dependency-free
// so `node:test` can run it; the DOM walk that uses it stays in Markdown.svelte.
//
// A small model can be told by a document to write a link ("visit … to continue"), and a
// markdown link renders as clickable. So an answer's web link stays a link only when its address
// is in one of the passages the answer was given: the backend sends that list as
// `TurnResult.source_links` (`source_checks.sourced_links`). Every other web link is shown as
// plain text, with its address visible. Kept identical to the backend's punctuation trim.

const TRAILING = /[.,;:!?*_]+$/

/** The address as both sides compare it: trailing sentence punctuation removed. */
export function normaliseLink(href: string): string {
  return href.replace(TRAILING, '')
}

/** Is this a web link (http/https), the only kind the check governs? */
export function isWebLink(href: string): boolean {
  return /^https?:\/\//i.test(href)
}

/**
 * May this link stay clickable? `allowed === null` means the text is not a model's answer (a
 * pre-rendered block the app wrote), so every link stays. `[]` means no link is sourced: while an
 * answer streams, and for a conversation reopened from history, which keeps no passages.
 */
export function linkAllowed(href: string, allowed: readonly string[] | null): boolean {
  if (allowed === null || !isWebLink(href)) return true
  const target = normaliseLink(href)
  return allowed.some((a) => normaliseLink(a) === target)
}
