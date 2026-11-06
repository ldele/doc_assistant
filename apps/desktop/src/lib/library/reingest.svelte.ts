// *Re-run ingestion* dialog state (ADR-048, ROADMAP 20/21; ADR-064 choice 5): which documents it
// re-runs, the label it names them by, and the parts ticked when it opens.
//
// A `.svelte.ts` rune module beside the pure, tested `reingest.ts` (the label, the cost statement,
// the report). One dialog, four doors, each calling `openRerun` directly: *Re-run…* in the
// document page's index, *Re-run text now…* in its letter-spacing block, *Re-run…* on the
// Library's select bar and *Re-run text now…* in Settings › Documents. App mounts the dialog
// while `rerun.targets` is set, because it mounts every overlay.
//
// What stays in App.svelte: what runs after a run finishes. It refetches the document list and
// re-reads the engine's health for the status bar, two domains, so it is glue.

export const rerun = $state({
  /** The documents to re-run; `null` while the dialog is closed. */
  targets: null as string[] | null,
  /** What is being re-run, in the user's terms: a title, or "12 documents". */
  label: '',
  /** Parts ticked on open, when the door knows which one applies (the letter-spacing doors take
   *  effect through Text & chunks). Still the user's to start. */
  parts: [] as string[],
})

export function openRerun(ids: string[], label: string, parts: string[] = []): void {
  if (ids.length === 0) return
  rerun.targets = ids
  rerun.label = label
  rerun.parts = parts
}

export function closeRerun(): void {
  rerun.targets = null
}
