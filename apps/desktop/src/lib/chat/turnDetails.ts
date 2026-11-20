// The words of what a turn shows besides its answer (A8b, docs/specs/chat-interface-pass.md):
// *What the answer carried* inside *How this answer was made* (A5's counts, A8a's carried
// exchanges), *What it looked up* for a records answer (A6), and a reopened chat's recorded
// decisions and reviewer verdict. Pure, pinned by `turnDetails.test.ts`; read from what the turn
// recorded, never worked out again.

/** The fields of `TurnContext` this reads (kept structural so the test needs no wire type). */
export interface ContextLike {
  window: number
  window_source: string
  rewrite?: { history_used: number; history_dropped: number; answer_shortened: boolean; skipped: boolean } | null
  answer?: { passages_used: number; passages_dropped: number; top_passage_clipped: boolean } | null
  carried?: { exchanges: number; kept: number; dropped: number } | null
}

export interface Row {
  key: string
  value: string
}

const WINDOW_FROM: Record<string, string> = {
  set: 'set by the app for the local model',
  server: "reported by the model's server",
  documented: 'documented by the provider',
  floor: "the server does not state it, so the app's floor was used",
}

function count(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`
}

/** *What the answer carried*: one row per thing a turn recorded, in the drawing's order. */
export function carriedRows(
  ctx: ContextLike | null,
  modeName: string | null,
  promptVersion: string | null,
): Row[] {
  const rows: Row[] = []
  if (ctx) {
    const rw = ctx.rewrite
    if (rw) {
      const total = rw.history_used + rw.history_dropped
      let v = rw.skipped
        ? 'none: the conversation did not fit, so the question was searched as asked'
        : `${rw.history_used} of ${total}`
      if (!rw.skipped && rw.history_dropped) v += ` · the ${count(rw.history_dropped, 'oldest', 'oldest')} left out`
      if (rw.answer_shortened) v += ' · the previous answer shortened'
      rows.push({ key: 'Earlier messages', value: v })
    } else {
      rows.push({ key: 'Earlier messages', value: 'none' })
    }
    const c = ctx.carried
    if (c) {
      let v = `${c.kept} of ${count(c.exchanges, 'carried exchange', 'carried exchanges')}`
      if (c.dropped) v += ` · the ${c.dropped} oldest left out`
      rows.push({ key: 'Carried', value: v })
    }
    const an = ctx.answer
    if (an) {
      const total = an.passages_used + an.passages_dropped
      let v = `${an.passages_used} of ${total}`
      if (an.passages_dropped) v += ` · ${an.passages_dropped} left out`
      if (an.top_passage_clipped) v += ' · the best one shortened'
      rows.push({ key: 'Passages', value: v })
    }
    rows.push({
      key: 'Window',
      value: `${ctx.window.toLocaleString('en-US')} tokens · ${WINDOW_FROM[ctx.window_source] ?? ctx.window_source}`,
    })
  }
  if (modeName) {
    rows.push({
      key: 'Mode',
      value: promptVersion ? `${modeName} · prompt version ${promptVersion.slice(0, 8)}` : modeName,
    })
  }
  return rows
}

/** Whether anything gave way to fit the window (the card then leads with a warning). */
export function anythingLeftOut(ctx: ContextLike | null): boolean {
  if (!ctx) return false
  return Boolean(
    ctx.rewrite?.history_dropped ||
      ctx.rewrite?.answer_shortened ||
      ctx.rewrite?.skipped ||
      ctx.answer?.passages_dropped ||
      ctx.answer?.top_passage_clipped ||
      ctx.carried?.dropped ||
      ctx.window_source === 'floor',
  )
}

/** The fields of `RecordsView` this reads. */
export interface RecordsLike {
  lookup: string
  args: Record<string, unknown>
  rows: number
}

/** *What it looked up*'s call line: "list_documents · folder Thesis ch.2 · 14 rows". */
export function recordsLine(r: RecordsLike): string {
  const args = Object.entries(r.args)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${k.replace(/_/g, ' ')} ${Array.isArray(v) ? v.join(', ') : String(v)}`)
  return [r.lookup, ...args, count(r.rows, 'row', 'rows')].join(' · ')
}

export const RECORDS_TAG = "Answered from the library's records · no model, no passages"

/** The fields of `ReviewView` this reads. */
export interface ReviewLike {
  faithfulness: number | null
  citation_density: number | null
  hedging_adequacy: number | null
  unsupported_claims: number | null
  error: string | null
}

/** The reviewer's verdict as recorded, in one line; its words, not a check the app ran. */
export function reviewLine(r: ReviewLike): string {
  if (r.error) return `Reviewer, as recorded: no verdict (${r.error})`
  const parts: string[] = []
  if (r.faithfulness !== null) parts.push(`faithfulness ${r.faithfulness}/5`)
  if (r.citation_density !== null) parts.push(`citation ${r.citation_density}/5`)
  if (r.hedging_adequacy !== null) parts.push(`hedging ${r.hedging_adequacy}/5`)
  if (r.unsupported_claims !== null)
    parts.push(count(r.unsupported_claims, 'claim it found unsupported', 'claims it found unsupported'))
  return `Reviewer, as recorded: ${parts.join(' · ') || 'no scores'}`
}

/** The fields of a claim's recorded decision. */
export interface DecisionLike {
  n: number
  decision: string
  edited_text: string | null
}

/** The decisions a person recorded on a turn's claims, one line each; pending ones left out. */
export function decisionLines(claims: DecisionLike[]): string[] {
  return claims
    .filter((c) => c.decision !== 'pending')
    .map((c) =>
      c.decision === 'edited' && c.edited_text
        ? `#${c.n} edited: “${c.edited_text}”`
        : `#${c.n} ${c.decision}`,
    )
}
