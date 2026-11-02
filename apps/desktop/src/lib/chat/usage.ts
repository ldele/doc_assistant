// The one-line turn cost under an answer. Pure, so the honesty rule below can be pinned by tests.

/** The fields of `TurnUsage` this label reads. Kept structural so the test needs no wire type. */
export interface UsageLike {
  turn_input: number
  turn_output: number
  is_local: boolean
  cost_usd: number | null
  billing?: string | null
}

/**
 * "1,204 tokens · $0.0031", "1,204 tokens · local", or — the case this exists for —
 * "local · tokens not reported".
 *
 * **A local run reports no token counts, and zero is not the same as unknown.** Ollama returns no
 * usage, so the counters stay at their initial 0 and the line read `0 tokens · local`: a
 * measurement of nothing, where the truth is that nothing was measured. It is the same distinction
 * ADR-044 draws for update checks — a failed check is `unknown`, never "up to date".
 *
 * The zero is what is checked, not `is_local` alone: a local provider that *does* report counts
 * (some Ollama builds return `eval_count`) should have them shown like any other.
 */
export function usageLabel(usage: UsageLike | null): string {
  if (!usage) return ''
  const total = usage.turn_input + usage.turn_output
  if (usage.is_local && total === 0) return 'local · tokens not reported'
  // ADR-060: a provider Provenote has no price for says who bills, never a guessed cost. Its
  // token counts are whatever the server reported, and none reads as not reported, not as zero.
  if (usage.billing) {
    const counted = total === 0 ? 'tokens not reported' : `${total.toLocaleString()} tokens`
    return `${counted} · ${usage.billing.replace(/\.$/, '')}`
  }
  const spend = usage.is_local ? 'local' : usage.cost_usd != null ? `$${usage.cost_usd.toFixed(4)}` : 'n/a'
  return `${total.toLocaleString()} tokens · ${spend}`
}
