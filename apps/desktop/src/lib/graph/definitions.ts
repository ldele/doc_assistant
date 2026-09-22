// Pure display logic for a concept's definition candidates (ADR-053). Tested under `npm test`;
// ConceptDefinition.svelte only renders what this returns.
import type { ConceptDefinitions, DefinitionCandidate, UsageLine, VocabularyMatch } from '../core/types'

export interface CandidateGroups {
  chosen: DefinitionCandidate | null
  suggested: DefinitionCandidate[]
  dismissed: DefinitionCandidate[]
}

/** Split a concept's candidates the way the panel shows them. The server already orders them
 *  (chosen, then by grade); this keeps that order inside each group. */
export function groupCandidates(view: ConceptDefinitions | null): CandidateGroups {
  const out: CandidateGroups = { chosen: null, suggested: [], dismissed: [] }
  for (const c of view?.candidates ?? []) {
    if (c.status === 'chosen') out.chosen = c
    else if (c.status === 'dismissed') out.dismissed.push(c)
    else out.suggested.push(c)
  }
  return out
}

/** The words shown for a grade — evidence, never a verdict. */
export function gradeLabel(grade: string): string {
  switch (grade) {
    case 'strong':
      return 'Strong evidence'
    case 'some':
      return 'Some evidence'
    case 'thin':
      return 'Thin evidence'
    case 'user':
      return 'Your words'
    default:
      return grade
  }
}

export function sourceLabel(c: DefinitionCandidate): string {
  if (c.source === 'user') return 'Written by you'
  if (c.source === 'model') return 'Suggested by a model'
  return c.document_title ? `From “${c.document_title}”` : 'From your library'
}

/** Where a usage example comes from, and how much that document uses the word. */
export function usageSource(u: UsageLine): string {
  const doc = u.document_title ? `“${u.document_title}”` : 'A document in your library'
  const times = u.doc_mentions === 1 ? 'once' : `${u.doc_mentions} times`
  return `${doc} · uses it ${times}`
}

/** Whitespace collapsed for reading. Display only: the stored passage stays verbatim (ADR-043). */
export function displayText(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

/** Search results the graph index does not already list — the ones only this search can reach. */
export function offGraphMatches(
  matches: VocabularyMatch[],
  graphIds: ReadonlySet<string>,
): VocabularyMatch[] {
  return matches.filter((m) => !graphIds.has(m.id))
}
