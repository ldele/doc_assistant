// Multi-select over a list, the same in every list that has a select mode (the chat history, the
// Library grid and list). Pure, so `node --test` reaches it; the components only pass the click.
//
// The semantics are the standard ones (ROADMAP 78, 80): a click or a Ctrl/Cmd-click toggles one
// row, and a Shift-click applies the clicked row's new state to every row between it and the last
// row clicked. So a Shift-click on an unticked row ticks the range, and on a ticked row unticks
// it: the same gesture undoes itself, and nothing outside the range moves.

/** What a click leaves: the new selection and the row the next Shift-click ranges from. */
export interface Picked {
  selected: string[]
  anchor: string
}

/**
 * The selection after a click on `target`, in a list shown in `order`.
 *
 * `anchor` is the row the last click landed on. A Shift-click without one, or with one that is no
 * longer shown (a filter changed, a chat was deleted), toggles the clicked row alone: a range to
 * a row the user cannot see would select things they never looked at.
 *
 * The result keeps the rows already selected in their order and appends the new ones in list
 * order, so a caller that acts on the selection in sequence acts in the order the user sees.
 */
export function pick(
  order: readonly string[],
  selected: readonly string[],
  anchor: string | null,
  target: string,
  shift: boolean,
): Picked {
  const on = !selected.includes(target)
  const from = shift && anchor !== null ? order.indexOf(anchor) : -1
  const to = order.indexOf(target)
  const range =
    from >= 0 && to >= 0 ? order.slice(Math.min(from, to), Math.max(from, to) + 1) : [target]
  if (on) {
    const have = new Set(selected)
    return { selected: [...selected, ...range.filter((id) => !have.has(id))], anchor: target }
  }
  const drop = new Set(range)
  return { selected: selected.filter((id) => !drop.has(id)), anchor: target }
}

/** True when a click asks for selection rather than for opening the row: Shift, Ctrl or Cmd. */
export function isSelectClick(e: Pick<MouseEvent, 'shiftKey' | 'ctrlKey' | 'metaKey'>): boolean {
  return e.shiftKey || e.ctrlKey || e.metaKey
}

/**
 * The one select-bar helper both lists show: *Select none* while anything is selected, else
 * *Select all*. It replaces a bare "Clear", which read as a possible delete (the user,
 * 2026-11-05): the label says what the click does to the selection, and nothing else.
 */
export function selectAllLabel(selected: readonly string[]): 'Select all' | 'Select none' {
  return selected.length > 0 ? 'Select none' : 'Select all'
}

/** What that helper does: empty a non-empty selection, else select every shown row (never one a
 *  filter hides, so it cannot act on what the user cannot see). */
export function selectAllOrNone(order: readonly string[], selected: readonly string[]): string[] {
  return selected.length > 0 ? [] : [...order]
}
