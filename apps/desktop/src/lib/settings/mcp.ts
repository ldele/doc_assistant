// Settings > AI programs: the words for what an outside program did (ADR-066 choice 9, A4).
// Pure, so `node:test` runs it; the section that shows it is `screens/settings/AiPrograms.svelte`.

import type { McpCall } from '../core/types/mcp'

/** `search_library · 5 items · 120 ms · 14:02`: one call, never its arguments. */
export function callLine(call: McpCall, locale = 'en-GB'): string {
  const when = new Date(call.at)
  const time = Number.isNaN(when.getTime())
    ? ''
    : when.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' })
  const items = `${call.items} ${call.items === 1 ? 'item' : 'items'}`
  return [call.tool, items, `${call.milliseconds} ms`, time].filter(Boolean).join(' · ')
}

/** How many tools a client sees, said the way the switch below it reads. */
export function toolsLine(tools: readonly string[], conversations: boolean): string {
  const n = tools.length
  return conversations
    ? `${n} tools, past conversations included.`
    : `${n} tools. Past conversations are not offered.`
}
