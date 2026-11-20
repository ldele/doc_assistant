// A chat's setup on the screen (A8a, docs/specs/chat-interface-pass.md): the folder tickboxes'
// rules, the sample questions per mode, and the words of *Search in*, the chat line, the rail's
// labels and its groups. Pure, so every rule is pinned by `setup.test.ts`; the state that uses it
// is `setup.svelte.ts`.

/** A folder choice: `null` is *All documents* (the whole library), else the ticked folder ids. */
export type FolderChoice = string[] | null

/** Tick or untick one box. Ticking *All documents* unticks every folder; ticking a folder
 *  unticks *All*; unticking the last folder ticks *All* again (W1). `id === null` is *All*. */
export function tickFolder(choice: FolderChoice, id: string | null, checked: boolean): FolderChoice {
  if (id === null) return checked ? null : choice
  const current = choice ?? []
  if (checked) return current.includes(id) ? current : [...current, id]
  const left = current.filter((f) => f !== id)
  return left.length === 0 ? null : left
}

/** `choice` without the ids no folder in `known` has (gone since it was chosen): what a tick
 *  starts from, so ticking a folder after the defaults' folders were deleted replaces them. */
export function pruneChoice(choice: FolderChoice, known: string[]): FolderChoice {
  if (choice === null) return null
  return choice.filter((id) => known.includes(id))
}

/** The built-in modes' ids, as the backend names them (`chat_modes.BUILT_IN`). */
export const RESEARCH_ASSISTANT = 'research-assistant'
export const CRITICAL_READING = 'critical-reading'

/** Three sample questions per built-in mode (the spec's open question 2). Corpus-agnostic: they
 *  run on any library. A person's own mode shows Research assistant's. */
export const SAMPLE_QUESTIONS: Record<string, string[]> = {
  [RESEARCH_ASSISTANT]: [
    'What are the main themes across my documents?',
    'What are the key findings, with citations?',
    'Which documents describe their methods in detail?',
  ],
  [CRITICAL_READING]: [
    'Where do my sources agree and disagree?',
    'What are the limits of the evidence?',
    'Which claims rest on a single study?',
  ],
}

export function sampleQuestionsFor(modeId: string): string[] {
  return SAMPLE_QUESTIONS[modeId] ?? SAMPLE_QUESTIONS[RESEARCH_ASSISTANT]
}

/** The fields of a folder this reads. */
export interface FolderLike {
  id: string
  name: string
  doc_count: number
}

function documents(n: number): string {
  return `${n} document${n === 1 ? '' : 's'}`
}

/** *Search in*'s button: "All documents · 85", "Thesis ch.2 + Methods · 21", "3 folders · 30".
 *  `union` is the scope preview's count (each document once); while it is not known yet, the
 *  number is left out rather than guessed from the folders' own counts. */
export function searchInLabel(
  choice: FolderChoice,
  folders: FolderLike[],
  libraryCount: number,
  union: number | null,
): string {
  if (choice === null) return `All documents · ${libraryCount}`
  const names = choice.map((id) => folders.find((f) => f.id === id)?.name ?? 'a folder that no longer exists')
  const head = names.length <= 2 ? names.join(' + ') : `${names.length} folders`
  return union === null ? head : `${head} · ${union}`
}

/** The count line under the folder list: "The whole library · 85 documents", "2 folders · 21
 *  documents: 2 are in both, searched once", "No folder holds a document: nothing is searched". */
export function countLine(
  choice: FolderChoice,
  folders: FolderLike[],
  libraryCount: number,
  union: number | null,
): string {
  if (choice === null) return `The whole library · ${documents(libraryCount)}`
  if (union === null) return `${choice.length} folder${choice.length === 1 ? '' : 's'}`
  const own = choice.reduce((n, id) => n + (folders.find((f) => f.id === id)?.doc_count ?? 0), 0)
  const head = `${choice.length} folder${choice.length === 1 ? '' : 's'} · ${documents(union)}`
  if (union === 0) return `${head}: nothing is searched`
  const shared = own - union
  if (shared <= 0) return head
  return `${head}: ${shared} ${shared === 1 ? 'is' : 'are'} in more than one, searched once`
}

/** The fields of a resolved scope this reads (`TurnScope`). */
export interface ScopeLike {
  folders: { folder_id: string; folder_name: string | null; doc_count: number }[]
  doc_count: number
}

/** The folders part of a chat line: "whole library", "Methods (9 documents)", "Thesis ch.2,
 *  Methods (21 documents)". A folder that no longer exists is said, never dropped silently. */
export function foldersText(scope: ScopeLike | null): string {
  if (scope === null) return 'whole library'
  const names = scope.folders.map((f) => f.folder_name ?? 'a folder that no longer exists')
  return `${names.length ? names.join(', ') : 'no folder'} (${documents(scope.doc_count)})`
}

/** The chat line: "Critical reading · whole library". */
export function chatLineText(modeName: string, scope: ScopeLike | null): string {
  return `${modeName} · ${foldersText(scope)}`
}

/** The folders that hold nothing, for the chat line's note (ADR-025 F2): each by name, or "a
 *  folder that no longer exists". Empty when every folder adds documents. */
export function emptyFolders(scope: ScopeLike | null): string[] {
  if (scope === null) return []
  return scope.folders
    .filter((f) => f.folder_name === null || f.doc_count === 0)
    .map((f) => f.folder_name ?? 'a folder that no longer exists')
}

/** "Continued from “Dopamine and ageing” (Critical reading · Thesis ch.2)": the spec's form,
 *  the folders by name only (the counts are the old chat's line's to give). */
export function continuedFromText(
  title: string | null,
  modeName: string,
  scope: ScopeLike | null,
): string {
  const where =
    scope === null
      ? 'whole library'
      : scope.folders.map((f) => f.folder_name ?? 'a folder that no longer exists').join(', ') ||
        'no folder'
  return `Continued from “${title ?? 'an earlier chat'}” (${modeName} · ${where})`
}

/** "4 earlier exchanges, carried as context". */
export function carriedText(n: number): string {
  return `${n} earlier exchange${n === 1 ? '' : 's'}, carried as context`
}

/** A rail row's labels after its date line: up to two names, then "+N". Folders first (fewer,
 *  wider), then documents. Empty when the chat cited nothing. */
export function labelLine(labels: {
  documents: { title: string }[]
  folders: { name: string }[]
}): string {
  const names = [...labels.folders.map((f) => f.name), ...labels.documents.map((d) => d.title)]
  if (names.length === 0) return ''
  const shown = names.slice(0, 2).join(', ')
  return names.length > 2 ? `${shown} +${names.length - 2}` : shown
}

export type GroupKey = 'none' | 'mode' | 'folder'
export const GROUP_KEYS: GroupKey[] = ['none', 'mode', 'folder']
export const GROUP_LABELS: Record<GroupKey, string> = {
  none: 'No groups',
  mode: 'By mode',
  folder: 'By folder',
}

/** The fields of a chat this groups by. */
export interface GroupableChat {
  session_id: string
  mode_name: string
  folders: { folder_id: string; name: string | null }[]
}

export interface ChatGroup<T> {
  key: string
  label: string
  chats: T[]
}

export const WHOLE_LIBRARY_GROUP = 'Whole library'
export const GONE_FOLDER_GROUP = 'A folder that no longer exists'

/** The chats in groups, keeping the order they came in within each group; groups in the order
 *  their first chat appears. *By folder*: a chat in several folders sits under each, and a chat
 *  on the whole library under "Whole library". `none` is one unlabelled group. */
export function groupChats<T extends GroupableChat>(chats: T[], by: GroupKey): ChatGroup<T>[] {
  if (by === 'none') return [{ key: '', label: '', chats }]
  const groups = new Map<string, ChatGroup<T>>()
  const add = (key: string, label: string, chat: T): void => {
    let g = groups.get(key)
    if (!g) {
      g = { key, label, chats: [] }
      groups.set(key, g)
    }
    if (!g.chats.includes(chat)) g.chats.push(chat)
  }
  for (const chat of chats) {
    if (by === 'mode') add(`mode:${chat.mode_name}`, chat.mode_name, chat)
    else if (chat.folders.length === 0) add('folder:', WHOLE_LIBRARY_GROUP, chat)
    else
      for (const f of chat.folders)
        add(
          f.name === null ? 'folder:gone' : `folder:${f.folder_id}`,
          f.name ?? GONE_FOLDER_GROUP,
          chat,
        )
  }
  return [...groups.values()]
}

/** What *Only chats that cited* can take: a folder or a document, by name. */
export interface CitedOption {
  kind: 'folder' | 'document'
  id: string
  name: string
}

/** The options matching `query` (case and spacing aside), folders first, at most `limit`. */
export function citedOptions(
  query: string,
  folders: { id: string; name: string }[],
  documents: { id: string; title: string }[],
  limit = 8,
): CitedOption[] {
  const q = query.trim().toLowerCase()
  if (!q) return []
  const hit = (s: string): boolean => s.toLowerCase().includes(q)
  return [
    ...folders.filter((f) => hit(f.name)).map((f) => ({ kind: 'folder' as const, id: f.id, name: f.name })),
    ...documents
      .filter((d) => hit(d.title))
      .map((d) => ({ kind: 'document' as const, id: d.id, name: d.title })),
  ].slice(0, limit)
}

/** The start screen's choice from the New chat defaults (W2): their mode and folders, else
 *  Research assistant on the whole library. Gone default folders stay (they search nothing,
 *  never the whole library, ADR-025 F2). */
export function startFromDefaults(
  defaults: { mode_id: string; folder_ids: string[] | null } | null,
): { modeId: string; folders: FolderChoice } {
  return {
    modeId: defaults?.mode_id ?? RESEARCH_ASSISTANT,
    folders: defaults?.folder_ids ? [...defaults.folder_ids] : null,
  }
}

/** *Continue in a new chat*'s prefill: the current chat's mode and folders, or the mode
 *  `/mode <name>` asked for (G2); the carry tickbox starts ticked. */
export function continuePrefill(
  from: { mode: { id: string }; folder_ids: string[] | null } | null,
  modeId: string | null,
): { modeId: string; folders: FolderChoice; carry: boolean } {
  return {
    modeId: modeId ?? from?.mode.id ?? RESEARCH_ASSISTANT,
    folders: from?.folder_ids ? [...from.folder_ids] : null,
    carry: true,
  }
}
