import { strict as assert } from 'node:assert'
import { test } from 'node:test'

import {
  CRITICAL_READING,
  GONE_FOLDER_GROUP,
  RESEARCH_ASSISTANT,
  SAMPLE_QUESTIONS,
  WHOLE_LIBRARY_GROUP,
  carriedText,
  chatLineText,
  continuePrefill,
  citedOptions,
  continuedFromText,
  countLine,
  emptyFolders,
  groupChats,
  labelLine,
  pruneChoice,
  sampleQuestionsFor,
  searchInLabel,
  startFromDefaults,
  tickFolder,
} from './setup.ts'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const folders = [
  { id: 't', name: 'Thesis ch.2', doc_count: 14 },
  { id: 'm', name: 'Methods', doc_count: 9 },
  { id: 'r', name: 'Reading group', doc_count: 22 },
]

test('ticking All unticks every folder, and unticking the last folder ticks All again', () => {
  let choice = tickFolder(null, 't', true)
  assert.deepEqual(choice, ['t'])
  choice = tickFolder(choice, 'm', true)
  assert.deepEqual(choice, ['t', 'm'])
  assert.equal(tickFolder(choice, null, true), null)
  choice = tickFolder(choice, 't', false)
  assert.deepEqual(choice, ['m'])
  assert.equal(tickFolder(choice, 'm', false), null)
  // Unticking All on its own changes nothing: a folder choice is made by ticking a folder.
  assert.equal(tickFolder(null, null, false), null)
  assert.deepEqual(tickFolder(['t'], 't', true), ['t'])
})

test('each built-in mode has three sample questions; a person\'s own mode shows Research assistant\'s', () => {
  assert.equal(SAMPLE_QUESTIONS[RESEARCH_ASSISTANT].length, 3)
  assert.equal(SAMPLE_QUESTIONS[CRITICAL_READING].length, 3)
  assert.deepEqual(sampleQuestionsFor('my-own-mode'), SAMPLE_QUESTIONS[RESEARCH_ASSISTANT])
  assert.deepEqual(sampleQuestionsFor(CRITICAL_READING), [
    'Where do my sources agree and disagree?',
    'What are the limits of the evidence?',
    'Which claims rest on a single study?',
  ])
})

test('Search in names the choice and the union count, never a sum of the folders', () => {
  assert.equal(searchInLabel(null, folders, 85, null), 'All documents · 85')
  assert.equal(searchInLabel(['t', 'm'], folders, 85, 21), 'Thesis ch.2 + Methods · 21')
  assert.equal(searchInLabel(['t', 'm', 'r'], folders, 85, 40), '3 folders · 40')
  assert.equal(searchInLabel(['t'], folders, 85, null), 'Thesis ch.2')
  assert.equal(searchInLabel(['gone'], folders, 85, 0), 'a folder that no longer exists · 0')
})

test('the count line says a shared document is searched once', () => {
  assert.equal(countLine(null, folders, 85, null), 'The whole library · 85 documents')
  assert.equal(
    countLine(['t', 'm'], folders, 85, 21),
    '2 folders · 21 documents: 2 are in more than one, searched once',
  )
  assert.equal(countLine(['m'], folders, 85, 9), '1 folder · 9 documents')
  assert.equal(countLine(['gone'], folders, 85, 0), '1 folder · 0 documents: nothing is searched')
  assert.equal(countLine(null, [], 0, null), 'The whole library · 0 documents')
})

const scope = (names: (string | null)[], counts: number[], union: number) => ({
  folders: names.map((n, i) => ({ folder_id: `f${i}`, folder_name: n, doc_count: counts[i] })),
  doc_count: union,
})

test('the chat line names the mode and the folders with the count', () => {
  assert.equal(chatLineText('Critical reading', null), 'Critical reading · whole library')
  assert.equal(
    chatLineText('Research assistant', scope(['Methods'], [9], 9)),
    'Research assistant · Methods (9 documents)',
  )
  assert.equal(
    chatLineText('Research assistant', scope(['A', null], [1, 0], 1)),
    'Research assistant · A, a folder that no longer exists (1 document)',
  )
  assert.deepEqual(emptyFolders(scope(['A', 'B', null], [3, 0, 0], 3)), [
    'B',
    'a folder that no longer exists',
  ])
  assert.deepEqual(emptyFolders(null), [])
})

test('continued-from and carried lines', () => {
  assert.equal(
    continuedFromText('Dopamine and ageing', 'Critical reading', null),
    'Continued from “Dopamine and ageing” (Critical reading · whole library)',
  )
  assert.equal(
    continuedFromText('Methods review', 'Research assistant', scope(['Methods', 'Thesis'], [9, 14], 21)),
    'Continued from “Methods review” (Research assistant · Methods, Thesis)',
  )
  assert.equal(carriedText(4), '4 earlier exchanges, carried as context')
  assert.equal(carriedText(1), '1 earlier exchange, carried as context')
})

test('a row\'s labels: two names, then +N', () => {
  assert.equal(labelLine({ documents: [], folders: [] }), '')
  assert.equal(labelLine({ documents: [{ title: 'Smith 2021' }], folders: [{ name: 'Thesis' }] }), 'Thesis, Smith 2021')
  assert.equal(
    labelLine({
      documents: [{ title: 'Smith 2021' }, { title: 'Lee 2019' }, { title: 'Ortiz 2020' }],
      folders: [{ name: 'Thesis' }],
    }),
    'Thesis, Smith 2021 +2',
  )
})

const chat = (id: string, mode: string, fs: [string, string | null][]) => ({
  session_id: id,
  mode_name: mode,
  folders: fs.map(([folder_id, name]) => ({ folder_id, name })),
})

test('grouping by mode keeps the order; by folder a chat sits under each of its folders', () => {
  const chats = [
    chat('1', 'Critical reading', [['t', 'Thesis']]),
    chat('2', 'Research assistant', []),
    chat('3', 'Critical reading', [['t', 'Thesis'], ['m', 'Methods']]),
    chat('4', 'Research assistant', [['x', null]]),
  ]
  const byMode = groupChats(chats, 'mode')
  assert.deepEqual(
    byMode.map((g) => [g.label, g.chats.map((c) => c.session_id)]),
    [
      ['Critical reading', ['1', '3']],
      ['Research assistant', ['2', '4']],
    ],
  )
  const byFolder = groupChats(chats, 'folder')
  assert.deepEqual(
    byFolder.map((g) => [g.label, g.chats.map((c) => c.session_id)]),
    [
      ['Thesis', ['1', '3']],
      [WHOLE_LIBRARY_GROUP, ['2']],
      ['Methods', ['3']],
      [GONE_FOLDER_GROUP, ['4']],
    ],
  )
  assert.deepEqual(groupChats(chats, 'none').map((g) => g.chats.length), [4])
})

test('Only chats that cited offers folders first, then documents', () => {
  const opts = citedOptions(
    'me',
    [{ id: 'm', name: 'Methods' }],
    [{ id: 'd1', title: 'Measurement theory' }, { id: 'd2', title: 'Other' }],
  )
  assert.deepEqual(
    opts.map((o) => [o.kind, o.id]),
    [
      ['folder', 'm'],
      ['document', 'd1'],
    ],
  )
  assert.deepEqual(citedOptions('  ', [{ id: 'm', name: 'Methods' }], []), [])
})

test('the start screen opens preselected from the defaults', () => {
  assert.deepEqual(startFromDefaults(null), { modeId: RESEARCH_ASSISTANT, folders: null })
  assert.deepEqual(
    startFromDefaults({ mode_id: CRITICAL_READING, folder_ids: ['t', 'm'] }),
    { modeId: CRITICAL_READING, folders: ['t', 'm'] },
  )
  assert.deepEqual(startFromDefaults({ mode_id: RESEARCH_ASSISTANT, folder_ids: null }).folders, null)
})

test('a tick after the default folders were deleted starts from the folders that exist', () => {
  assert.deepEqual(pruneChoice(['gone', 't'], ['t', 'm']), ['t'])
  assert.deepEqual(tickFolder(pruneChoice(['gone'], ['t']), 't', true), ['t'])
  assert.equal(pruneChoice(null, ['t']), null)
})

test('Continue in a new chat is prefilled with the setup of the chat, or the mode /mode named', () => {
  const from = { mode: { id: CRITICAL_READING }, folder_ids: ['m'] }
  assert.deepEqual(continuePrefill(from, null), { modeId: CRITICAL_READING, folders: ['m'], carry: true })
  assert.deepEqual(continuePrefill(from, RESEARCH_ASSISTANT).modeId, RESEARCH_ASSISTANT)
  assert.deepEqual(continuePrefill(null, null), { modeId: RESEARCH_ASSISTANT, folders: null, carry: true })
})

test('the composer holds the text box and Send only: no mode or folder picker (W1)', () => {
  const here = dirname(fileURLToPath(import.meta.url))
  const chat = readFileSync(join(here, '../../screens/chat/Chat.svelte'), 'utf-8')
  const footer = chat.slice(chat.indexOf('<footer>'), chat.indexOf('</footer>'))
  assert.ok(footer.includes('<textarea'), 'the composer has its text box')
  assert.ok(footer.includes('Send'), 'the composer has Send')
  for (const picker of ['<select', 'FolderTicks', 'ModeChoice', 'StartScreen', 'type="checkbox"', 'type="radio"']) {
    assert.ok(!footer.includes(picker), `the composer has no picker (${picker})`)
  }
})
