<!-- status: active · updated: 2026-11-20 (A8b built: As built added; the spec's two sessions done); 2026-11-20 (A8a built: As built added; A8b to build); 2026-11-20 (accepted by the user as written, open question 1 as recommended; written from the approved wireframe and A8's four-question grill) · class: living -->

# Spec — The chat interface pass: how a chat is set up, and what a turn shows (ROADMAP 103, sessions A8a and A8b)

> The executor brief. Target executor: **Claude Code**. Sessions A8a and A8b of the plan
> (`docs/plans/PLAN_2026-10-31_next-two-releases.md`): two build sessions, one PR each. The
> approved drawing and its reading are `docs/wireframes/chat.excalidraw` and
> `docs/wireframes/chat.md`. Element ids below (`[s2-scope]`) are the reading's. This file is the
> code-level contract. Where it and the drawing differ, this file wins, and it says where.

## In one sentence

A chat's mode and folders are chosen on the start screen *New chat* opens, then fixed and shown at
the top of the chat. Changing them means continuing in a new chat that carries the old one's
exchanges. Each turn shows its cited claims, what it looked up, and what it carried.

## Today (read in the code, 2026-11-20)

- **Start screen.** `screens/chat/Chat.svelte:101` shows the empty state with three fixed sample
  questions. The composer (`Chat.svelte:132`) holds the text box, *Test override* (when an
  override is set), a native folder `<select>` and *Send*. The scope lives in memory only
  (`App.svelte` `chatScopeFolderId`, ADR-025 S9) and goes as `scope_folder_ids: [id] | null`
  (A7b).
- **Mode.** A conversation keeps `conversation_meta.mode_id` (A7). `/mode <name>` switches it from
  the next turn (`chat_modes.mode_command`). There is no picker.
- **Defaults.** None. `app_settings.SETTINGS_PATH` is `settings.json` in the library's own data
  home (`config.DATA_PATH`).
- **Continuing a past chat** switches to its `session_id` with an **empty** memory
  (`feature-conversation-resume.md`, decision 1): new questions are answered as if asked fresh.
- **Rail.** `ChatHistory.svelte` has *Filter chats…*, select, and a sort menu (*Newest first ·
  Oldest first*). Each row shows its title, date and turn count. `/api/conversations` already
  returns labels and filters by cited document or folder (A7b), and nothing draws them.
- **Claims.** Each answer sentence is stored in `answer_claims` (`synthesis.segment_claims`)
  with its citations and a retrieval marker: `ok` (cites a well-ranked passage), `weak` (cites a
  low-ranked one, or one with no score), `unsupported` (cites nothing). `Turn.svelte` shows the
  flagged ones in a list under the answer (`ClaimReview`) and colours nothing in the text.
- **Records answers** (A6) and the **fitting note** (A5) are text in the answer and the
  citation block. No card shows either.

## Decisions — the user, 2026-11-20

| # | Branch | Decision | Reopens if |
|---|---|---|---|
| W1 | Where a chat's options are chosen | On the start screen *New chat* opens: mode chips, *Search in* with the folder tickboxes (*All* set apart), and sample questions that follow the mode. They are then fixed and shown in a line at the top of the chat. The composer holds the text box and *Send* only (revisions 1–3 of the wireframe, approved) | — |
| W2 | Defaults | A *New chat defaults* window, opened from the ⚙ beside *New chat*, separate from Settings. *Make this my default* on the start screen writes the same place | — |
| W3 | The rail | Unchanged. Grouping (*No groups · By mode · By folder*) and *Only chats that cited* go in the existing sort menu. Each row's date line gains its labels | — |
| G1 | The build | Two sessions, one PR each: **A8a** how a chat is set up, **A8b** what a turn shows | — |
| G2 | `/mode <name>` in an open chat | Changes nothing, and answers by opening *Continue in a new chat* prefilled with that mode. `/mode` alone lists the modes and names the chat's own. An open chat never changes mode | a person needs one chat to change mode for a reason carry-over cannot serve |
| G3 | How much a new chat carries | The whole old chat, one tickbox (*Carry this chat's exchanges*). A5's fitting keeps what the window allows, newest first, and *What the answer carried* says how many made it. **The drawing's "last 4" is replaced** | carrying a long chat makes answers measurably worse on the eval's conversation family |
| G4 | What the claim colours mean | Lilac: the sentence cites a passage (click opens it). Amber: it cites one that ranked low for the question, worded that way. Uncited sentences stay plain. **The drawing's amber key ("the reviewer did not find it in its source") is replaced**: the app runs no such check | the reviewer, or a later check, does verify a claim against its passage; then a third state can say so |

Fixed without a question (stated during the grill, not objected to):

- **Defaults live in `settings.json`** under a `new_chat` key (`{"mode_id", "folder_ids"}`). The
  file sits in the library's data home, so defaults are per library, and a folder default always
  names that library's folders. A default folder that no longer exists is dropped and the start
  screen says so; the defaults never widen (ADR-025 F2).
- **Chats from before A8** have no stored setup. Their line at the top is worked out from their
  last turn's record (its mode and scope). Nothing is written back.
- **The gate is tests plus a person driving the built page.** No eval run: no prompt, retrieval
  setting or locked setting changes, and carried exchanges go through the history path A5 already
  fits.

## The contract — A8a, how a chat is set up

### The stored setup (amends ADR-025 S9)

- `conversation_meta` gains `scope_json` (A7b's record shape: folders, or `null` for the whole
  library) and `continued_from` (a `session_id` or `null`), by an additive migration. `mode_id`
  is already there.
- The setup is written once, by the chat's first turn. Later turns read it and the request's
  scope field is ignored for a chat that has one, except a one-off `/folder <name>: <question>`
  (A7b), which scopes its own turn only and leaves the setup alone.
- **The amendment to ADR-025 S9.** The scope is now stored on the chat. The safeguard against a
  forgotten scope is the line at the top of the chat, which always names it, and each answer's
  scope note (unchanged). An addendum in ADR-025 records this, with the reason (the user,
  2026-11-20: pickers in the composer got in the way).
- Every scope rule of F2 holds: an unknown or emptied folder narrows or empties the search, never
  widens it, and the line says which folder held nothing.

### The start screen (`[s2]`, `[s3]`)

- Mode chips for every mode (built in first, then the person's own, marked *not evaluated*).
  *Search in* opens the folder list: *All documents* set apart. Ticking it unticks every folder,
  and unticking the last folder ticks it again. The count gives the union size (A7b's
  `doc_count`), each document once.
- Sample questions per built-in mode, three each, in the frontend. A person's own mode shows
  Research assistant's three.
- The start screen opens preselected from the defaults. *Make this my default* saves the current
  choice to `new_chat`.
- Both themes: the list, a ticked box and the chosen state follow night mode (T-015's styling
  half).

### The line at the top of a chat (`[s1-chatline]`)

- Shows the chat's mode and its folders (or *whole library*) with the document count, from the
  stored setup. *Continue in a new chat…* sits beside it.
- A chat with `continued_from` shows "Continued from “<title>” (<its mode> · <its folders>)",
  linking to that chat.

### Continue in a new chat (`[s5]`, `[s6]`)

- The dialog: the same choices, prefilled with the current chat's, and *Carry this chat's
  exchanges* (ticked by default).
- **Carried.** The new chat's memory starts with the old chat's exchanges, read from its answer
  records in order. Each exchange is the question and the answer text, without their citation
  blocks. A5's fitting applies as to any earlier message: oldest dropped first, said in *What the
  answer carried*. Nothing is summarised and no model is called to carry them.
- **Recorded.** `continued_from` names the old chat. Each new answer's record keeps how many
  carried exchanges its prompt held, beside A5's counts.
- **Shown.** The carried exchanges sit folded at the top of the new chat
  ("N earlier exchanges, carried as context ▸"), each opening as it was answered, with its
  citations (read from the old chat's records).
- The old chat is not changed.

### `/mode` (G2)

- `/mode` lists the modes and names the chat's own. `/mode <name>` on a chat that has turns
  answers "This chat answers as <mode>. To answer as <name>, continue in a new chat." and the
  window opens the dialog prefilled with `<name>`. On a chat with no turns yet, it sets the
  start screen's mode chip.

### The rail (`[s9]`)

- The sort menu gains *Group: No groups · By mode · By folder* (a chat in several folders sits
  under each) and *Only chats that cited*, which takes a folder or a document and calls A7b's
  filter. The choice is kept in the window's own preferences (`lib/shell/prefs`), not in the
  library.
- Each row's date line appends its labels: up to two names, then "+N".

## The contract — A8b, what a turn shows

### Claims in the answer (row 42, G4)

- The answer's sentences are matched to its stored claims by text, in order. A claim that is not
  found in the rendered text is not highlighted, and nothing is guessed.
- `ok` with a citation: lilac, clickable, and opens its first cited passage in the source panel,
  where that passage is marked. The open one is dark lilac. `weak`: amber, also clickable, with a
  tooltip "Cites a passage that ranked low for this question". `unsupported`: plain. The flagged
  list under the answer stays as it is.
- Evidence-only turns (row 40) and records answers have no claims, and show none.

### The records card (A6, `[s7]`)

- A rule-answered turn shows "Answered from the library's records · no model, no passages" and a
  card *What it looked up*: the lookup's name, its arguments, the row count, the first rows (each
  opening its document, folder or concept), and *N more*. Read from the record's `records_json`
  (A6). No new storage.

### What the answer carried (A5, `[s8]`)

- Inside *How this answer was made*: earlier messages kept of total, passages kept, shortened and
  left out, the window and where it came from (or the floor), the mode and its template hash, and
  (A8a) the carried exchanges kept. Read from fields A5 and A7 already record. The fitting note
  stays in the text as today.

### A reopened chat (row 44, `[s10]`)

- The turns of a reopened chat show their claims as above, and the claims' recorded decisions
  (accepted, rejected, edited), from `answer_claims`. The reviewer's verdict shows when the record
  holds one. See open question 1 for its memory.

## Hard constraints

- A scope never widens (F2). The stored setup, the defaults and a one-off command all narrow or
  empty; none falls back to the whole library when a folder is missing.
- An unscoped, default-mode turn on a new chat is byte-identical to today's (the turn-parity
  test, unchanged).
- No locked setting, prompt or retrieval knob changes. Carrying exchanges uses A5's history
  fitting and nothing else.
- Claim colours say only what the marker measures (G4).
- 0 documents: the start screen still shows (folders list empty, *All documents · 0*), and the
  first question gets today's empty-library answer.
- Frontend: strict checks as `apps/desktop/CLAUDE.md` sets (ADR-063, GLOSSARY U-001). Every new
  control is named in `GLOSSARY.md` before use.
- `structlog` only; `encoding="utf-8"` on file I/O; `uv run --no-sync mypy src`.

## Guard tests

- **A8a:**
  - The first turn writes the setup, and later turns ignore a request scope. A one-off
    `/folder X: q` scopes only its own turn.
  - A missing default folder is dropped and named.
  - `/mode <name>` with turns changes nothing and returns the carry-over prompt. On an empty
    chat it sets the mode.
  - Carry-over: the new chat's first prompt holds the old exchanges, fitted. On a fake small
    window the oldest are dropped and counted. `continued_from` is set, and the old chat's rows
    are unchanged.
  - Pre-A8 chats get their line from their last record.
  - Frontend: the start screen preselects the defaults; ticking *All* unticks folders and the
    reverse; the composer has no picker; the dialog is prefilled.
- **A8b:**
  - Claims are matched by text. An unmatched claim is not highlighted. `weak` is amber with its
    tooltip, `unsupported` is plain.
  - The records card renders from `records_json`.
  - The carried card shows A5's counts.
  - A reopened chat shows its claims' decisions.
- **Both:** the turn-parity test passes unchanged. The chat-UI checks run under the fetch mock
  (no real turn is sent).

## Done when

- Each session's guard tests pass (A8a adds: *Continue this chat* on a reopened chat carries its own exchanges, fitted, open question 1), with `pytest`, `mypy src`, `pre-commit`, CI's bandit command,
  `docs_check --strict`, `integrity_check`, the secret-scan gate, both audits, `npm test` and
  `svelte-check`.
- The built page is driven by a person in both themes. The driven check for A8a covers
  `[s2]`–`[s6]` and `[s9]`; for A8b, `[s1]`, `[s7]`, `[s8]` and `[s10]`. The wireframe's PNG is
  compared with a screenshot of the running app.
- DEVLOG entries, an `## As built` section here per session, ROADMAP rows 103, 42 and 44 marked,
  ADR-025's addendum (A8a), `docs/how-answers-work.md` and `GLOSSARY.md` updated.
- Staged with a commit command handed back; nothing committed by the agent.

## Out of scope

- An editor for a person's own modes (routes exist since A7; there is no screen yet).
- The per-chat "no memory" switch (the user, 2026-11-12; in no spec yet).
- Group by in the Library, or labels a person types (ADR-066 choice 7, not taken).

## Open questions

| # | Question | State |
|---|---|---|
| 1 | Should *Continue this chat* on a reopened chat (`[s10]`) carry its own exchanges into memory, through the same carry-over mechanism, rather than start empty as `feature-conversation-resume.md` decision 1 says? | resolved, 2026-11-20: **yes** (the user accepted the spec as written). A8a builds it with carry-over; `feature-conversation-resume.md` gets a note that decision 1 is superseded |
| 2 | Sample questions for each built-in mode: their wording | settled in the A8a build (As built), pinned in `lib/chat/setup.test.ts`; the user reads them in the driven check |
| W1–W3, G1–G4 | The drawing's and the grill's branches | resolved, 2026-11-20 (Decisions above) |

## As built — A8a (2026-11-20)

Built to the A8a contract and open question 1; staged for the user's commit. A8b (claims in the
answer, the records card, the carried card, a reopened chat's claims) is not started.

**Backend.** `src/doc_assistant/chat_setup.py` holds the setup: `read_setup` (stored, else worked
out from the last record, else the default; reading writes nothing), `turn_setup` (a turn's setup,
stored on the chat if it was not), `continue_chat` and `carried_exchanges` (carry-over),
`carried_held` (the count each record keeps), `mode_command` (G2), `setups_for` (the rail's
groups, three queries for a page) and the New chat defaults (`settings.json` → `new_chat`). The
controller calls `turn_setup` at the top of every turn but `/mode`, and the turn searches the
stored folders; a one-off `/folder <name>: <question>` keeps its own scope (A7b). The carried
count goes in `context_json` as `carried: {exchanges, kept, dropped}`, only on a chat that
carried. Routes: `GET /api/conversations/scope` (the union for *Search in*),
`GET /api/conversations/{id}/setup`, `POST …/{id}/continue`, `POST …/{id}/resume`,
`GET|PUT /api/settings/new-chat`; `ChatRequest.mode_id`; `TurnResultPayload.mode_offer`;
`ConversationSummaryPayload.mode_name` and `folders`; `ConversationDetailPayload.setup`. ADR-025
has its addendum; `feature-conversation-resume.md` and `chat-modes.md` say what A8a superseded.

**Window.** `screens/chat/StartScreen.svelte` (mode chips, *Search in*, the samples, *Make this my
default*), `FolderTicks.svelte` (*All documents* set apart, the tick rules, the union line),
`ModeChoice.svelte` (*Answer as*), `NewChatDefaults.svelte`, `ContinueInANewChat.svelte`,
`ChatLine.svelte` (the line, *Continue in a new chat…*, continued-from, the folded carried
exchanges). State in `lib/chat/setup.svelte.ts`, rules in `lib/chat/setup.ts` (tested). The
composer is the text box and *Send* (and *Test override* while an override is set). The rail's
sort menu gains *Group* and *Only chats that cited*, kept in `lib/shell/prefs`; rows' date lines
carry their labels. The ticks and the chosen chip follow night mode through `accent-color` and the
accent's tint (T-015's styling half).

**Open question 2, the sample questions.** Research assistant: "What are the main themes across my
documents?", "What are the key findings, with citations?", "Which documents describe their methods
in detail?". Critical reading: "Where do my sources agree and disagree?", "What are the limits of
the evidence?", "Which claims rest on a single study?". A person's own mode shows Research
assistant's. Corpus-agnostic, so the drawing's "receptor density" became a methods question.

**Not foreseen by the spec.**

- **The carried exchanges reach the rewrite call only, at most three of them.** A5's history path
  is the rewrite call, and `context.fit_history` takes at most the newest six messages
  (`cap=6`, A5's structural constant). So the whole chat is carried into memory, but a turn's
  prompt holds at most its newest three exchanges, fewer on a small window; the answer call holds
  no history at all, as before A8. Each record says how many it held (`carried.kept`) and how many
  it did not (`carried.dropped`). G3's reopen condition is the place to revisit it.
- **A third column.** `conversation_meta.carried_exchanges` records how many exchanges *Continue in
  a new chat* carried, so a reopened continued chat can say it carried them before any answer
  record holds the count. Additive like the other two.
- **The setup is written at the start of the first turn**, before retrieval, not after its
  record: a turn that fails still leaves the chat's setup stored, which is what the window then
  shows. A chat from before A8 has its worked-out setup stored by its first new turn (reading it
  still writes nothing).
- **`Session`'s default id gained a random suffix.** Two sessions made in the same second shared an
  id, harmless before, but a stored setup now belongs to the id.
- **Defaults whose folders are all gone keep the gone ids**, which resolve to nothing: "dropped"
  would leave no folder, and no folder reads as the whole library, which the hard constraint
  forbids. The start screen says so, and ticking a folder replaces them. `settings.json` keeps the
  folders' and the mode's names beside their ids, so a gone one can still be named.
- **`/mode <name>` on an empty chat** sets the chip and its reply shows as the start screen's note,
  not as a turn (a turn would have closed the start screen whose chip it set).
- **A7's `PATCH /api/conversations/{id}` with `mode_id`** is a 409 on a chat with turns (G2); on an
  empty chat it still sets the mode.
- **The start screen shows under the first-run steps** when they are left (it was replaced by them),
  so the 0-document start screen the hard constraints ask for is there.
- **Group by folder** groups by the folders a chat searches (its setup), not by what it cited (its
  labels, which the row's date line shows); a whole-library chat sits under "Whole library".
- **Working out a pre-A8 chat's setup** skips a rule-answered turn (it records no scope) and a
  one-off command's turn (its scope was its own); a concept scope was always a command.
- **Carry-over leaves out a turn that retrieved nothing**, as the live history does.
- **`/resume` on a chat live in this process** keeps the memory the process already holds.
- **No PNG of the chat wireframe exists** (only the scene and its reading), so the driven check
  compared the running page with the reading's elements.

**Gate.** Tests and a driven check, $0, no eval run: `tests/integration/test_chat_setup.py` (28),
A7's `/mode` and PATCH tests rewritten to G2, `test_retrieval_scope`'s S9 test rewritten to the
addendum (the scope stays with its chat and does not leak into another), `lib/chat/setup.test.ts` (13). The turn-parity test
passes unchanged. The driven check ran the dev build over a scratch data home of invented
documents, with `/api/chat` mocked in the page (no turn reached the backend); the counts are in the
DEVLOG entry.

## As built — A8b (2026-11-20)

Built to the A8b contract; staged for the user's commit.

**Backend.** `src/doc_assistant/turn_details.py` reads what a turn shows besides its answer, all from
what the answer path already records: `TurnClaim` (an `answer_claims` row: its text, cited passages,
marker, recorded decision), `claims_for` and `reviews_for` (one query each for a chat's answers),
`describe_records` (a records answer's `records_json`, its first five rows named as documents,
folders or concepts, three queries) and `read_context`. A live turn's `TurnResult` gains `claims`
(every claim, from the same segmentation that writes `answer_claims`), `records`, `context` (the
`context_json` it recorded), `chat_mode` and `prompt_version`; a reopened turn gains the same plus
`review`. Wire: `TurnClaimPayload`, `RecordsPayload`, `ReviewPayload` on `TurnResultPayload` and
`ConversationTurnPayload`. No new storage, no prompt or retrieval change.

**Window.** `Markdown.svelte` colours the claims after it renders the answer: their text is
matched in order against the shown text, citation buttons left out (`lib/chat/claims.ts`, tested);
a claim not found is left plain. Lilac for `ok` with a citation, amber for `weak` with its tooltip,
plain for `unsupported`; a claim span is a button to mouse and keyboard and opens its first cited
passage; the open one is drawn darker (`chat.activeClaim`). `RecordsCard.svelte` is *What it looked
up*; `CarriedCard.svelte` is *What the answer carried*, inside the fold now named *How this answer
was made* (it read *Provenance & usage*; GLOSSARY U-008). `ReadonlyTurn.svelte` shows a reopened
turn's claims, the decisions recorded on them, the reviewer's recorded verdict, its lookup and its
fold. The words are `lib/chat/turnDetails.ts` (tested).

**Not foreseen by the spec.**

- **"Where that passage is marked."** The source panel opens the cited passage, which is the unit
  a claim cites; nothing narrower is marked inside it. A passage carries no character offsets, so
  the sentence that supports a claim cannot be located in it (the memory note on the source viewer
  and the chunk locator says the same).
- **A claim's highlight is broken by its citation buttons**: the sentence's text and its full stop
  are two spans either side of `[n]`, both coloured, both opening the passage. The button keeps
  opening its own source.
- **The fold was renamed** from *Provenance & usage* to the drawing's *How this answer was made*;
  `docs/release-ux-checklist.md` follows, and the old label is forbidden in the glossary.
- **The records card names five rows** (`RECORD_ROWS_SHOWN`) and says how many more are in the
  answer above; the rows are the lookup's recorded ids, so a document deleted since is left out.
- **The prompt version** shown beside the mode is the turn's recorded `prompt_version` (A7 folds
  the template hash into it), its first eight characters.
- **The citation pattern** is copied into `claims.ts` (a sibling import would need the `.ts`
  extension the type check refuses); `claims.test.ts` pins it equal to `citations.ts`'s.

**Gate.** Tests and a driven check, $0, no eval run: `tests/integration/test_turn_details.py` (8),
`lib/chat/claims.test.ts` (7), `lib/chat/turnDetails.test.ts` (4). The turn-parity test passes
unchanged. Driven over the scratch data home with a seeded chat (claims with decisions, a review, a
records lookup, fitting counts) and a mocked live turn: `[s1]`, `[s7]`, `[s8]`, `[s10]`, both
themes, the amber tooltip, a claim opening its passage, a records row opening its document.
