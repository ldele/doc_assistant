<!-- status: active · updated: 2026-11-20 (built, A7b: As built added); 2026-11-20 (accepted by the user as written; G2's note uses the graph's staleness verdict, as the graph stores no build date) · class: living -->

# Spec — A question scoped to several folders or to a concept, and the chat list's labels (ADR-066 choice 7, T-015's backend, session A7b)

> The executor brief. Target executor: **Claude Code**. Session A7b of the plan
> (`docs/plans/PLAN_2026-10-31_next-two-releases.md`). ADR-066 choice 7 and ADR-025 F2
> (`docs/specs/feature-corpus-folders-scope.md`) are the decisions this builds on; this file is the
> code-level contract.

## In one sentence

A turn can be scoped to **several folders**, which retrieval searches as one set of documents,
each document once. A command asks a question within **one folder or one concept's documents**.
The app's conversation list gains each conversation's **labels** and a **filter** by document or
folder, worked out from the answer records each time it is read.

## Today (read in the code, 2026-11-20)

- **One folder per turn.** `ChatRequest.scope_folder_id: str | None` (`apps/api/models/chat.py:67`)
  reaches `ChatController.handle(…, scope_folder_id)` and `compare(…, scope_folder_id)`
  (`chat_controller/controller.py:219,316`). `_resolve_scope` (`chat_controller/helpers.py:235`)
  turns it into a `frozenset` of document hashes and a `ScopeView(folder_id, folder_name,
  doc_count)` (`views.py:18`). An unknown, deleted or empty folder gives an **empty** set, never
  `None`: the turn retrieves nothing (F2, S3).
- **The record.** `_scope_dict` writes `{folder_id, folder_name, doc_count}` to
  `answer_records.retrieval_scope_json`. Read back by `conversations.py:404` (history),
  `provenance.py:321`, and `read_surface/conversations.py:303`, where MCP's `get_conversation`
  gives each turn `folder: str | None` (frozen, ADR-066 choice 8).
- **The window.** `Chat.svelte:177` is a native `<select>` (one folder, *All documents* first);
  `App.svelte` holds `chatScopeFolderId` in memory only (S9). `Turn.svelte:45` and
  `ReadonlyTurn.svelte:36` say "Searched **<folder>** only — n documents".
- **Commands.** A6's `/folder <name>` and `/concept <name>` answer by rule
  (`record_lookups.answer_command`, `controller.py:351`). Neither asks a question.
- **Labels.** The read surface already derives them and filters by them:
  `read_surface/conversations.list_conversations(document_id=, folder_id=)`. The app's
  `/api/conversations` (`apps/api/routers/conversations.py:26`) has neither.
- **A concept's documents.** `concept_presence` (`db/models.py:636`), rebuilt with the graph,
  counts a concept's **exact** forms; broad-form documents are listed beside it, never in it.
  `docs/knowledge-layer.md` rates presence trustworthy.

## Decisions — the user, 2026-11-20 (A7b's scoped grill)

| # | Branch | Decision | Reopens if |
|---|---|---|---|
| G1 | A folder among several is unknown, deleted or empty | Search the union of the folders that still hold documents; the scope note names the one that held nothing. All of them bad: the turn searches nothing, as today | a person reads a turn's answer as covering a folder the note says was dropped |
| G2 | Which documents are a concept's | `concept_presence`, exact forms only. A concept with no presence, or a library whose graph was never built, gives an empty scope: nothing is searched, and the answer says to rebuild the graph. When the graph is out of date (`concept_graph_view`'s staleness verdict), the note says so; the graph stores no build date, since a rebuild is byte-identical | presence stops being rated trustworthy in `knowledge-layer.md` |
| G3 | The command's syntax | `/folder <name>: <question>` and `/concept <name>: <question>`. With no colon, A6's lookup runs unchanged. Names match as A6's do; several matches → it asks which, and asks no question. One folder per command; several come from the picker | folder names with a colon turn out to be common |
| G4 | How much of the chat list lands here | `/api/conversations` gains labels and a filter by document or folder, through the read surface's derivation. The controls that show and use them are A8's, drawn with the picker | — |
| G5 | The gate | Tests only, no eval run: no prompt, retrieval setting or locked setting changes | a later change makes a scope alter how retrieval ranks, not only which documents it sees |
| G6 | MCP's frozen turn shape | Add `scope: {folders: [names], concept: name \| null, documents: n} \| null` to `get_conversation`'s turn. `folder` keeps its meaning: filled only when the turn ran on exactly one folder, else `null`. Same version (a field added) | a client is found that relies on `folder` being set for every scoped turn |

Fixed without a question (stated in the grill, not objected to):

- **The wire field becomes a list.** `scope_folder_ids: list[str] | None` replaces `scope_folder_id`
  on `ChatRequest` and the compare request. The desktop app is the only client, and choice 8 leaves
  the app's routes free. `None` or an empty list both mean the whole library (the picker's *All*).
- **Old records read as before.** One reader turns either JSON shape into a `ScopeView`.
- **Per turn only.** A command's scope lasts its own turn and is never stored on the session (S9).

## The contract

### The scope (G1)

- `ScopeView` becomes `folders: list[ScopeFolder]`, `concept: ScopeConcept | None`, `doc_count: int`
  (the size of the union, the number retrieval actually saw). `ScopeFolder` is `(folder_id,
  folder_name | None, doc_count)`, per folder before the union; `folder_name is None` = deleted.
  `ScopeConcept` is `(concept_id, name, graph_stale: bool)`, the stale flag read from `concept_graph_view`'s staleness verdict. A turn has folders or a concept,
  never both.
- `_resolve_scope(folder_ids: list[str] | None, concept_id: str | None = None)`. `None` or `[]` and no
  concept → `(None, None)`: the unscoped path, byte-identical to today. Otherwise the union of the
  folders' document hashes as one `frozenset` (each document once by construction). Duplicate ids
  in the request count once. A folder that is unknown, deleted, empty or fails to read
  contributes nothing and stays in `folders` with its count of 0, so the note can name it. If the
  union is empty the turn retrieves nothing, as F2 does today.
- One query for the union (`document_folders.folder_id IN …`), not one per folder.

### The record (G6)

- `_scope_dict` writes `{folders: [{folder_id, folder_name, doc_count}], concept: {concept_id,
  name, graph_stale} | null, doc_count}`.
- `read_scope(raw) -> ScopeView | None` reads both this shape and the old `{folder_id,
  folder_name, doc_count}`. Every reader above goes through it: history, provenance, the read
  surface.
- MCP `get_conversation`: each turn gains `scope` (G6). `folder` is the one folder's name when
  `len(folders) == 1`, else `null`. `test_read_surface_shapes.py` gains the field, and no existing
  field changes.

### The commands (G3)

- In `answer_command`'s caller: `/folder` or `/concept` whose argument contains `: ` splits at the
  **first** `: ` into a name and a question. An empty question after the colon → A6's lookup on
  the name.
- The name resolves through A6's matchers (`record_lookups`). No match → A6's "unknown folder"
  answer, which lists the folders (for a concept, the concepts). Several matches → A6's "which one
  do you mean?" answer. In both cases, no question is asked and no retrieval runs.
- One match → the question runs as an ordinary RAG turn (`_handle_rag`) with that scope. It is
  recorded like any turn, its scope in `retrieval_scope_json`, and its `query` is the question
  without the command. The command text is kept as the record's `original_query`.
- A concept with no presence (G2): the turn runs with an empty scope, retrieves nothing, and the
  answer says the concept has no documents yet and the graph needs a rebuild (no model call).

### The notes in the window

- `_scope_note`, `_scope_label`, `Turn.svelte` and `ReadonlyTurn.svelte` name every folder:
  "Searched **A**, **B** only: 14 documents". A dropped folder: "**C** held no documents and was
  left out." A deleted one: "a folder that no longer exists". A concept: "Searched the documents
  where **<concept>** is present: 6 documents", plus "The concept graph is out of date; rebuild it to refresh this list." when `graph_stale`.
- The picker stays a single choice in this session. It sends `[id]` or `null`. The tickboxes are
  A8's.

### The chat list (G4)

- `GET /api/conversations?document_id=&folder_id=` — both optional, both together = both must
  hold, as in the read surface. Each `ConversationSummaryPayload` gains `labels: {documents:
  [{document_id, title}], folders: [{folder_id, name}]}`.
- The route calls the read surface's derivation (`read_surface/conversations`), and there is no
  second implementation. Pinned and archived flags stay as they are.

## Hard constraints

- A scope never widens (F2). A bad folder, an unknown concept or a read error narrows the search
  or empties it; nothing ever falls back to the whole library.
- An unscoped turn is byte-identical to today, and the turn-parity test stays green unchanged.
- No locked setting, prompt or retrieval knob changes.
- 0 documents: a scoped turn on an empty library never reaches retrieval, as today. 10,000
  documents: the union and the filtered list are each one query, timed in a test.
- Labels are derived on every read; nothing new is stored (choice 7).
- `structlog` only; `encoding="utf-8"` on any file I/O; `uv run --no-sync mypy src`.

## Guard tests

- Union: two folders that share a document → it is searched once, and `doc_count` counts it once.
- G1: one unknown and one live folder → the live one's documents only, and the note names the
  unknown one. All unknown → empty, no retrieval. Duplicate ids → counted once.
- G2: concept with presence → exactly its exact-form documents (a broad-form document is absent).
  No presence, or no graph → empty, no retrieval, and the rebuild line is shown.
- G3: `/folder Thesis: what is X?` scopes to Thesis. `/folder Thesis` stays A6's lookup, and so
  does `/folder Thesis:` with nothing after the colon. An ambiguous or unknown name asks or lists,
  and retrieval is never called (spy). A question containing a further `: ` stays whole.
- The record round-trips in the new shape, and an old-shape row reads as a one-folder scope.
- MCP shape: `scope` is present, `folder` is set for one folder only, and no field is removed.
- `/api/conversations`: labels are present; the filter by folder and by document, alone and
  together, matches the read surface's result.
- Scale: on an invented library of 10,000 documents in 20 folders, resolving a 5-folder union and
  the filtered list are each under a stated bound, recorded in the DEVLOG entry.
- Parity: the unscoped turn test, unchanged.

## Done when

- The guard tests pass, with `pytest`, `mypy src`, `pre-commit`, CI's bandit command, `docs_check
  --strict`, `integrity_check`, the secret-scan gate and both audits.
- The desktop app sends a list and its scope lines name several folders (`npm test`, `svelte-check`).
- The DEVLOG entry, a `## As built` section here, ROADMAP rows for choice 7 marked, and
  `docs/how-answers-work.md`'s scope section updated.
- Staged with a commit command handed back; nothing committed by the agent.

## Out of scope

- The picker's tickboxes, *All* set apart, night mode on a folder choice (T-015's front half, A8).
- The chat list's label and filter controls (A8, G4).
- MCP `search` taking several folders (it keeps `folder_id`); a later, additive parameter if wanted.
- Several folders in one command, or a folder and a concept together.
- Tags a person types, or tags from keywords (choice 7, not taken).

## Open questions

| # | Question | State |
|---|---|---|
| G1–G6 | The grill's six branches | resolved, 2026-11-20 (Decisions above) |
| 1 | The wording of the scope lines | settled in the build, pinned by a test |

## As built (2026-11-20, A7b)

Built to the contract above; staged for the user's commit. What the build settled, and what the
spec did not foresee:

- **One module for the scope.** `src/doc_assistant/retrieval_scope.py` holds `ScopeView`,
  `ScopeFolder`, `ScopeConcept`, `scope_dict`, `read_scope` and the words (`scope_summary`,
  `scope_label`, `GRAPH_STALE_NOTE`). `chat_controller/views.py` re-exports the three types. It
  imports nothing from the answer path, so `provenance`, `conversations` and the read surface
  read through `read_scope` without an import cycle (`chat_controller` imports `provenance`).
  `read_scope` reads text that is not JSON, or a value of neither shape, as an **empty** scope,
  never as `None`: a record that says it was scoped is never shown as a whole-library one.
- **The union.** `library.folders_doc_hashes(ids)` reads the folders' names, then their
  non-archived members' hashes with one `document_folders.folder_id IN …` query: two queries
  whatever the number of folders (a test counts them). `_resolve_scope` keeps the first-asked
  order, counts duplicates once, and keeps a folder that added nothing with its count of 0.
- **A concept's documents.** `concept_graph_view.concept_doc_hashes` (one join of
  `concept_presence` on `documents`, archived left out) and `graph_staleness()` (the verdict
  alone, `None` when the graph was never built). A graph never built gives an empty scope even if
  presence rows exist.
- **The words** (open question 1), pinned in `test_scoped_questions.py` and in the desktop's
  `lib/chat/scope.test.ts`: "Searched **A**, **B** only: 14 documents." · "**C** held no documents
  and was left out." · "A folder that no longer exists was left out." · "Nothing was searched."
  when no folder held a document · "Searched the documents where **X** is present: 6 documents."
  plus `GRAPH_STALE_NOTE`. The provenance card's line keeps its "🔎 **Retrieval scope (this answer
  only):**" lead. A concept with no documents answers "The concept **X** has no documents yet: the
  concept graph lists none where it is present, so nothing was searched. Rebuild the concept
  graph, then ask again." A one-folder scope keeps its pre-A7b no-sources sentences.
- **The commands.** `record_lookups.scoped_question(cmd, arg)` returns `None` (A6's lookup runs),
  a `RecordAnswer` (unknown or several matches: recorded as a command turn, nothing retrieved),
  or a `ScopedQuestion`, which the controller runs through `_handle_rag` with `typed=` the command
  text: the record's `query` is the question, `original_query` the command, the history holds the
  question alone and the export shows the command.
- **Not foreseen: the question's case.** `commands.parse_command` lowercases its argument, so the
  controller re-reads the argument from the typed text before splitting: the question is searched
  as typed.
- **Not foreseen: `/folder Thesis:`.** The argument then ends with `:` and holds no `: `, so it
  reached A6's lookup as the name "Thesis:". `answer_command` now drops one trailing colon from a
  `/folder` or `/concept` argument (G3's reopen condition covers folder names with colons).
- **Not foreseen: a command beside the picker.** The command's scope replaces the picker's for its
  turn; it does not intersect with it. Both name one place, and the command is the later, more
  explicit choice. Never stored, as S9 asks.
- **Ambiguity.** Root folder names are unique, case aside (`create_folder`), and folders are flat
  in v1, so two folders can only share a name once nesting exists; the "which one" answer is built
  and tested against two such folders. `get_concept` returns one concept or none, so a concept
  never asks which; an unknown concept gives A6's sentence plus the concepts taken on.
- **The record.** Unchanged in storage: the new shape goes in the same `retrieval_scope_json`
  column. `AnswerProvenance.retrieval_scope` is the new-shape dict, read through `read_scope`;
  `ConversationTurn.retrieval_scope` is now a `ScopeView`.
- **MCP.** `read_surface.shapes.TurnScope(folders: list[str | None], concept, documents)` on
  `Turn.scope`; `folder` is set only for exactly one folder. A deleted folder is `null` in
  `folders`. `test_read_surface_shapes.py` gains `Turn.scope` and the `TurnScope` entry; nothing
  removed, version unchanged.
- **The chat list.** `read_surface.conversations` now exposes `cited_documents`, `keep_cited` and
  `labels_for` (two queries for a whole page's labels, where it was two per conversation), and its
  own `list_conversations` uses them; `conversations.list_conversations(document_id=, folder_id=)`
  applies the filter **before** its 100-row cap, so an older match is not cut off.
  `ConversationSummaryPayload.labels` and `core/types/conversations.ts` carry them.
- **The window.** `screens/chat/ScopeChip.svelte` is the one chip both turns render (the copied
  `.scopechip` rule went with it: `COPIED_RULES` 179 → 178); its icon is `tag` for a concept.
  `App.svelte` sends `[id]` or `null`.
- **Scale** (10,000 documents in 20 folders, one in ten in two; 1,000 conversations citing ten
  documents each): the 5-folder union 0.006 s against a bound of 0.5 s, the folder-filtered list
  0.020 s against 2.0 s, on the dev machine.
- **Gate (G5):** tests only, $0. See the DEVLOG entry for the suite's count.
