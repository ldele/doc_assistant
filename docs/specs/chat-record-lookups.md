<!-- status: active · updated: 2026-11-19 (built; the gate recorded under As built); 2026-11-19 (accepted by the user as written, open question 1 as recommended; written after the user's five answers in A6's scoped grill) · class: living -->

# Spec — The chat answers record questions from the records, by rule and command (ROADMAP 102, session A6)

> The executor brief. Target executor: **Claude Code**. Session A6 of the plan
> (`docs/plans/PLAN_2026-10-31_next-two-releases.md`, local). It reads, and must not re-litigate:
> ADR-066 choice 5 (lookups by rule and command; the model is never asked to choose) and its seam
> table (the lookup functions are the tools' own), ADR-054 (a *term* is not a concept),
> `docs/knowledge-layer.md` (why gaps stay out), and `docs/specs/eval-harness-chat.md` (the
> gate's families and scorers). **Accepted by the user, 2026-11-19, as written: buildable.**

## In one sentence

A question about the library itself (a count, a folder, one document's metadata, what is missing,
a curated concept) is answered from the library's records by a fixed rule or a named command,
with no model call, the same on every model, and the answer and its provenance say so.

## Today (read in the code, 2026-11-19)

- **The router** (`query_router.py`) has seven patterns: latest, how many, list/show, what is in
  the library, library stats, broken/marginal, document count. A trailing topical word ("about",
  "on", "with", …) declines the match, so a content question goes to retrieval.
  `answer_library_query` answers from `library_summary()` and `list_documents()`.
- **The controller** (`chat_controller/controller.py:353`) sends a matched question to the
  router and wraps the text with `_command_result(route="library")`: **no answer record is
  written**, so the turn is missing from `/records`, from the conversation's history and from
  `get_conversation`.
- **The commands** (`commands.py`) include `/library [filter]` and `/document <id prefix>`.
- **The read surface** (`read_surface/records.py`, A3) already holds the lookups:
  `list_folders`, `list_documents(folder_id=…)`, `get_document(id)`, `get_concept(name or form)`
  (curated concepts only; a term returns `None`), `list_concepts`.
- **The eval's records family** (`tests/eval/cases.records.yaml`, 8 cases, `expected_route:
  library`) is answered by the router in 2 cases; the other 6 fall through to retrieval.

## Decisions — the user, 2026-11-19 (A6's scoped grill)

| # | Question | Decided | Why |
|---|---|---|---|
| 1 | Which record questions do the rules answer? | **Records and concepts:** counts and health, folders (count, listing), one document's metadata (title, authors, year, type), documents missing a field, and a curated concept's forms and chosen definition. **Gaps stay out** | Gap signals are of uneven trust (`docs/knowledge-layer.md`); an answer "from the records" would state them as fact |
| 2 | How does a rule find a named document? | **By its recorded title, else its filename.** One match: answer from its record. Several: list them and ask which. None: the question goes to retrieval | The same on every model, testable at no cost, no model or embedding call. A search would pick a document that merely cites the paper and report its year as the record |
| 3 | What does a rule-answered turn leave behind? | **An answer record and a note.** The record has no passages and no model; a new nullable column names the lookup, its arguments and the ids it read. The answer ends with a line saying it came from the library's records. Labels stay derived from cited passages only | ADR-066 choice 5: the answer and its provenance say it came from the records. A lookup naming every untitled document must not label the conversation with them |
| 4 | Which commands? | **`/folder <name>`, `/concept <name>`, and `/document` by title** (as well as by id prefix). Each calls the same lookup as the rules. The scoped question (ask within a folder or a concept's documents, choice 7) is **parked to A7**, beside the labels it feeds | A way in when a rule misses a phrasing, with one code path |
| 5 | The gate? | **Records exact, the others no drop.** See *Done when*. The plan's clause "a local model that cannot do it says so" is set aside: no model takes part in a rule answer | It returns if a routing call or tool calling is added (ADR-066's seam) |

## The contract

### The lookups

- One module in `src/doc_assistant/` holds the rules: each maps a question to **one lookup**, a
  call of a `read_surface` function (or `library_summary` for counts and health) with its
  arguments. The rule module formats the result; it adds no query of its own where a read-surface
  function serves, so the MCP tools and the chat read the same records.
- The lookups: library count, health and latest (today's); a folder's document count; a folder's
  documents; one document's title, authors, year or type; the documents with no title, no year
  or no authors; a concept's forms and chosen definition.
- The topical-word rule stays: "which documents in the Demo corpus folder are about attention?" is
  a content question and goes to retrieval.
- A folder is found by its name, case not mattering. An unknown folder says so and lists the
  folders; it never falls through to retrieval, since the question was about a folder.

### Finding a named document (decision 2)

- The named span is the words the question gives for the document ("the Pointer Networks
  paper", "the authors of *X*"). Its content words are compared, case and punctuation aside,
  with the document's recorded title; where no title is recorded, with its filename stem.
- **A match** is a document whose title (or stem) contains every content word of the span.
  Exactly one: the rule answers from it. Several: the answer lists them (title, filename, id
  prefix) and asks which, and records that it asked. None: the question goes to retrieval,
  unchanged.
- No fuzzy score and no threshold: a match holds or it does not (no corpus-tuned numbers, the
  robustness contract).

### What a rule-answered turn records (decision 3)

- An answer record (`provenance.record_answer`) with no retrieved chunks, no model name and no
  tokens. Its route stays `library` (the eval's `route` scorer reads it).
- A new nullable column, `AnswerRecord.records_json`, migrated additively in
  `db/migrations.py` like A5's `context_json`: the lookup's name, its arguments, the ids it read
  and how many rows it returned. Never the question's text beyond what `query` already holds.
- The answer's last line says it came from the library's records, without a passage read or a
  model called. The wording is settled in the build and pinned in a test.
- The turn appears in `/records`, the conversation's history and `get_conversation`. It adds no
  label to the conversation.

### Commands (decision 4)

- `/folder <name>`, `/concept <name>`, and `/document <id prefix or title>` call the same lookups
  and use decision 2's match for a title. `/help` lists them.
- A command that calls a lookup writes the same record as a rule (open question 1, resolved).
  Other commands are unchanged.

### An empty or partial library

Every lookup answers at 0 documents without an error: a count of zero, an empty folder, "no
document is recorded under that name", "no concept of that name has been taken on" (a term is
not a concept, ADR-054). A field that is not recorded is said to be not recorded, never guessed.

## Hard constraints

- No model or embedding call on a rule or lookup-command path.
- No write but the answer record. No change to the read-surface tools' shapes (ADR-066 choice 8).
- All logic in `src/doc_assistant/`; `apps/` unchanged beyond what a new route field needs.
- Locked settings untouched. `structlog` only. `encoding="utf-8"` on every file read.

## Guard tests

- Each of the 8 records cases' questions reaches the expected lookup, over a fixture library.
- **No rule catches a content question:** every query of the single, conversation, absent and
  hostile case files goes to retrieval.
- A title match with several documents lists them; with none, the question goes to retrieval.
- Each lookup on an empty library: the honest answer above, no error.
- A rule-answered turn writes an answer record with `records_json`, no chunks and route
  `library`, and `get_conversation` returns it.
- Each new command calls the same lookup as its rule (one test each); `/concept` on a term says
  it is not a concept.
- The note's wording, pinned.

## Done when

- The guard tests pass.
- **Records family:** all 8 cases routed to `library` with `contains_all` on each, on both
  columns of the panel (`llama3.1:8b`, `qwen2.5:7b`, controller adapter, `--repeat 3`).
- **Absent, single and conversation:** no drop beyond the interval against the baseline recorded
  on 391ea0e (`tests/eval/baselines/chat_baseline_2026-11-19.md`; store `C:/Projects/eval-h4/eval_h4.duckdb`, notes `Baseline 391ea0e: <family>`),
  per column.

## Out of scope

- Gaps and the library summary as lookups (decision 1).
- The scoped question command and labels a person types (A7, ADR-066 choice 7).
- A routing call or tool calling by the model (ADR-066's seam; reopens decision 5).
- Keywords, tags and epistemic markers on a concept (ADR-066's seam table).
- Any write tool, and any change to the MCP tools.

## Open questions

| # | Question | State |
|---|---|---|
| 1 | Do the lookup commands write an answer record, as a rule-answered turn does? | resolved, 2026-11-19: yes, the same record, since they call the same lookup; other commands unchanged (the user accepted the spec as written) |
| 2 | The wording of the closing line | settled in the build, pinned by a test |
| G1–G5 | The grill's five branches | resolved, 2026-11-19 (Decisions above) |

## As built (2026-11-19)

**Code.** `src/doc_assistant/record_lookups.py` holds the rules, the lookups, their formatting
and the note. `answer_record_question(text)` returns a `RecordAnswer` (the text, the lookup's
name, its arguments, the ids the answer names, the row count) or `None`, which sends the question
to retrieval. The rules run in a fixed order: a folder's count, a folder's list, the folders,
documents missing a field, the concepts, one concept, one document's field, then today's seven
patterns (`query_router.is_library_query`). Folders come first, so a folder count is never the
library's (KI-95). `query_router.library_lookup` still answers count, health, latest and the
overview, and now also says what it read. The controller (`_records_result`) writes the answer
record with `records_json` = `{lookup, args, ids, rows, via}` (`via`: `rule` or `command`;
`asked: true` when several documents matched). It has no passages, no model and no tokens, the
route is `library`, and the turn joins the session's history.

**Settled in the build:**

- **The note** (open question 2): *From the library's records: no passage was read and no model
  was asked.*, in italics, as the answer's last line (`RECORDS_NOTE`, pinned by
  `test_the_records_note_wording_is_pinned`).
- **A named span** is what the question gives after the field: the tail ("in my library",
  "according to my library", "as my library records it") and words that do not name a document
  ("the", "paper", "document", …) are dropped. A topical word after a document noun ("the paper
  about X") declines, as `query_router`'s rule does. Matching compares words, not substrings.
- **A document with no title** is one whose read-surface title is its file name (`DocumentItem`
  falls back to it). The missing-title lookup and the stem match both read it so.
- **`/document` by id prefix keeps today's details view** (health, chunks, ingestion history),
  which writes no record. Only an argument that is not a matching id prefix goes through the
  title lookup and is recorded. No rule takes an id, so "the same lookup as its rule" applies to
  the title.
- **A concept rule needs the word "concept"** (or "my/the chosen definition of"), so *what is the
  definition of attention?* stays a content question. An unknown concept is then said to be
  unknown; it does not fall through.
- **`ids`** are the records the answer names, capped at the 50 a list answer shows; `rows` counts
  every row.
- **Cost at scale.** A document-field rule, a missing-field rule and `/document` by title read
  every document a page (100) at a time through `list_documents`: about 100 small reads at 10,000
  documents. Bounded by the page and not by memory.

**The gate (2026-11-19, `tests/eval/baselines/a6_record_lookups_2026-11-19.md`).** Records: 8 of
8 routed to `library` with `contains_all` 1.000 on both columns, every trial (391ea0e: 3 of 8;
0.500 and 0.625). Absent, single and conversation: no drop beyond the interval on either column;
absent's citation coverage on qwen-7b reads higher at this n. Two cases flipped once in three
trials, each matched by another flipping back; none of those turns reached a rule. $0, 2,521 s,
local Ollama, reviewer pinned to `llama3.1:8b`. Guard tests: `tests/unit/test_record_lookups.py`.
