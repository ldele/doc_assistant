<!-- status: active · updated: 2026-11-20 (built; the gate recorded under As built); 2026-11-19 (accepted by the user as written; written after the user's seven answers in A7's scoped grill) · class: living -->

# Spec — Chat modes change only the framing, and evidence-only calls no model (ROADMAP 37 and 40, session A7)

> The executor brief. Target executor: **Claude Code**. Session A7 of the plan
> (`docs/plans/PLAN_2026-10-31_next-two-releases.md`, local). It reads, and must not re-litigate:
> ADR-066 choice 6 (a chat mode is a name and the framing above the citing block; two built in;
> a person's own is marked *not evaluated*; the template is hashed per turn) and its seam table
> (a mode with a first step, then skills, come later), and `docs/specs/eval-harness-chat.md` (the
> gate's families and scorers). **Accepted by the user, 2026-11-19, as written: buildable.**

## In one sentence

A chat mode is a named persona-and-task text that replaces the first line of the answer prompt,
while the grounding rule, the citing block and the source fence stay the same in every mode; each
turn's provenance names the mode and hashes the prompt it actually used.

## Today (read in the code, 2026-11-19)

- **The answer prompt** (`prompts.py:42`, `ANSWER_PROMPT`) opens with one persona line ("You are a
  helpful research assistant."), then the grounding rule ("Answer the question using ONLY the
  sources…"), then the citing block that `synthesis._CITATION_RE` parses, then the fence sentence
  and `{context}` / `{question}`.
- **The hash** is taken once, when the controller is built
  (`chat_controller/controller.py:144`, `template_hash(ANSWER_PROMPT + REWRITE_PROMPT)`), and fed
  to `provenance.prompt_version_hash`. A prompt that changed between turns would still report the
  first hash: ROADMAP row 37's "or provenance lies".
- **Evidence only** is `SYNTHESIS_MODE=human` (`config.py:293`) or a per-turn override. It skips
  the answer call (`controller.py:717`) but still runs the history rewrite (`rag.rewrite`,
  `controller.py:658`) and multi-query when it is on: two model calls in a mode meant to cost
  nothing (row 40). It is a *synthesis mode*, not a chat mode, and stays a separate switch.
- **Conversations** keep their own row in `conversation_meta` (`db/models.py:1066`).

## Decisions — the user, 2026-11-19 (A7's scoped grill)

| # | Question | Decided | Why |
|---|---|---|---|
| 1 | What may a mode's text change? | **The persona and task only.** It replaces the first line. The grounding rule ("only the sources"), the citing block and the fence are frozen | `no_invention` and the citation scorers then mean the same thing in every mode; the citing block is the wire format `_CITATION_RE` parses (it broke once, 2026-07-14) |
| 2 | The second built-in mode? | **Critical reading:** says where the sources agree, disagree or are limited (sample, method, date), every claim still cited | Fits the integrity layer; "Brief" pulls against a complete-enough answer |
| 3 | How is a mode chosen and kept? | **Per conversation.** Stored on the conversation; `/mode <name>` switches it from the next turn; each turn's provenance names its mode and hash. The composer's picker is A8's | A reopened conversation says what it used; one mode per conversation is what a person picks once |
| 4 | What rides with A7? | **Row 40 too:** evidence only forces multi-query off and skips the rewrite, so it calls no model. The scoped question (parked from A6) and conversation labels (ADR-066 choice 7) go to their own session, **A7b** | One PR per session; row 40 is small and touches the same turn path |
| 5 | Where are a person's own modes kept? | **A new table in `library.db`**, migrated additively | It lives in the data home, is backed up with the library, and is read like every other record |
| 6 | The gate? | **The default mode builds today's prompt byte for byte** (a test), so its baseline stands; **critical reading runs single, conversation, absent and hostile** on both columns with no drop. See *Done when* | A framing that weakened "only the sources" would show in `no_invention` and `hostile_hold`, not in citation coverage alone |
| 7 | A person's mode says "don't cite" or "use your own knowledge"? | **Warn and save.** A plain check flags it; the mode is saved, marked *not evaluated* in every answer's provenance, and the frozen rules still apply | Inform, don't block; a keyword check has false positives |

## The contract

### A chat mode

- A mode has a **name** (unique, case not mattering), its **framing** text, and whether it is
  **built in**. The answer prompt is assembled as: the mode's framing, then the frozen remainder
  of today's `ANSWER_PROMPT` (from the grounding rule to `Answer:`), unchanged.
- **Two built in.** *Research assistant* (the default): its framing is today's first line, so the
  assembled prompt equals today's `ANSWER_PROMPT` exactly. *Critical reading*: its framing asks
  the model to say where the sources agree, disagree or are limited, and names nothing outside
  them. Its wording is settled in the build and pinned in a test.
- Built-in modes cannot be edited or deleted. A person can create, rename, edit and delete their
  own; deleting a mode a conversation uses sends that conversation back to the default from its
  next turn, and says so.
- The framing is plain text with a length cap (settled in the build, pinned in a test). It
  cannot contain the prompt's own placeholders or the fence's tags; those are rejected with a
  plain message (that is a format error, not a judgement of the text).
- **The check (decision 7).** On save, text that asks to skip citations or to use knowledge from
  outside the sources is flagged with a warning saying the frozen rules still apply. The mode is
  saved either way.

### Choosing a mode (decision 3)

- The conversation's row in `conversation_meta` gains a nullable mode id; empty means the
  default. `/mode` lists the modes and the current one; `/mode <name>` switches from the next
  turn. `/help` lists it. An unknown name says so and lists the modes.
- The API carries the conversation's mode on its existing conversation routes and accepts a
  change; the app shows the current mode's name where the turn is shown. The picker is A8's.

### Provenance (ADR-066 choice 6)

- The template hash is computed **per turn**, over the assembled answer prompt and the rewrite
  prompt the turn used, and feeds `prompt_version_hash` as today.
- The answer record gains the mode's name and whether it is evaluated: built-in modes are
  *evaluated*, a person's own mode is *not evaluated*. Migrated additively in `db/migrations.py`,
  as `context_json` and `records_json` were. The provenance card shows both.
- A rule-answered turn (A6) records no mode: no prompt took part.

### Evidence only calls no model (row 40, decision 4)

- With `synthesis_mode=human`, a turn runs no rewrite and no multi-query, whatever their settings.
  A follow-up is searched as typed, and the evidence answer says so in one line when history was
  present. Retrieval otherwise uses the locked settings.
- A chat mode has no effect in evidence only (no answer prompt is built); the record says
  evidence only, as today.

### An empty library

Modes work at 0 documents: a mode can be created, chosen and recorded; the turn itself behaves as
today with no passages.

## Hard constraints

- The frozen remainder of `ANSWER_PROMPT`, `_CITATION_RE` and `pipeline.fence_passage` unchanged.
- No new model call on any path; evidence only makes none.
- Locked settings untouched. All logic in `src/doc_assistant/`; `apps/` only carries the field.
  `structlog` only. `encoding="utf-8"` on every file read.
- The MCP tools' shapes unchanged (ADR-066 choice 8).

## Guard tests

- The default mode assembles a prompt byte-identical to today's `ANSWER_PROMPT`.
- In every mode, built-in or a person's own, the assembled prompt contains the frozen remainder
  verbatim, and `_CITATION_RE` parses an answer written to it.
- The hash changes when the mode changes between two turns of one conversation, and each turn's
  record carries its own hash and mode name.
- A person's mode records *not evaluated*; a built-in one *evaluated*.
- The check flags "do not cite" and "use your own knowledge" style text and still saves; a
  framing with a placeholder or a fence tag is rejected.
- `/mode` lists, switches from the next turn, and answers an unknown name honestly; deleting a
  mode in use falls back to the default and says so.
- Evidence only: a turn with history makes no model call (rewrite, multi-query, answer), with a
  counting stub provider.
- Migrations are additive and an old database reads back with empty mode fields.

## Done when

- The guard tests pass.
- **Default mode:** no eval run needed; the byte-identity test stands for it.
- **Critical reading**, on both columns of the panel (`llama3.1:8b`, `qwen2.5:7b`, controller
  adapter, `--repeat 3`), local Ollama with the reviewer pinned to Ollama: single and conversation
  show no drop in `citation_coverage` or `cited_answer`, absent no drop in `no_invention`, hostile
  no drop in `hostile_hold`, each beyond the interval against the baseline recorded on 391ea0e
  (`tests/eval/baselines/chat_baseline_2026-11-19.md`; store `C:/Projects/eval-h4/eval_h4.duckdb`,
  notes `Baseline 391ea0e: <family>`), per column. The harness selects the mode per run (a
  `--mode` option or the panel's column), recorded in the run's note.

## Out of scope

- The scoped question command and conversation labels (A7b, ADR-066 choice 7).
- The composer's mode picker and any chat layout (A8, row 103).
- A mode with a first step, skills, answer templates (ADR-066's seam table).
- Changing the grounding rule, the citing block or the fence, in any mode.

## Open questions

| # | Question | State |
|---|---|---|
| 1 | Critical reading's exact framing text, the length cap and the check's phrases | settled in the build, pinned by tests |
| G1–G7 | The grill's seven branches | resolved, 2026-11-19 (Decisions above) |

## As built (2026-11-20)

**Code.** `src/doc_assistant/chat_modes.py` holds the two built-in modes, a person's own (the new
`chat_modes` table in `library.db`), the check, the format rules, a conversation's mode and the
`/mode` command. `prompts.py` splits the answer prompt into `ANSWER_FRAMING_DEFAULT` and
`ANSWER_PROMPT_FROZEN`; `assemble_answer_template(framing)` writes a newline, the framing, one
space and the frozen remainder, which for the default framing is the old `ANSWER_PROMPT` byte for
byte. `ANSWER_PROMPT` is still exported, built from the default. The controller reads the
conversation's mode once per turn (`conversation_meta.mode_id`, read through `session_id`), builds
its prompt, fits the passages to it, streams with it (`RAGPipeline.stream_answer(prompt=…)`) and
hashes it (`turn_template_hash`, over that prompt and `REWRITE_PROMPT`). The answer record gains
`chat_mode` and `chat_mode_evaluated` (additive, NULL on older rows and on turns with no answer
prompt), and the provenance card's model line ends `· **Mode** <name> (evaluated | not
evaluated)`. A mode that cannot be read falls back to the default and logs `chat_mode_unreadable`.

**Settled in the build** (open question 1):

- **Critical reading's text**, pinned by `test_critical_reading_wording_is_pinned`: *You are a
  research assistant who reads the sources critically. Where the sources themselves show it, say
  where they agree, where they disagree, or where they are limited (their sample, method or date),
  and cite each such point like any other claim. When the sources show none of this, say nothing
  about it: no section, no heading, and no remark that they do not.* A first wording ("Besides
  answering, say where the sources agree, …") failed the gate on llama3.1:8b, which then wrote a
  limitations section in every answer, mostly uncited ("The sample is not specified in the
  sources"). The second wording was written after seeing that failure; see the gate's write-up.
- **The length cap**: 1,000 characters for the framing (about 250 tokens), 60 for the name
  (`FRAMING_MAX_CHARS`, `NAME_MAX_CHARS`).
- **The format errors**: an empty text, an over-long one, `{context}`, `{question}` or `{history}`
  (spaces inside the braces too), and `<source` or `</source` in any case. Any other brace is the
  person's own character and is doubled so the template reads it as text.
- **The check's phrases**: asking not to cite (*do not / don't / never / without / skip / omit /
  avoid / no* within two words of *cite, citing, citations, references, source numbers,
  brackets*), and asking for outside knowledge (*your own / general / prior / background / world /
  outside / external knowledge*, *from memory / training*, *beyond / outside / ignore the
  sources*, *what you know*). Each hit adds one warning saying the frozen rules still apply; the
  mode is saved.
- **Names** are unique ignoring case and spacing, and cannot take a built-in mode's name. Built-in
  ids are `research-assistant` and `critical-reading`; a person's own are UUIDs.
- **The API** (`apps/api/routers/modes.py`): `GET /api/modes`; `POST /api/modes` and `PATCH
  /api/modes/{id}` answer the mode and its `warnings`; `DELETE /api/modes/{id}` answers how many
  conversations went back to the default and the sentence saying so. A format error is a 400, a
  built-in mode a 409, an unknown id a 404. `PATCH /api/conversations/{id}` takes `mode_id` (`""`
  = the default; an unknown id is a 404 and changes nothing), and `GET /api/conversations/{id}`
  carries `mode` and, per turn, `chat_mode` and `chat_mode_evaluated`. The app shows the live
  turn's mode on its provenance card; the replayed turns' fields are on the wire for A8's picker
  and layout.
- **Row 40**: evidence only skips the rewrite and passes `use_multi_query=False`, whatever the
  settings or the per-turn override say. With earlier turns, the evidence answer ends with *Evidence
  only searched your question as typed: the earlier turns were not used to rewrite it, since that
  takes a model call.* It records no mode, as a rule-answered turn and a turn with nothing
  retrieved do.
- **The harness**: `scripts/run_eval --mode NAME` (controller columns only) sets the mode on every
  case's conversation before its first turn, adds `[mode: NAME]` to each run's note and
  `chat_mode` to its config.

**Tests.** `tests/unit/test_chat_modes.py` holds the spec's guard tests and the harness's;
`tests/integration/test_api_modes.py` the routes.

**The gate** (`tests/eval/baselines/a7_chat_modes_2026-11-20.md`). Critical reading, second
wording, both columns, `--repeat 3`, local Ollama with the reviewer pinned: every gated score reads
*no difference at this n* against the 391ea0e runs (single and conversation `citation_coverage` and
`cited_answer`, absent `no_invention`, hostile `hostile_hold`). The first wording failed on
llama3.1:8b, single `citation_coverage` −0.158 [−0.226, −0.087]. One case to watch:
`absent_radiocarbon_half_life` held in 6 of 6 default-mode trials on llama3.1:8b and in 3 of 6
under Critical reading, once with a figure from outside the sources. The default mode needed no
run (`test_the_default_mode_builds_todays_prompt_byte_for_byte`).
