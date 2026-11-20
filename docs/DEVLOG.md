<!-- status: active · updated: 2026-11-19 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps at most 20 entries** (`devlog_max_entries = 20` in `scripts/conventions.toml`,
> cpc rule 13b, user standard 2026-09-10; `tests/unit/test_doc_sizes.py` pins the same number).
> Past 20, one batch cuts it to the newest 10 (`devlog_rotate_to`, cpc 1.12.0, since 2026-09-29).
> Rotate with `python tools/conventions/rungate.py rotate --root . --file devlog --write` (the shim —
> calling `tools/conventions/cpc/rotate.py` directly fails under `.venv`; corrected 2026-09-16) — it moves the
> oldest entries **verbatim** into the highest-numbered archive, or starts the next one when that is
> past `archive_max_tokens`, and verifies the bytes; then update the range below by hand (cpc ticket
> T-003), and check a new archive's title (cpc 1.12.0 titles every new one "archive 001"). A day may
> be split across two files at the cut. Every archived heading is listed in
> [`docs/archive/DEVLOG-INDEX.md`](archive/DEVLOG-INDEX.md). Older entries, newest-first, unedited:
> **2026-11-05 (3) → 2026-11-16 (4)** in [`docs/archive/DEVLOG-archive-010.md`](archive/DEVLOG-archive-010.md)
> (rotated 2026-11-09: 11 entries, 2026-11-12: 11 more, 2026-11-16: 11 more, and 2026-11-20: 11 more) ·
> **2026-10-24 → 2026-11-05 (2)** in [`docs/archive/DEVLOG-archive-009.md`](archive/DEVLOG-archive-009.md)
> (rotated 2026-10-31: 11 entries, 2026-11-02: 11 more, 2026-11-05: 11 more, and 2026-11-06: 11 more) ·
> **2026-09-30 (3) → 2026-10-22** in [`docs/archive/DEVLOG-archive-008.md`](archive/DEVLOG-archive-008.md)
> (rotated 2026-10-17: 11 entries, 2026-10-21: 11 more, and 2026-10-27: 11 more) ·
> **2026-09-07 → 2026-09-30 (2)** in [`docs/archive/DEVLOG-archive-007.md`](archive/DEVLOG-archive-007.md)
> (rotated 2026-09-30, the first batch: 11 entries, and 2026-10-09: 11 more) ·
> **2026-08-12 (1) → 2026-09-04 (2)** (the first 2026-09-02 and 2026-09-04 entries are unnumbered) in [`docs/archive/DEVLOG-archive-006.md`](archive/DEVLOG-archive-006.md)
> (rotated 2026-09-04, 2026-09-10, four times on 2026-09-16, on 2026-09-17, 2026-09-18, 2026-09-20, twice on 2026-09-21, twice on 2026-09-22 and on 2026-09-29) ·
> **2026-08-08 (1) → 2026-08-11 (4)** in [`docs/archive/DEVLOG-archive-005.md`](archive/DEVLOG-archive-005.md)
> (rotated 2026-08-30) ·
> **2026-08-05 → 2026-08-07** in [`docs/archive/DEVLOG-archive-004.md`](archive/DEVLOG-archive-004.md)
> (rotated 2026-08-28) · **2026-08-02 → 2026-08-03** in
> [`docs/archive/DEVLOG-archive-003.md`](archive/DEVLOG-archive-003.md) (rotated 2026-08-26 — this
> row was missing until 2026-08-28) · **2026-07-15 → 2026-08-01** in
> [`docs/archive/DEVLOG-archive-002.md`](archive/DEVLOG-archive-002.md)
> (rotated 2026-08-15) · **2026-05-21 → 2026-07-14** in
> [`docs/archive/DEVLOG-archive-001.md`](archive/DEVLOG-archive-001.md) (rotated 2026-07-21).
>
> **The 4,000-line cap in `tests/unit/test_doc_sizes.py` stays as the backstop** (an entry cap cannot
> see an entry that is itself an ADR in disguise). When either trips, rotate — **do not raise
> the cap.** The cap exists because this log reached 8,244 lines before anyone noticed: every entry
> is individually small and correct, so unbounded growth is invisible per commit.

---

## 2026-11-20 — Chat modes, and evidence only calls no model (A7)

**What changed.** `src/doc_assistant/chat_modes.py`: a chat mode is a name and the framing that
replaces the answer prompt's first line; `prompts.ANSWER_PROMPT_FROZEN` (the grounding rule, the
citing block, the fence) follows it in every mode. *Research assistant* builds the old prompt byte
for byte; *Critical reading* is pinned in a test. A person's own modes live in the new
`chat_modes` table; on save, text asking not to cite or to use outside knowledge earns a warning
and is saved, and placeholders or fence tags are refused. A conversation keeps its mode on
`conversation_meta.mode_id`; `/mode` lists and switches it from the next turn, `/help` lists it,
and the conversation routes carry it (`/api/modes` for a person's own). The controller hashes the
prompt per turn; `AnswerRecord.chat_mode` and `chat_mode_evaluated` (additive) name the mode, and
the provenance card shows both. A rule-answered turn records no mode. Row 40: evidence only runs
no rewrite and no multi-query, so it makes no model call, and says so when it had history.
`run_eval --mode` selects the mode per run and records it in the note. Detail: the spec's As
built.

**Why.** ROADMAP 37 and 40, ADR-066 choice 6, `docs/specs/chat-modes.md` (accepted 2026-11-19).

**Measured.** `tests/eval/baselines/a7_chat_modes_2026-11-20.md`: Critical reading on single,
conversation, absent and hostile, both columns, `--repeat 3`, local Ollama: every gated score
reads no difference at this n against the 391ea0e runs. The first wording failed: llama3.1:8b wrote
an uncited limitations section into most answers, and single's citation coverage dropped,
−0.158 [−0.226, −0.087]. The second wording, chosen after seeing that, removed it (39 of 54 single answers
to 0). To watch: `absent_radiocarbon_half_life` held 6 of 6 in the default mode on llama3.1:8b,
3 of 6 under Critical reading. $0; 3,888 s, 580 s and 3,419 s. Full suite: 3,568 passed, 3
skipped (before the second wording).

**Rejected.** Hashing once with the mode name beside it (an edited mode would report the old
hash); the mode on the in-memory session (a reopened conversation would forget it); shipping the
first wording with its drop noted.

**What it opens.** A8's composer picker over `/api/modes` and the conversation's `mode`; the
radiocarbon case under Critical reading on a wider absent family; A7b (the scoped question and
conversation labels).

## 2026-11-19 (5) — The chat-modes spec, accepted (A7)

**What changed.** `docs/specs/chat-modes.md`, accepted by the user as written: a chat mode is a name and a persona-and-task
text that replaces the answer prompt's first line; the grounding rule, the citing block and the
fence stay frozen. Two built in (*Research assistant*, today's prompt byte for byte, and *Critical
reading*); a person's own are kept in a new `library.db` table, flagged when their text fights the
frozen rules and recorded *not evaluated*. A mode is chosen per conversation (`/mode`), and the
template is hashed per turn. Row 40 rides here: evidence only runs no rewrite and no multi-query.
ROADMAP rows 37 and 40 point at it; the plan gains A7b.

**Why.** Rows 37 and 40 and ADR-066 choice 6. The user answered the scoped grill's seven questions
on 2026-11-19, all as recommended.

**Rejected.** A mode that may loosen "only the sources" (`no_invention` would stop meaning one
thing); a global or per-turn mode; a modes file; refusing to save a flagged mode; bundling the
scoped question and labels (moved to A7b, one PR per session).

**What it opens.** The build and the gate: critical reading on single,
conversation, absent and hostile against the 391ea0e baseline, per column.

## 2026-11-19 (4) — Record questions answered from the records, by rule and command (A6)

**What changed.** `src/doc_assistant/record_lookups.py`: fixed rules map a question about the
library itself to one read-surface lookup, or to `library_summary` (`query_router`): counts and
health, folders, one document's metadata, missing fields, a curated concept. A named document is found by its recorded title, else its
filename stem, every content word matching. One match answers, several are listed, none goes to
retrieval. The controller now writes an answer record with no passages and no model, the new nullable
`AnswerRecord.records_json` (migrated additively: lookup, args, ids, rows, via), route `library`.
The answer ends with *From the library's records: no passage was read and no model was asked.*
New commands `/folder`, `/concept`, and `/document` by title call the same lookups and write the
same record. 85 guard tests in `tests/unit/test_record_lookups.py`; the controller tests' seam is
now `answer_record_question`. Detail: the spec's As built.

**Why.** ROADMAP 102, ADR-066 choice 5, `docs/specs/chat-record-lookups.md` (accepted
2026-11-19). On 391ea0e the records family routed 3 of 8 cases to the library, and KI-95 answered
a folder count with the library's count.

**Measured.** `tests/eval/baselines/a6_record_lookups_2026-11-19.md`: records 8 of 8 routed with
`contains_all` 1.000 on llama3.1:8b and qwen2.5:7b, every trial (391ea0e: 3 of 8; 0.500, 0.625).
Absent, single and conversation: no drop beyond the interval on either column. Two binary cases
flipped once in three trials, each matched by one flipping back; 0 of 264 content case-runs
reached a rule. $0, 2,521 s. KI-95 closed. Full suite: 3,517 passed, 3 skipped.

**Rejected.** Sending `/document <id prefix>` through the new lookup: it would drop today's
details view (health, chunks, ingestion history), and no rule takes an id. A concept rule on
"what is the definition of X" without the word "concept": it would turn content questions into
lookups.

**What it opens.** The scoped question command (A7). A concept fixture for the eval data home, so
the concept lookups are measured and not only unit-tested. A title match reads every document a
page at a time: about 100 small reads at 10,000 documents, not yet timed.

## 2026-11-19 (3) — cpc's deny rules laid for the agent's git commands

**What changed.** `cpc-init --root . --add deny-rules` created `.claude/settings.json` with cpc's 22
deny entries (cpc ADR-080): Claude Code now refuses the agent's `git commit`, `push`, `merge` and
`tag` (and the `git -C` forms), `gh pr merge`, `gh release create`, and any command carrying the
push marker, in Bash and PowerShell. The file is local-only like the rest of `.claude/` (ADR-029),
so it is not in the commit.

**Why.** The owner runs every commit, push, merge and tag in this repo already; the rule is now
enforced by the tool instead of by memory.

**Rejected.** Force-adding `.claude/settings.json` past the `.claude/` ignore: it would reverse
ADR-029 for one file, which is the owner's call.

**What it opens.** Three lines contradicted the rules; the owner chose to reword them and keep the
entries, and they now say the maintainer runs the command: `AGENTS.md` (Non-negotiables, first
bullet), `docs/RELEASE.md` §6, §7 and §7b (the runbook's `git tag`, `git push` and
`gh release create`), and the project memory `release-publish-needs-user-go`. The rules apply on
this machine only, since `settings.json` stays local.

## 2026-11-19 (2) — The chat-eval baseline re-recorded on 391ea0e

**What changed.** `tests/eval/baselines/chat_baseline_2026-11-19.md`: the five chat families on
llama3.1:8b and qwen2.5:7b (controller adapter, `--repeat 3`, the 28 public papers and the hostile
home, reviewer forced local, $0, 3,213 s), with the paired comparison against H4.

**Why.** H4 predates A2's fence, and A2 was gated on hostile and single only, so A6 and A7 gated
against H4 would confound A2 with their own change.

**Measured.** Against H4: citation coverage on single higher at this n on both models (+0.068,
+0.086); conversation on llama a measured drop (−0.160, interval [−0.343, −0.002]), as A5's run
saw and its control showed is not A5's; every other continuous score no difference at this n.
Records still routes 3 of 8. The language-switch attack now breaks in every trial on both models.

**Rejected.** Repeating H4's `/v1` column: it measured KI-94, not a model.

**What it opens.** A6's gate (`docs/specs/chat-record-lookups.md` § Done when) reads these runs.
The conversation drop has no isolated cause.

## 2026-11-19 (1) — The record-lookups spec, accepted (A6)

**What changed.** `docs/specs/chat-record-lookups.md`, accepted by the user as written: record questions (counts, health,
folders, one document's metadata, missing fields, a curated concept) answered by rule or by
`/folder`, `/concept` and `/document` by title, through the read surface's own lookups and with no
model call. A named document is found by its recorded title, else its filename, with no score:
one match answers, several are listed, none goes to retrieval. A rule-answered turn writes an
answer record with a new nullable `records_json` and says it came from the library's records.
ROADMAP row 102 points at it.

**Why.** Row 102 and ADR-066 choice 5. The user answered the scoped grill's five questions on
2026-11-19, all as recommended.

**Rejected.** Gaps as a lookup (signals of uneven trust); finding a document by search (a passage
citing a paper would report the wrong document's year); the scoped question command, parked to
A7. The plan's "a local model that cannot do it says so" is set aside: no model takes part.

**What it opens.** The build and the gate run. Open question 1 is resolved as recommended: the
lookup commands write the same answer record as a rule.

## 2026-11-17 (4) — Each turn fitted to the model's window, and said (A5)

**What changed.** `context.py`: one estimate (moved from the wiki), the window each call is fitted
to (`set` for Ollama, a server's own, Anthropic's documented 200,000, else the floor and said), the
fit in the spec's order (oldest history, then the newest earlier answer shortened, then the
lowest-ranked passages; the best passage clipped inside its fence, never dropped), a record in the
new `AnswerRecord.context_json`, and a **Fitted to the window** line on the answer when anything
gave way. A dropped passage leaves the sources too. `prompt_version` hashes `REWRITE_PROMPT` with
`ANSWER_PROMPT`. `run_eval` migrates the data home before the controller adapter, and a failed
provenance capture is now logged. 14 guard tests.

**Why.** ROADMAP row 101, spec `docs/specs/context-handling.md` (accepted 2026-11-17): a prompt past
the window was cut by the server, silently.

**Measured.** Against H4, conversation on llama read as a citation-coverage drop; a same-day control
on the commit before A5 showed no difference on either model (coverage −0.030 llama, −0.108 qwen,
both intervals across zero), and the run's records show nothing gave way in any of 204 turns: on
the public cases A5 sends what was sent before. The 13-turn case keeps its thread on llama in 3 of
3 on A5 and on the control; where a thread is lost the rewrite drops the subject's name.
Figures in the spec's As built.

**Rejected.** Gating on H4 alone: it predates A2's fence, which was never gated on conversations,
so a drop there could not be told from A2's.

**What it opens.** The rewrite losing the subject on long threads (decision 1's seam: earlier turns
in the answer prompt). The fit is 127 tokens from acting on the public set. KI-94's `/v1` cut stays:
the floor does not reach a server that ignores it.

## 2026-11-17 (3) — The context-handling spec, drafted (A5)

**What changed.** `docs/specs/context-handling.md`, a draft: every model call of a turn is measured
by one shared estimate, fitted to the model's window in a fixed order (oldest history first, then
the newest earlier answer shortened, then the lowest-ranked passages; never the instructions, the
question or the best passage), recorded in a new `AnswerRecord.context_json`, and said on the
answer when anything gave way. A server with no stated window is fitted to the Ollama floor and
says so. `prompt_version` will hash `REWRITE_PROMPT` too. ROADMAP row 101 points at it.

**Why.** Row 101 was gated on a spec. The user answered its four questions on 2026-11-17 (all four
as recommended). Today a prompt longer than the window is cut by the server, silently.

**Rejected.** Earlier turns in the answer prompt and a running summary: seams for later, once the
conversation family says whether the rewrite carries the thread.

**What it opens.** The user's acceptance, then the build and the A5 gate run.

## 2026-11-17 (2) — The MCP endpoint and its bridge (A4)

**What changed.** `src/doc_assistant/mcp_server.py`: a hand-written MCP handler (`initialize`,
`tools/list`, `tools/call`, `ping`) over the nine read tools, with each tool's description, and a
bounded in-memory call log. `apps/api/routers/mcp.py`: `POST /mcp` with its own bearer token, off
until switched on, refusing a web page's `Origin`; `/api/mcp*` for the window (status, switches,
setup, token replacement). `apps/api/mcp_bridge.py`: `--mcp-bridge` forwards a client's stdio to
`/mcp`. The token lives in `credentials.json`; the switches in `settings.json`. *Settings › AI
programs* (`AiPrograms.svelte`) turns it on, shows the setup for both kinds of client, replaces the
token, offers past conversations, and lists recent calls. `/mcp` is exempt from the launch token.
Tests: `tests/integration/test_api_mcp.py` (9), `lib/settings/mcp.test.ts`.

**Why.** ADR-066's build order: the endpoint after the read surface. The user chose a hand-written
protocol and the bridge in this session (2026-11-17). Claude Code, run headless with a temporary
MCP config on a copy of the public papers, drove both transports.

**Rejected.** The MCP SDK: a dependency tree for three requests. Returning the token on a GET: a
POST needs the window's launch token. A stream on `GET /mcp`: nothing is ever pushed.

**What it opens.** The bridge from an installed build is unverified (driven from a checkout). The
release's walkthrough should connect one client to the installed app.

## 2026-11-17 — The read surface after a stand-in client trial: nine tools (A3)

**What changed.** A Sonnet subagent played an outside AI program on the public papers, with nine
tasks, before and after the changes (`tests/eval/baselines/a3_surface_trial_2026-11-17.md`). New
tools: `get_passage` and `list_concepts`; `list_conversations` gains a `text` filter. A passage is
sent once (`Passage.text` removed: `fenced` holds the same text), with its page derived from the
document's page markers (`library.source_view.page_from_meta`, now public). A conversation's cited
excerpt is placed on its page by finding it in the document's text. `SURFACE_VERSION` stays 1: no
client has seen the shapes. ADR-066 gains an amendment; T6 counts nine tools.

**Why.** The first run found what the tests could not: no page anywhere, no reading around a
passage, no way to list concepts, no way to find a conversation by its question. The user chose
all four changes after confirming none had been left out for security. Run 2: 6 of 9 tasks done
(4 before), 24 calls (30 before).

**Rejected.** A score floor for "nothing relevant": a number tuned to this corpus until measured.
Storing a page in answer records now: a provenance change of its own; the excerpt places 471 of
480 GPipe citations on the trial copy.

**What it opens.** A4 serves these shapes and freezes them. Author metadata stays extraction's
(KI-54, KI-89); an author filter is additive.
