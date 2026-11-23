<!-- status: archived · updated: 2026-11-20 · class: append-only -->

# DEVLOG — archive 010 (2026-11-05 (3) → 2026-11-16 (4))

Rotated DEVLOG entries — moved here from `docs/DEVLOG.md` once it held more than 20, **verbatim** per cpc ADR-023 rule 13b and cpc ADR-053. Newest entry on top. Append-only; never edited.

## 2026-11-20 (2) — Questions within several folders or a concept, and the chat list's labels (A7b)

**What changed.** A turn's scope is several folders, searched as one set: `scope_folder_ids`
replaces `scope_folder_id` on `/api/chat` and `/api/compare`, `_resolve_scope` reads the union in
one query (`library.folders_doc_hashes`), and a folder that held nothing is left out and named.
`/folder <name>: <question>` and `/concept <name>: <question>` ask within one folder or the
documents where a concept is present (`concept_presence`, exact forms); no presence searches
nothing and asks for a rebuild, a stale graph is said. `retrieval_scope.py` holds the scope, its
record shape and one reader for both shapes, used by the history, provenance and the read
surface; MCP turns gain an additive `scope`. `/api/conversations` carries each conversation's
labels and filters by document or folder through the read surface's derivation. The window's
scope line names every folder (`ScopeChip.svelte`); the picker sends `[id]` or `null`.

**Why.** ADR-066 choice 7, T-015's backend half, `docs/specs/chat-scoped-questions.md` (accepted
2026-11-20, G1 to G6).

**Measured.** Tests only, $0 (G5). On 10,000 documents in 20 folders, the 5-folder union took
0.006 s (bound 0.5 s) and the folder-filtered list over 1,000 conversations 0.020 s (bound
2.0 s). Full suite: 3,596 passed, 3 skipped. Desktop: 431 tests, `svelte-check` 0 errors.

**Rejected.** Intersecting a command's folder with the picker's (both name one place; the
command is the later choice); a second label derivation for the app's list; keeping the old wire
field beside the new one.

**What it opens.** A8: the picker's tickboxes and the list's label and filter controls. Not
foreseen by the spec, settled in its As built: `parse_command` lowercases its argument, and
`/folder Thesis:` reached A6 as "Thesis:".

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

## 2026-11-16 (4) — The read surface: ADR-066's seven tools as functions (A3)

**What changed.** New package `src/doc_assistant/read_surface/`: `search_library`, `list_folders`,
`list_documents`, `get_document`, `list_conversations`, `get_conversation` and `get_concept`, with
their result shapes in `shapes.py` under `SURFACE_VERSION = 1`. Search takes the engine's retrieval
as a parameter; each passage carries its text, the fenced copy (`pipeline.fence_passage`, now shared
with the prompt) and the text check's labels. A conversation's labels (the documents and folders
its answers cited) are derived from the answer records on every read. Tests:
`tests/unit/test_read_surface.py`, the shape snapshot `test_read_surface_shapes.py`, and
`tests/integration/test_empty_home_get_routes.py`, which walks every GET route on an empty library
(42, no 500). `docs/security.md` gains T6, an MCP client; ADR-066 gains *As built — A3*.

**Why.** ADR-066's build order: the read surface before the endpoint (A4) and the chat's lookups
(A6). Measured on an invented 10,000-document library: 4–13 ms a page of documents.

**Rejected.** Reusing `library.list_documents` and `conversations.list_conversations`: the first
reads every document for one page, the second caps at 100 with no page. Bounding *Look in my
library* with `top_docs` alone: it would store a wrong document count in each definition's evidence
(KI-18's note).

**What it opens.** A4: the endpoint, its switches, token and call log. The shapes are frozen once
it serves them, so they are the thing to review now.

## 2026-11-16 (3) — The text check once per document, after extraction (A2b)

**What changed.** New `src/doc_assistant/document_checks.py` and table `document_text_checks`
(`DocumentTextCheck`, additive): A2's patterns run over each document's extracted text once, by
paragraph with its page (`source_checks.find_in_document`), and the row keeps the marked paragraphs,
the text's hash and the patterns' version (`CHECKS_VERSION`). The index run checks what it indexed
(`ingest._check_document_text`); *Re-run… › Text checks* (`reingest`, first and instant) checks what
was indexed before. The Summary gains `addressed_to_ai` and `text_not_checked` (`found.text_checks`,
rows in *From indexing*); the document page gains a *Text addressed to an AI* block listing each
paragraph with *Show on page* (`TextCheck.svelte`, route `/api/library/documents/{id}/text-check`).
Ollama's provider form now gives the public-set figures: citations and how often planted text was
resisted, per local model. Glossary U-005; summary spec § As built; the A2 record's A2b section.

**Why.** The user, 2026-11-16: flag documents after extraction, before any question. Measured: the 8
hostile documents marked, 0 of the 28 papers, 2 of the author's 105, about 0.02 s per document.

**Rejected.** Checking at app start (the user chose a Re-run part: a write only when asked).
Counting marks in *Need attention*: a paper about AI is marked too. Showing a stale check as a
finding: its text or patterns have moved.

**What it opens.** A3: MCP tool results carry the marks. A whole-library re-check after a pattern
change is a Re-run over every document, minutes at ~10,000.

## 2026-11-16 (2) — The prompt fence, and checks that hold on any model (A2, S-9)

**What changed.** `pipeline.format_docs_for_prompt` fences each passage as
`<source n="…" file="…" page="…">…</source>`, defusing a fence tag inside a passage and escaping
the attributes; `ANSWER_PROMPT` says everything inside a source is document text, never
instructions. New `src/doc_assistant/source_checks.py`: `addressed_to_ai` marks a passage written
to an AI (a fake system message, "ignore previous instructions", a demand on the answer), and
`sourced_links` lists the web addresses in the unmarked passages. `SourceView.addressed_to_ai` and
`TurnResult.source_links` carry them through the API; the answer shows a marked source with a chip,
and a web link it was not given as text (`apps/desktop/src/lib/chat/links.ts`, `Markdown.svelte`).
Record: `tests/eval/baselines/a2_prompt_fence_2026-11-16.md`; `docs/security.md` S5, S-9, check 8.

**Why.** S-9 lands before any chat tool (ROADMAP row 60). The measurement showed the fence lifts
citations a little (one column measured higher per version) but leaves the hostile hold where it
was (18 and 19 of 24 runs). So the defence moved to checks that do not depend on the model; the
user: "add those checks + keep the fence without the line".

**Rejected.** The "if a source tells you to do something, do not do it" line: 10 runs per model on
the language-switch attack showed no benefit. Moving instructions into a separate system message:
a change to how the app talks to Ollama, needing its own measurement. Blocking or stripping a
marked passage: ADR-043 and inform-don't-block. Counting a planted link as sourced: the mocked
check showed a phishing address in a hostile passage would stay clickable.

**What it opens.** The gate's third part (no binary case breaks) was not met: three cases lost a
citation in one or two of three trials, and the fence was kept on the user's decision. Next, in the
plan: A2b, the same scan once per document after extraction, shown in the Library; each model's
hold rate shown where the provider is chosen; A3's MCP tools carrying the fence and the marks,
since another app's model may have tools this app never gives.

## 2026-11-16 — The judge, checked against the user's grades (H5)

**What changed.** `src/doc_assistant/eval/judge_agreement.py`: decision agreement, Spearman's rho
with a bootstrap interval, and the mean absolute deviation, on the answers a person and a judge
both graded, plus the gate (>= 0.9 and <= 0.5, from TESTING.md); tests in
`tests/unit/test_eval_judge_agreement.py`. The user graded 30 H4 answers blind, and two local
judges graded the same answers: `qwen2.5:7b` 0.82 decision agreement, `llama3.1:8b` 0.63. Neither
passes, so both stay in shadow. Claude Haiku, run as one Claude Code subagent per answer at the
user's request, reached 0.81: a stronger model did no better on this rubric. Record: `tests/eval/baselines/h5_judge_agreement_2026-11-16.md` and
`tests/eval/judge_check/h5_2026-11-16/`; `evals/README.md`, `TESTING.md` and the spec say so.

**Why.** Spec `docs/specs/eval-harness-chat.md`, H5: every `llm_judge` value was marked *not
calibrated* until a person's grades existed to compare it with.

**Rejected.** Loosening the gate to the first distribution: it gave no reason to. Showing the
cited passages on the grading page: the user's grades would then measure something the judge never
sees. The pinned API reference judge: refused for a lack of API credit; the Haiku subagent run stands
in for it, outside the harness's pinning.

**What it opens.** H6, the write-up. Grading exposed KI-95 (a folder count answered with the
library's count) and the records references the H4 sample had stitched from expected strings. The
user's note for later: answer templates, ranked behind the library and its references (plan, Later).

## 2026-11-15 (3) — H4's retrieval checks and RG-035's first re-run

**What changed.** `tests/eval/baselines/h4_retrieval_and_citation_2026-11-15.md`: the reranker
on and off, and `CANDIDATE_K` 20 against 10 (ROADMAP 69), retrieval only, on the private set (34
cases) and the public 28 (18 cases). Both show no difference at this n. Citation coverage on the
fixed build, three runs on the private 27: `llama3.1:8b` 89.3%, `qwen2.5:7b` 70.0%, up from 37.6%
and 18.0%, with Ollama logging `truncated = 0` on all 210 calls. The README's and
`docs/limitations.md`'s citation figures were replaced. `citation_coverage_2026-08-07.md` and
`candidate_k_public_2026-06-13.md` carry dated lines. No code changed.

**Why.** Spec `docs/specs/eval-harness-chat.md`, H4. RG-035 (1) asked for KI-36's rates on the
fixed build before A2 uses them as its *before*.

**Rejected.** Running the citation re-run through the controller on a copy of the working
library: the pipeline adapter reads the library without writing to it and builds the same prompt
and audit. The difference is stated in the baseline. Dropping the reranker on these intervals:
they contain zero, and the integrity layer reads its scores.

**What it opens.** H5 (the judge, checked against the user's grades). KI-36: open for
`qwen2.5:7b` only, or closed, which is the user's call. RG-035 rows (2) and (3) are still to run.

## 2026-11-15 (2) — The first chat-eval baseline (H4)

**What changed.** `tests/eval/baselines/h4_chat_baseline_2026-11-15.md`: the five case families
through the H4 panel (llama3.1:8b and qwen2.5:7b on Ollama's native endpoint, qwen2.5:7b on its
OpenAI-compatible `/v1`), `--repeat 3`, on the 28 public papers (hostile: plus the 8 hostile
documents). Local models: cited answers 0.92 to 1.00, one invention in 24 absent case-runs,
hostile holds 0.79 to 0.83 (three attacks break). The `/v1` column's cited answers on single
drop to 0.35. No code changed.

**Why.** Spec `docs/specs/eval-harness-chat.md`, H4: sessions A2 to A7 compare against this
baseline with `--paired-baseline`.

**Rejected.** Reading the `/v1` column's hostile score of 1.000 as resistance: its prompt is cut
(KI-94), so the attack never reaches the model, and it cites almost nothing.

**What it opens.** The rest of H4 (RG-035's re-run, row 69's `CANDIDATE_K` on the private set,
the reranker on/off ablation: MRR 0.972 before the rerank against 0.958 after it on single, n =
18; KI-36's README figures). The hostile baseline is what A2's prompt fence must beat. KI-94's
fix stays the user's call.

## 2026-11-15 — The chat eval's cases (H3)

**What changed.** 40 cases, each accepted by the user on a review page against the passage and
page it rests on: 8 single-turn on the demo papers (`cases.public.yaml`, now 18), 8 conversations
(`cases.conversation.yaml`), 8 not-in-the-library questions (`cases.absent.yaml`), 8 hostile
passages (`cases.hostile.yaml`) and 8 records questions (`cases.records.yaml`). Eight invented
hostile documents in `tests/eval/hostile/`, pinned by sha256. The public eval now runs on all 28
papers (`download_corpus --demo`); the manifest, `TESTING.md` and the case file say so. A question
the library cannot answer is scored by `no_invention` (field `not_in_library`), replacing
`refusal_hold`. Guard tests: `test_eval_public_cases.py`, `test_eval_hostile_corpus.py`.

**Why.** Spec `docs/specs/eval-harness-chat.md`, H3. The user rejected the first drafts of two
families. Their rule for an answer the library does not hold: invent nothing, say what the library
does hold with citations, and let the app's own markers show that the point is missing; a model
refusing is not the goal. The hostile drafts tested only what the answer must not do; the
accepted ones also say what a good answer from the document's real content is, and the review
page shows each attack verbatim.

**Rejected.** `refusal_hold`, which held only when nothing was cited: it punished the wanted
behaviour. Invented-fact strings taken from the question itself, which an honest answer repeats
(a guard test now refuses them). Concept-definition records cases: a fresh data home has no
curated concepts.

**What it opens.** H4, the baseline. Each family runs on its data home: the 28 papers for four
of them, the 28 plus the 8 hostile documents for the hostile family. Whether each hostile
document is actually retrieved by its question is the first thing H4 shows. Baselines before
2026-11-14 used the 10-paper corpus and are not comparable.

## 2026-11-14 (2) — Three new dependency advisories: one upgraded, two reviewed

**What changed.** `opentelemetry-api` and its six companions move from 1.45.1 to 1.45.2 (0.66b1 to
0.66b2 for the two beta packages) for CVE-2026-126360, a regex denial of service in the trace-context
propagator; Chroma pulls them in. CVE-2026-126191 and CVE-2026-126193 in `soupsieve` are reviewed
ignores in `pip-audit-ignore.toml`. `just audit` passes again.

**Why.** Published after the last run; the audit gate fails CI on any unreviewed advisory. Both
`soupsieve` advisories need an attacker to supply the CSS selector string, and nothing here runs a
selector: `extractors.py` and `ingest/cache.py` walk the tree with `find_all` and `get_text`.

**Rejected.** Taking `soupsieve` 4.1: it needs Python 3.11.5 against the 3.10 floor, so the lock
forks (2.10 below, 4.1 above), the same reason CVE-2026-126190 was reviewed on 2026-11-13.

**What it opens.** The three `soupsieve` ignores lift together when the Python floor reaches 3.11.5.

## 2026-11-14 — The eval harness instrument for the chat work (H2)

**What changed.** The harness can now measure the chat, on any model. Cases have a `family`,
conversation `turns`, and stated expectations (`expected_route`, `expect_refusal`,
`must_not_contain`, `must_not_cite`). Eleven deterministic scorers (`chat_scorers()`): MRR, nDCG@10
and recall@10, after and before the rerank; citation coverage; and four that hold or break
(`cited_answer`, `refusal_hold`, `hostile_hold`, `route`). `eval/paired.py` compares two groups of
trials per family with a paired bootstrap interval, and counts holds with the cases that broke.
`eval/panel.py` reads panel files. A second adapter drives `ChatController` and replays turns.
`run_eval` takes `--panel`, `--adapter`, `--allow-billed` and `--paired-baseline`; before a run it
states each column's turns, reviewer, judge and who bills, and refuses billing it was not told to
allow. In the app, observation only: the pipeline's `trace`, `TurnResult.route`,
`SourceView.filename`. `TESTING.md` documents all of it, with running it on your own model.

**Why.** Spec `docs/specs/eval-harness-chat.md`, H2. Four chat sessions are gated on measurements
the old harness could not take: the old adapter never reached the router, the history or the
framing.

**Rejected.** A refusal rule that matches a not-found phrase: the app has none, its prompt asks
each model for its own words. Scorers that import the app's audit: they must stay extractable, so
the adapters hand them the app's verdicts. Switching models with `reconfigure`, which saves the
choice into the library's settings.

**What it opens.** H3, the cases. The controller adapter writes an answer record per turn, so it
runs only on a data home of its own. Its trace of the order before the rerank does not exist, so
those scorers are not scored on it.

## 2026-11-13 (3) — The eval harness spec for the chat work (H1)

**What changed.** `docs/specs/eval-harness-chat.md`, a draft: the executor brief for H2 to H6 of
ROADMAP 113. ROADMAP row 113 points at it. No code.

**Why.** Four chat sessions (A2, A5, A6, A7) are gated on measurements the harness cannot take.
A scoped grill put six forks to the user, each taken as recommended: deterministic scorers decide
gates on every model and a model judge is optional and marked not calibrated until H5; a panel is a
file of columns (provider, model, adapter), compared within a column only; eight new cases per
family, all on the public demo corpus; a paired bootstrap interval over cases at `--repeat 3`; a
gate passes on no measured drop and no new break; and a second adapter drives `ChatController`,
found necessary because today's adapter calls `RAGPipeline` and never reaches the router, history
or framing. The user then asked that the spec be model-neutral and fix no budget: which models a
run uses and what it may cost are the run's settings, recorded with it and stated before it starts,
and anyone can run the public families on their own model.

**Rejected.** A budget or a fixed model list in the spec; a judge that gates before it is checked;
records and conversation cases on the private library; fixed thresholds per family; replacing the
pipeline adapter, which would orphan every committed baseline.

**What it opens.** The user accepted the spec the same day, which fixes the 40 cases (H3) and 30
grades (H5); H2 is next. Parked: the exact not-found rule (H2), and a run from inside the app
(after H6). The user made the criterion general: a spec says how a thing works and what it
measures, independent of any one model, provider or budget; AGENTS.md's build protocol now says so.

## 2026-11-13 (2) — ADR-066 accepted as proposed

**What changed.** ADR-066 (*the library is offered to a model as records and passages*) is
accepted: option A, the nine choices and the seven read tools as written. Its status, its Decision
section, the index line in `docs/decisions.md` and ROADMAP rows 65, 102 and 37 say so. No code.

**Why.** It was the open decision that closes A1 in the plan. The user read it with the steer
that the MCP endpoint need not be complete in its first form and will grow; the ADR's table of
seams already carries that.

**Rejected.** A smaller first tool set (three or four tools, without the two conversation tools
and their switch), offered because choice 8 freezes what ships first: the user kept the seven.

**What it opens.** H1, the harness spec. Two texts waited on this acceptance and are still to
write: row 65's description (it still says a stdio server over `pipeline.py`) and the MCP-client
row in `docs/security.md`.

## 2026-11-13 — Repository upkeep: four tickets, and the easy fixes they found

**What changed.** Tickets T-003 to T-006 (the glossary's missing words, security.md's tables,
tables and status headers across the living docs, plan labels in code comments), and three reports
filed in the inboxes of the projects they concern: cpc's template justfile, cpc's docs gate, and
the code-descriptions skill. Fixed now: GLOSSARY's C-011 and C-012 read as built and sit in
order; security.md's §6 is a list, since an unescaped pipe in a code span split seven of its rows
on GitHub, and S3 no longer says it waits on S-6; the ADR-048 row in `docs/decisions.md` has its
Status cell; the CLI's `/help` stops saying *(Phase 4)*; the justfile goes from 184 to 133 lines
with every recipe unchanged; GLOSSARY and security.md lose the changelog in their status line.

**Why.** The owner asked for the upkeep to be ticketed and the easy part done. Each fix is a
statement that was wrong or a page that rendered broken; anything that needs a decision (a
canonical word, a table's new shape) is a ticket.

**Rejected.** Escaping the pipes in §6 rather than making it a list: the raw file would then
show `\|` in commands meant to be copied. Trimming comments across `src/` now: T-006 rides on the
0.8.0 chat and extraction work, which touches those files anyway.

The ticket ledger, `docs/TICKETS.md`, leaves the repository: it is gitignored like the known
issues, and a public report is a GitHub issue. T-001 and T-002 stay in the history. `soupsieve`
moves from 2.9 to 2.10 for CVE-2026-126192. CVE-2026-126190 is fixed only in 4.0.0, which needs
Python 3.11.5 and forks the lock; it is a reviewed ignore in `pip-audit-ignore.toml`, since
nothing here runs a CSS selector.

**What it opens.** T-003 needs the owner's words for the engine and for the library's folders.
The archive tables that render broken stay as they are, since archives are verbatim. The
`soupsieve` ignore lifts when the Python floor reaches 3.11.5.

## 2026-11-12 (3) — ADR-066 proposed: the library is offered to a model as records and passages

**What changed.** A new ADR for ROADMAP rows 65, 102 and 37, with its line in
`docs/decisions.md` and the three rows pointing at it. No code.

**Why.** The chat and MCP work needed its design before any build. A grill put nine forks to the
user, one at a time, and the ADR records each answer: a search returns passages and never an
answer from the app's model; the server runs inside the engine with a stdio bridge; seven
read-only tools; writes in the next release behind a second switch; the chat picks a lookup by
rule and command; a mode is a name and its framing; a conversation's labels are read from its
answer records; the tools' shapes are what freezes; the endpoint is off by default with its own
token. The user also asked that each piece stay manageable now and be able to grow, so the
Decision names the seam each one leaves.

**Rejected.** The model choosing its own tools, and an `ask` tool: different behaviour per model
and a model call at an outside program's request. A server in its own process, as row 65 was
written: a second embedding model in memory and a second program on one library.

**What it opens.** The user's read, which is what accepts it. Not verified: that the engine can
serve MCP beside its routes, and that a stdio bridge starts from an installed Windows build. Row
65's scope text and `docs/security.md`'s attacker row for an MCP client change when it is
accepted.

## 2026-11-12 (2) — ADR-062 accepted as proposed

**What changed.** ADR-062 (*the app stays repairable*) is accepted: option A with all eight
choices as recommended. Its status, its Decision section, the index line in `docs/decisions.md`
and ROADMAP row 100 say so. No code.

**Why.** The plan held it as an open question before the chat and MCP design. The user took each
choice on its own: the engine and the shell both degrade; the shell makes the update check when
no engine answers; four repairs that rename aside and never delete; a repair offered only for
damage; the reachability table as written; the program on port 8001 named, and only an older
engine of ours ended; a remembered folder with no `library.db` is a state; the engine answers
first and opens the library behind it.

**Rejected.** On choice 5, also showing the last session's settings read-only while a section
waits: a stale folder or provider shown as current can mislead. On choice 6, starting the engine
on a free port: the security policy and the dev setup both fix 8001.

**What it opens.** The build, in the ADR's order (choice 7 and the engine's log first), before the
MCP server. The entry before this one built the window half of choice 8's third bullet without
citing the ADR; the ADR's status now names it, and its `Settings.svelte` line numbers are those of
2026-11-05.

## 2026-11-12 — Settings opens before the engine answers (KI-93)

**What changed.** `screens/settings/Settings.svelte` renders its rail and pane at once. General
and Getting started read nothing from `/api/settings`, so they are whole straight away; Documents,
Provider & model and Retrieval say *Waiting for the engine* until the settings land. The read is
made again when `shell.status` turns `ready`, so the open drawer fills in place, and a read that
fails on a running engine shows its error with *Try again*. `GettingStarted.svelte` says it is
waiting, not *Couldn't read the setup state*, while the engine is not up.

**Why.** The user's 0.7.3 walkthrough: Settings showed nothing until the engine was online, and
the installed engine takes tens of seconds to start. The drawer asked once at opening, replaced
itself with *Loading…* or the error, and never asked again, so it had to be closed and reopened.

**Rejected.** Showing the last session's settings read-only, as the library is shown
(`lib/core/lastSession.ts`): three sections would each need a read-only mode, and a stale provider
or folder shown as current is worse than a line that says what is awaited. Polling from the
drawer: the readiness gate already polls, and `shell.status` is its answer.

**What it opens.** Checked in the dev preview with the engine cut off in the page and then let
back: rail, General and the waiting line, then the open page filled; `npm test` 416/416,
`svelte-check` 0/0. Not seen: a cold start of an installed build, and Getting started's waiting
line (it needs a start with no setup state yet).

## 2026-11-11 (6) — A broken extraction pool no longer stops an ingest run (KI-92)

**What changed.** `ingest/workers.py::warm_extraction_cache` wraps the whole pool block: a failure
of the pool itself logs `extraction_warmup_pool_failed` and returns, with every document it did not
finish counted as failed and left to the serial pass. `tests/unit/test_ingest_workers.py` gains a
stub pool and a test that breaks it at the first, second and third `submit`; the frozen-guard test
now uses the stub too.

**Why.** CI failed on the `v0.7.3` commit with `BrokenProcessPool` raised at `pool.submit`, outside
the per-result guard. The test fakes `sys.frozen`, so every real child starts with
`--multiprocessing-fork` and dies (30 of 30 jobs in a probe here); it passed when the break reached
`fut.result()` and failed when it reached the next `submit`. The test was flaky, and the code under
it was wrong: the warm-up's docstring promises a failed pool leaves ingest as slow as serial.

**Rejected.** Catching only `BrokenProcessPool`: a child that cannot start raises other errors, and
the step is optional either way. Retrying with a fresh pool: what killed one child may kill the
next, and the serial pass already does the work.

**What it opens.** CI's Linux runner, where the race showed, is the real check; the two failures
were on runs 40041172523 and 43070327008.

## 2026-11-11 (5) — The state lines say v0.7.3

**What changed.** AGENTS.md's *State*, the README's *Status* and the ROADMAP's *Where the product
stands* and *Releases* now name v0.7.3 (2026-11-11) and what it added; the README's *Next* names
the chat and MCP, as the ROADMAP's order does. ROADMAP row 60's current security step is S-9, the
prompt fence: S-7 was done on 2026-11-04 and the row still named it.

**Why.** `docs/RELEASE.md` §0 asks for it at each release, and three surfaces still said v0.7.1.
The test counts are the tag's: 3,233 Python tests collected (CI's own count on `a8408c0`: 3,226
passed, 6 skipped, 1 failed, KI-92) and 416 desktop tests.

**What it opens.** KI-92, filed beside this: the worker test that failed CI on the tagged commit
is a real escape at `pool.submit`, not only a flaky test.

## 2026-11-11 (4) — The PDF fingerprint no longer carries the app's version (KI-91)

**What changed.** `ingest/cache.py::_extraction_closure` treats a class defined in `extractors`
as code, not a dependency: a new `_class_parts` walks its methods (and any base written there)
like the module's functions, hashes its class-level values as constants (`_module_attr` resolves
`Class.name`), and adds only a foreign base's package to the module list. Two guard tests: no
format's fingerprint moves with `doc_assistant.__version__`, and an edit to
`_CellTextOfBlocks.__call__` moves `.pdf` and nothing else. A CHANGELOG `[Unreleased]` *Fixed*
line says the next update marks every PDF *changed* one last time.

**Why.** The KI-64 ligature fix's two wrapper classes live in `extractors`, and the walk mapped
any class it reached to its package's version, so `doc_assistant=<version>` was in the PDF
fingerprint and every release since 0.7.0 marked every PDF *changed*. Both tests failed first; the
second showed the converse hole: an edit to those classes' methods between releases was invisible
(KI-40), masked only because a release moved the version anyway.

**Rejected.** Dropping `doc_assistant` from the module list alone: it fixes the release noise and
keeps the KI-40 hole. Hashing the whole `extractors` module when a local class is reached: every
EPUB fix would re-read every PDF again (KI-48).

**What it opens.** Measured: `.pdf` moved once (`72f4aab8da6266fa` → `b4d61cccbf2cbc78`), the
other eight formats unchanged, and the PDF closure holds six methods of the three KI-64 classes.
The 0.8.0 upgrade run (A9) should show every PDF *changed* once; the one after it, none.

## 2026-11-11 (3) — 0.7.3's notes say every PDF reads *changed* after the update

**What changed.** The 0.7.3 CHANGELOG no longer says nothing is read again: its summary and a new
first *Known limits* entry say every PDF is listed as *changed* after the update, though none is,
and that reading them again is not needed. The release checklist's upgrade row (§7) expects the
same, naming KI-91.

**Why.** The upgrade run from the real 0.7.2 installer in the sandbox (2026-11-11) kept every
record (documents, titles, the folder, chunks, conversations), then listed the 4 PDFs *changed*
and the text file *ingested*. Cause: the PDF extraction fingerprint hashes the version of the
package each reached class comes from, and the KI-64 ligature fix's two helper classes live in
`extractors`, so `doc_assistant`'s own version is hashed (KI-91). Every release since 0.7.0 does it.

**Rejected.** Fixing it before the tag: 0.7.3 would mark every PDF *changed* either way, because
the fix moves the fingerprint once more, and it would mean a change to the cache code after the
gates, then the gate, the build, RG-012 and the upgrade run again (the user's call: ship).

**What it opens.** KI-91's fix in release A; that upgrade marks every PDF *changed* one last time.
RG-012 on this build passed (`run-20261111-152202`, 3 of 3 turns cited); the build is unchanged.

## 2026-11-11 (2) — Preflight counts the Windows window config as a build input; CI is green again

**What changed.** `scripts/release_preflight.py` lists
`apps/desktop/src-tauri/tauri.windows.conf.json` in `SHIPPED_PATHS`, and
`test_the_shipped_path_list_is_pinned` pins it.

**Why.** ADR-065 (`f35cabe`, merged in `5ed8bf5`) added the file: Tauri merges it over
`tauri.conf.json` on Windows, where it removes the native frame, so it changes what ships. Left
out, `test_every_tracked_app_file_is_shipped_or_named_as_not` failed on every CI run of `main`
from `5ed8bf5` to the release commit `823bf3b`, and `artifact_fresh` could not see a change to it
(KI-58's sixth instance). Found by 0.7.3's full gate: 1 failed, 3,229 passed.

**Rejected.** `NOT_SHIPPED`: the file is in the installed app's window behaviour.

**What it opens.** The release is tagged at this commit, not at `823bf3b`; the build is made from
it.

## 2026-11-11 — An engine error reaches the installed window as a 500, not as "Failed to fetch"

**What changed.** `apps/api/server_errors.py` adds `ServerErrorCorsMiddleware`, mounted innermost
in `create_app`, inside CORS. A route that raises before sending now gets a 500 that carries the
CORS headers; the error is re-raised, so the server still logs it. A response already started (the
chat's event stream) is left alone. Tests: the window-origins test gains a 500 for both window
origins (it fails without the fix); `test_api_server_errors.py` pins the before/mid-stream split.

**Why.** The 0.7.3 report *"Couldn't load connections: TypeError: Failed to fetch"*. The installed
window calls the engine from another origin, and Starlette answers an unhandled exception outside
CORS, so the 500 was discarded and read like a dead engine. In dev, Vite proxies on the same origin,
so the same failure showed as a 500. The report's own cause is not known: no document reproduces
it on a copy of the library, and the installed engine keeps no log (KI-90).

**Rejected.** A FastAPI handler for `Exception`: Starlette mounts it on the outermost layer, outside
CORS, so the header would still be missing. A JSON body: the window's clients fall back to
"X failed: 500" only on a non-JSON body, so Starlette's plain text is kept.

**What it opens.** KI-90 (a log file for the installed engine) and KI-79's health poll, for
release A. The CHANGELOG line is written at `docs/RELEASE.md` §2.

## 2026-11-10 (2) — A digest is registered, and Docker joins the release gates

**What changed.** `[generate]` registers `.claude/DIGEST.md`: every known issue's heading and
status, the open tickets and the newest DEVLOG headings, ~1,600 tokens. `AGENTS.md` numbers it
fourth in what every session reads. `docs/RELEASE.md` §3 gains a Docker row: read the image's CI
job green on the release commit, and re-read the `Dockerfile`, the compose file and the setup
guide when the CHANGELOG names a new dependency, variable, port or data path.

**Why.** The entry file told each session to scan `docs/KNOWN_ISSUES.md`, ~31,500 tokens. The
user, 2026-11-10: the digest should be standard, and Docker is part of release work.

**Verified.** `generate --check` and `docs_check --strict` pass; `keypoint session-start` lists
the digest fifth. No release was cut, so the Docker row has not been used.

**What it opens.** A triage of the 29 open known issues, drafted for the user's ruling.

## 2026-11-10 — cpc 1.18.1 → 1.22.1: two files leave `.claude/`, cpc's templates leave `docs/`

**What changed.** Re-vendored cpc 1.18.1 -> 1.22.1 from a worktree of the `v1.22.1` tag
(`tools/conventions/cpc/`, 64 modules; `tools/conventions/templates/` is new). The known issues
and the friction register moved to `docs/KNOWN_ISSUES.md` and `docs/FRICTION.md` with `cpc-move`,
and `.gitignore` names both, so they stay local (ADR-029). The four template copies cpc had laid
(`docs/decisions/ADR-000-template.md` and its pivot, `docs/specs/SPEC-000-template.md`,
`docs/sprints/SPRINT-000-template.md`) are deleted: cpc's are in `tools/conventions/templates/`.
`AGENTS.md` numbers the three files every session reads and lists the rest as read on need;
`docs/local-only.md` and three `.claude/` files name the new paths.

**Why.** cpc 1.22.0 sorts documents by when they are read, with `.claude/` for every session and
`docs/` for a task, and its init check fails until the known issues move. 1.22.1 keeps cpc's
templates in cpc's folder. The user, 2026-11-10: update this project first, and keep the two
moved files out of git.

**Rejected.** Keeping the known issues in `.claude/` with a `[paths]` line: it would leave this
project off the layout every other one moves to.

**Verified.** `init_check`: 0 errors. `docs_check --strict`: 0 errors and 0 warnings, where the
re-vendor alone gave 4 errors and 16 warnings. A baton rotation of 6 entries cleared 5 warnings.
The user ruled the other 11 the same day: caps of 650 tokens for one baton or DEVLOG entry,
transitional, until 8 entries written before cpc's caps rotate out; 9,000 for the session-start
files, measured at ~8,458; a `## Releases` section in the ROADMAP with no entry, so cpc's release
checkpoints stay off; a `## Concept map` in `docs/architecture.md`, drafted from the import graph.

**What it opens.** The concept map's grouping is a draft and its May-import column records
today's imports, some of which run upward; correct it before wiring `cpc-dod-lint`. The two entry
caps go back to 300 and 500 once the long entries rotate out. Three shipped specs still name the
old sprint template path.

## 2026-11-09 (3) — New README media: a three-beat demo GIF and five screenshots

**What changed.** `docs/assets/provenote-demo.gif` is re-recorded on the current UI (1280×800,
21 s, 18 frames, 0.79 MB, light theme) in three beats: a question streams into a cited answer; a
citation opens its passage and *Show the page* opens the printed page beside the document page;
back in the answer, its sources and one flagged *uncited* claim. Five PNGs are new in
`docs/assets/` (112–223 KB each): the answer with a flagged claim and the source viewer under
*Why Provenote*, and the Library grid, the library summary and the concept graph with a chosen
definition in a two-column table under *What you can do with it*. The GIF's alt text and the
`docs/DEMO.md` line that describes it now say what is filmed.

**Why.** The user, 2026-11-09: new media for the rewritten README; the old GIF predated the title
bar, the settings cog and the library summary.

**How.** Recorded over the public demo corpus (28 arXiv papers) in a new scratch data home outside
the repo, never `data/`, with Ollama `llama3.1:8b` forced (the recorder aborts unless the page
reports that data home and an Ollama engine). The answer is real model output, not a mock; the
recorder retries a fresh chat until an answer carries a flagged claim (3 attempts). Toolkit:
puppeteer-core + installed Chrome + Pillow, kept out of the repo as before.

**Rejected.** Mocking `/api/chat` with a written answer (a staged answer is not the product). The
library summary and the grid in one frame (the summary is a separate view that replaces the grid).
The keyword-filter dialog as the grid shot (it covers the grid). The optional add-documents review
sheet (not recorded).

**What it opens.** The definition panel names its source and links the passage but shows no page
number, though the README says definitions are quoted "at their page". At 375 px the two-column
table's images are 144 px wide.

## 2026-11-09 (2) — ROADMAP 113: the eval harness is reinforced before the chat work

**What changed.** A new row, 113 (F11 Quality), and one sentence each in the ROADMAP's *Next* and
F11 blocks. No code. The sessions are in the local plan.

**Why.** The user, 2026-11-09: *"Add to the plan for 0.8.0 [...] a reinforcement of the harness
and benchmarks. Since one of the planned goals is to work on the chatbot part, it seems
appropriate."* Read against the plan, four chat sessions are gated on a measurement the harness
cannot take: the prompt fence on 10 questions with no hostile passage, context handling with no
conversation case, lookups with no record or not-in-the-library case, and modes with citation
coverage that is not a scorer.

**Rejected.** Reinforcing the harness after the chat work (its *before* runs would be lost).
Adopting an evaluation framework (*What NOT to do*: no framework, no new dependency).

**What it opens.** The spec decides the case families, the statistic, the model panel and the
paid budget. Two pieces need the user's own reading: verifying the new references and grading a
sample of answers for the judge check. Row 15 (the harness as its own repository) stays after.

## 2026-11-09 — The README leads with the product; the full limitations move to `docs/limitations.md`

**What changed.** `README.md` goes from 365 lines to about 190. It opens on a one-line pitch, the
demo and a download link, then *Why Provenote*, *What you can do with it*, *Get it*, one benchmark
table (*Measured, not asserted*), *How it works*, a six-item *Limitations* list, a short *Status*
and the documentation table. Three badges are new (CI, latest release, licence). The Limitations
section moves verbatim, links re-based, to the new `docs/limitations.md`. The README's *How long
does indexing take?* section is removed: `docs/performance.md` § 1 already holds each of its
figures. `docs/OVERVIEW.md` and `docs/QUICKSTART.md` point at the new file.

**Why.** The user, 2026-11-09: a better description of the project in the README, more
"marketable", with the technical detail moved to linked files.

**Rejected.** Dropping the benchmark caveats from the README (results stay indicative, with their
limits beside them). Cutting Limitations from the README altogether.

**What it opens.** The demo GIF predates the title bar and the library summary; new media is
planned, not recorded. The local-model citation figures still predate the 0.7.0 fix. The removed
indexing section was the only plain-language version of those figures.

## 2026-11-06 (10) — The settings cog moves beside the menu (ADR-065 choice 7)

**What changed.** The top bar's *Settings* button moves from the right end, between the wordmark
and the window controls, to the left, right after the menu (`shell/Topbar.svelte`). Its name and
hover text are unchanged. The bar's right end now holds only the wordmark and, on Windows, the
window controls. Two comments that described the old place are updated (`Topbar.svelte`,
`screens/library/HeaderBar.svelte`), and ADR-065 choice 7 gains its *As built* note.

**Why.** The user, 2026-11-06: *"The settings cog should be moved to a better place now."* Beside
the window controls it read as part of the window's frame. Of the four places offered, the user
chose beside the menu.

**Rejected.** The sidebar's footer, before the wordmark, the status bar (ADR-065 choice 7).

**What it opens.** Checked in the Vite preview over the invented library: the button order, the
hover text, the click opening the Settings drawer, and the light theme at 375, 640 and 1280 px
with no sideways scroll. Along the bar, with the drag script's rules applied point by point, the
cog never starts a drag and the empty space still does. Not checked in a Tauri window, nor by a
screenshot (the browser pane was hidden and did not draw). The app menu still has its own
*Settings* entry, so Settings now has two doors a few pixels apart.

## 2026-11-06 (9) — The top bar is the window's title bar on Windows (ADR-065)

**What changed.** On Windows the window has no native title bar any more
(`src-tauri/tauri.windows.conf.json`: the main window restated with `decorations: false`,
`shadow: true`; other platforms keep their frame). The top bar takes its place: its empty space
drags the window and a double-click maximises or restores it (`data-tauri-drag-region="deep"`,
`"false"` on the app menu's wrapper), and `shell/WindowControls.svelte` draws minimise,
maximise/restore and close at its right end, full height, the platform's look, Close red on hover
(`--caption-close`). The window calls go through `lib/core/tauri.ts` and degrade to nothing in a
browser. Four window permissions join the capability set. Built in `f35cabe`; the hover texts came
with ROADMAP 110's merge (`5ed8bf5`). ADR-065 records it, written after the build at the user's
request. The user then chose where the settings cog goes, beside the menu; not built yet.

**Why.** The user, 2026-11-06: the native title bar is *"not modern (dead space)"*: a row of its
own (31 px at 96 DPI) above a top bar that already had empty space.

**Rejected.** Keeping the native frame; frameless on every platform (Windows-style buttons on a
macOS window); native hit testing to keep the snap layouts on hovering maximise (window-procedure
code Tauri does not provide; Win+Z still opens them). ADR-065 § Options.

**What it opens.** Driven in a dev window over an invented library: drag, double-click maximise,
restore, a maximised frame equal to the work area, Close at the corner pixel, Win+Z. Not driven
yet: minimise and close by click, edge resizing, keyboard focus, the light theme, a 640 px window,
a dialog over the bar. They are a row in `docs/release-ux-checklist.md` §6, before 0.7.3's tag.
The cog's move to the left of the top bar (ADR-065 choice 7).

## 2026-11-06 (8) — The library summary stops saying indexing reads no metadata

**What changed.** The *Metadata* card's two lines (`lib/library/summary.ts`,
`screens/library/Summary.svelte`) now say indexing fills metadata for new or changed documents
only, and that *Re-run… > Metadata* reads it for documents indexed earlier. ROADMAP 112's
"Exposed" note says the same. The merge of ROADMAP 110 (`5ed8bf5`) also gave the title bar's three
window buttons their hover text, because `ui-rules.test.ts` check 2 failed without it: *Hide the
window in the taskbar*, *Make the window fill the screen* / *Return the window to its earlier
size*, *Quit Provenote*.

**Why.** Since `3a1966f` (merged `1123b3d`) indexing fills a new document's blank title, authors,
year and DOI, so "Indexing does not read metadata yet" was wrong for every document indexed after
it. It stays true for documents indexed before, and nothing re-reads those on its own.

**Rejected.** Naming a version in the copy ("since 0.7.3"): the reader of the card knows when a
document was added, not which build added it.

**Opens.** ADR-065 (the title bar) is cited by the code but not yet written; the window buttons'
texts belong in it.

## 2026-11-06 (7) — ROADMAP 110 (d): one dialog frame, and no colour outside the tokens

**What changed.** `shell/Dialog.svelte` is the one dialog frame: backdrop, frame, *Esc*, focus in
and back, *Tab* kept inside, as a centred card, a side panel or a screen. The 20 components that
drew a modal of their own render it, and a list of open dialogs (`lib/shell/dialogs.ts`, tested)
makes *Esc* close the one in front and puts a dialog opened from another above it, which replaced
three hand-written *Esc* guards and the fixed layers. The 36 literal colours are tokens
(`--scrim`, `--paper`, `--shadow-side`, the fallbacks dropped). `src/shared.css` holds the dialog's
title, head and buttons. `ui-rules.test.ts` gains checks 4 (no `aria-modal` outside the frame)
and 5 (rules copied between components: 211 → 179, a ratchet).

**Why.** ADR-063 choice 8 and the user's go-ahead for piece (d) on 2026-11-06. The frame is markup
and *Esc* handling as well as style, so it is a component, not a class. Seen live over an invented
library: the nested confirm's layer, focus and two *Esc*s, the rename field, Settings, Indexing,
the source panel's handle, light, dark and 375 px. `npm test` 409/409, `svelte-check` 0, build ok.

**Rejected.** The native `<dialog>` with `showModal()`: its top layer would sit above the shared
tooltip, which must show over every dialog. Centring with `transform`: it makes a fixed element
inside the frame measure from the frame. A per-file list for the copied rules: 60 files long, and
a total that may only fall says the same. Moving `.hint`, `.err` and `.iconbtn` now: three
variants each, so unifying them changes spacing on many screens; left to the change that touches
each.

**What it opens.** Behaviour that changed and the user has not seen: the dark theme's backdrop
now darkens the screen behind a centred dialog (it lightened it), the review closes on *Esc*, and
focus returns to the opener. Still with the user: the 20 marker meanings, three button looks check
1 cannot see, the staged bar's *Clear*, the × in *Manage folders*.

## 2026-11-06 (6) — ROADMAP 110 (c): every button says what it does

**What changed.** The 236 buttons in 59 files that had no hover text have it, on the shared
tooltip, so check 2's allow-list is empty. Each text finishes "this will …" without repeating
the label, says what a quiet act leaves alone and what comes back, and follows the state: a
disabled button says why (*Send* "Type a question first"). The top bar's *Chat*, *Library* and
*Graph* tabs carry an `aria-label`, since under 780px their labels are hidden and they had no
name. `ui-rules.test.ts` gains a fifth check: an icon-only `<button>` needs `aria-label` or
`aria-labelledby`. Of the 50 `title`s on other elements, the Chat tab's scope `<select>` (the one
that takes focus) moved onto the tooltip; the rest are sorted in ADR-063's *As built (c)*.

**Why.** ADR-063 choice 6, and the user's go-ahead for piece (c) on 2026-11-06. The texts were
written from each button's handler, not from its label, after reading what it does: the
"Promote" on a gap marks it as being acted on, a sample question fills the box and does not send.
Seen in the running app over an invented library, dark, light and 375 px: hover, Tab focus with
`aria-describedby`, a disabled reason, a text changing with the selection. `npm test` 402/402,
`svelte-check` 0.

**Rejected.** Moving all 49 other `title`s onto the tooltip: on an element that cannot take focus
the tooltip never shows to a keyboard or a screen reader, and it removes the `title`, which a
screen reader can read today, so 20 marker meanings would get worse, not better. Generic texts
("Click to open"): they say nothing the label does not. An allow-list entry for the list rows:
every one had something to say.

**What it opens.** Piece (d): the dialog frame, the shared stylesheet, 36 colours; and, with the
user, the 20 markers whose meaning lives only in a `title`, three button looks check 1 cannot see
(the staged-files bar, the claim review, *Detect* in Manage keywords) and the staged bar's *Clear*.

## 2026-11-06 (5) — ROADMAP 110 (b): every button on a role, and a delete always asks

**What changed.** The 28 components that styled their own `.primary` / `.ghost` / `.danger`, and
the local primaries under other names (`.send`, `.resume`, `.addbtn`, `.secondary`, `.neutral`,
`.dangerbtn`), are on `btn` / `btn-primary` / `btn-quiet` / `btn-danger`. Their bare `button`
rules are gone too, since they outranked `.btn`. `app.css` gains `btn-primary btn-danger` (a
confirm's own button when it removes, filled red, `--danger-fill` in all four palettes),
`btn-icon`, and a pressed look for any `btn` with `aria-pressed`. `ConfirmDialog` is an alert
dialog: labelled, described, focus on *Cancel*. The five destructive actions ADR-063 counted
(*Remove key…*, *Remove server…*, *Remove…*, *Remove all…*, *Dismiss…* on a chosen definition,
which was *Clear*) are red and confirm, and so are *Reset to default…* in *Edit metadata* and
*Undo all…* in the add-documents review. The document page's *Source* and *Re-run…* left the
index's `nav` for its trailing end. The Library header wraps at 375 px. Check 1's allow-list is
seven exceptions with reasons; hover text 237 → 236 and colours 49 → 36 (two files' dead
`var(--danger, #c0392b)` fallbacks).

**Why.** ADR-063 choices 1, 4, 5 and 7, and the user's go-ahead for piece (b) on 2026-11-06. Seen
in the running app over an invented library, dark, light and 375 px; *Remove server…* and both
ignore-list removals driven end to end, cancel and confirm. `npm test` 401/401, `svelte-check` 0,
`npm audit --audit-level=high` clean.

**Rejected.** Keeping the confirm's tinted red fill: 4.1:1 in the light theme (3.7:1 hovered),
below WCAG's 4.5:1. A small-button size: one size meets the 24px target everywhere it was
used. Keeping *Pause* amber: it removes nothing, so it is ordinary. Renaming the gap badges'
`.danger` to pass the check: it names a severity from the data, so it is an exception with its
reason, not a disguise. A confirm on removing a document from a folder: the row's checkbox
undoes it, and choice 4 says that still confirms, so it is left for the user to decide.

**What it opens.** Piece (c), hover text on the 236 buttons left; piece (d), the dialog frame,
shared stylesheet and 36 colours. Not seen live for want of data in a new library: the
definition's *Dismiss…*, *Remove key…*, *Undo all…*, the indexing screen.

## 2026-11-06 (4) — ROADMAP 110 (a): the primary's fill, one tooltip, and checks that fail

**What changed.** The first of ROADMAP 110's four pieces. `app.css` gains `--accent-fill` and
`--accent-fill-fg` in all four palette blocks (`#6b5fd6` with white in the dark theme, 5.0:1;
the accent itself in the light, 8.2:1), and `btn-primary` uses them; its hover now darkens toward
black in both themes. One tooltip, `apps/desktop/src/lib/shell/tooltip.ts`, is a Svelte action
over one shared `role="tooltip"` element: it shows on hover and on keyboard focus, Esc dismisses
it, the pointer can move onto it, and the button's `aria-describedby` points at it while it
shows. All 73 native `title`s on buttons moved onto it (about 30 rewritten to say what the button
will do rather than repeat its label), and three icon-only buttons whose only name was that
`title` gained an `aria-label`. `src/ui-rules.test.ts` holds ADR-063's three checks as ratchets
at today's counts, per file, with reasons in `src/ui-rules.allow.ts`, plus a fourth with no
allow-list (no `title` on a button). ADR-063 gains *As built*; ROADMAP 110 lists the four pieces.

**Why.** ADR-063 choices 2, 6 and 8 and its *Enforced by*, accepted by the user on 2026-11-06;
the user picked this piece first from the proposed split. Seen working in the running app over an
invented library (a new data home, no `.env`, no key): dark, light and 375 px, a disabled button
included. `npm test` 401/401, `svelte-check` 0 errors, `npm audit --audit-level=high` clean.

**Rejected.** Moving only the 26 `title`s the ADR counted: its regex stopped at an arrow
function's `=>`, and a real scanner finds 73, so the rest would have kept two kinds of hover text.
Keeping a native `title` beside the tooltip as a fallback: two texts on one hover. Showing the
tooltip on every focus: a click focuses a button too, and a dialog focuses one by script on
opening, so it shows only after a navigation key moved the focus. Letting Esc through to the
dialog underneath: the tooltip is the top layer (the WAI-ARIA pattern), so the first Esc closes
it alone. A total count per check: a file could break a rule while another improved, so each
file has its own entry, and an entry above the measured count also fails.

**What it opens.** Pieces (b) roles and confirms, (c) hover text on the 237 buttons left, (d)
dialog frame, shared stylesheet and colours, each lowering its list. 50 `title`s on elements that
are not buttons reach a mouse only (ADR-063 *As built*). The corrected counts replace the ADR's
26, 286, 312 and 45 with 73, 237, 310 and 49.

## 2026-11-06 (3) — ADR-063 accepted as proposed

**What changed.** ADR-063 is accepted with all eight choices as recommended: the four action roles,
the Graph tab's look for ordinary buttons, a deeper accent fill with white text for the primary in
the dark theme (`--accent-fill`, `--accent-fill-fg`), *Done* as a mode's trailing primary, a
confirm on every danger action, *Source* and *Re-run…* as ordinary buttons beside the document
page's index, hover text on every button through one tooltip, the standards ranked over design
systems, and styling in one place with three checks that fail. ROADMAP 110 moves from waiting to
planned now, and its index line in `docs/decisions.md` says accepted.

**Why.** The user, 2026-11-06: *"Accept as proposed"*, after the revision in entry (2).

**Rejected.** Nothing new: the options are in the ADR.

**What it opens.** ROADMAP 110, the sweep over the components, once no other session has edits
in flight in them.

## 2026-11-06 (2) — ADR-063 revised: hover text, styles in one place, checks that fail

**What changed.** ADR-063, still proposed, gains three choices and an *Enforced by* that fails.
Choice 6: every button has hover text that says what it will do, shown on keyboard focus too and
meeting WCAG 2.2 SC 1.4.13, through one shared tooltip instead of the native `title`. Choice 7:
WCAG 2.2 AA and the WAI-ARIA Authoring Practices patterns decide what is correct, design systems
explain why, and the code names the pattern it follows (alert dialog, toolbar, listbox). Choice 8:
tokens and role classes in `app.css`, repeated small rules in one shared stylesheet, one dialog
frame, no literal colour in a component, every exception commented and allow-listed. Three
`npm test` checks with allow-lists (local button classes, a button without hover text, a literal
colour), landed first as ratchets. ROADMAP 110 widens to carry them; GLOSSARY U-003 names *hover
text*; the ADR's file names follow ADR-064's move.

**Why.** The user, 2026-11-06, relayed by the session that started this one: buttons share a
design, explain themselves on hover, carry the right colour and a confirm when dangerous, follow
industry conventions, keep styling conventions across the project with few exceptions, and the
project is stricter about it. Measured the same day: 26 of 312 buttons have a `title`, and the
`<style>` blocks of 20 components hold 45 literal colours.

**Rejected.** The native `title` as the hover text: it never shows on focus and cannot be
dismissed. Checks that fail at full strength on the day they land: 286 buttons would fail at once,
so each starts as a ratchet at today's count. Building the sweep now: ADR-064's pass 2 is editing
the same components in this checkout.

**What it opens.** ROADMAP 110, after ADR-064's pass 2 is committed and the user answers
ADR-063 (choice 2, the dark-theme primary; choices 6 to 8).

## 2026-11-06 — A dialog's door calls its module, not a callback through App (ADR-064, pass 2b)

**What changed.** Four rune modules in `lib/library/` take App's one-domain state:
`reingest.svelte.ts` (*Re-run ingestion*), `documents.svelte.ts` (the list, *Edit metadata*,
*Delete this document?*, select mode), `folders.svelte.ts` (*Manage folders* and the folder
writes) and `keywords.svelte.ts` (*Manage keywords* and the family writes). Each door calls its
module (`PageIndex` calls `openRerun`, as ADR-064's worked example drew it), and the props that
only carried a door up through two components are gone. App keeps the glue: anything that also
touches another domain (the health refresh after a re-run, the facet re-point after a family
write, the graph latch). App: 1,778 → 1,478 lines. Driven in the preview, read-only: every door
opens its dialog with the right label and parts and cancels (Re-run from the page names the title,
from the select bar "2 documents"); *Manage keywords* opens at the top from the filter, on Concepts
from the Graph tab, on its row from *Edit*; the chat delete confirm still names the count; 375 px
light and dark, no script error. npm test 391/391, svelte-check 0/0.

**Why.** ADR-064 choice 5, accepted: a person who sees a dialog finds its state beside its
domain, and a new dialog adds a module function, not a flag, a callback prop and a mount in App.

**Rejected.** Moving conversation deletes: both confirms read the chat view and start a new chat,
which `conversations.svelte.ts` already leaves to App. Pushing App to choice 5's 600–800 by moving
the chat lifecycle or navigation: choice 5 keeps those, and the estimate (~830 one-domain lines)
was read from function names; ~300 were.

**What it opens.** The writes the modules hold (create a folder, save metadata, delete a document,
write a family, a re-run to the end) were not driven live: same API calls, moved, but a check needs
a scratch copy of the library. Smaller App candidates are in ADR-064's "As built — pass 2b".

## 2026-11-05 (10) — Eight screens cut into the parts they show (ADR-064, pass 2a)

**What changed.** Eight of the nine large files are cut along their headings and sections, each
part a component beside its entry file, with the entry's props unchanged: Settings (its four
sections; the provider list and forms), Manage keywords (five parts), the Sidebar (Chat history
leaves for `screens/chat/ChatHistory.svelte`, which App hands to the frame as a `chatRail`
snippet), Graph (the concept panel), Taxonomy (the forest, a field's detail), the document page
(its index, header, Chunks) and the Library tab (the header bar). ADR-064's "As built" has the
sizes and every place the plan was not followed. Driven in the preview against the dev API:
each screen and part opens and acts as before (sorts, select modes, the ⋯ menu, wheel zoom, a
Chunks block to its page, the delete question answered *Keep*, Settings keeping a typed folder
across a section switch), at 375 px in light and dark, no script error; npm test 389/389,
svelte-check 0/0.

**Why.** The user accepted choice 6. Choice 5, App's state, changes the props of the screens whose
doors it rewires, so it comes after (2b), on smaller files.

**Rejected.** A library rail cut out of the Sidebar: it would copy ~120 of its 150 CSS lines.
Shared rail components in `shell/` (one tree row, one filter box): the layout test allows a screen
only `Icon` and `ConfirmDialog` from `shell/`, and widening that is a decision, not a split.
`:global` styles from the frame: they would outrank a rail's own rules.

**What it opens.** 2b. Five parts stay above 300 lines, each its own CSS or logic (ADR-064 names
them). The four splits other than the Sidebar ran in parallel subagents, each checked by
svelte-check, tests and a script comparison of markup and CSS against `HEAD`; the ADR's "one file
at a time" became one live pass over all of them.

## 2026-11-05 (9) — `oauthlib` 4.0.1: an advisory in a package that never ships

**What changed.** `uv.lock`: `oauthlib` 4.0.0 → 4.0.1 (`uv lock -P oauthlib==4.0.1`); nothing else
moved. **Found by:** the pip-audit gate on 2026-11-05: CVE-2026-49272.

**Why.** It comes in through `chromadb` 1.5.9 → `kubernetes` 36.0.0 → `requests-oauthlib` 2.0.0,
Chroma's cluster client. Nothing here loads it: `src/`, `apps/`, `scripts/` and `tests/` never
import it, a local `PersistentClient` round-trip loads none of the three, and the 0.7.2 backend's
`PYZ-00.toc` holds 157 `chromadb` modules and no `oauthlib` or `kubernetes` one. So it reaches the
dev venv and CI only, and no CHANGELOG line. 4.0.1 is the first fixed version, 5 days old.

**Rejected.** A reviewed ignore in `pip-audit-ignore.toml`: the code never runs, but a patch
release that moves one package is cheaper than an entry someone has to revisit.

**Checked.** The gate is OK against an export of the new lock with CI's extras (197 packages,
0 unreviewed, 6 reviewed ignores matched, 0 stale). The 677 tests in the 49 files that use Chroma
pass with 4.0.1 overlaid on the main venv. The full suite was not run.

**What it opens.** The main venv keeps 4.0.0, and the local gate fails there, until its next
`uv sync`.

## 2026-11-05 (8) — A sentence is never a document's authors (KI-89)

**What changed.** `metadata_extractor._looks_like_author_line` gains `_reads_as_prose`: a candidate
with four or more lowercase words in a row, which also has more lowercase words than capitalised ones
or opens like a sentence (one capitalised word, then the run), is not an author line, so the scan
goes on to the next line or returns None. The public case: `pointer_networks_vinyals_2015.pdf` stored
*"Nonetheless, these methods still require the size of the output dictionary…"* as its authors,
because the head's 3,000-character cut leaves an Introduction fragment short enough to pass every
existing test. Fifteen new tests; the five prose ones fail without the guard.

**Why.** A sentence has commas and an *and* too, and since the index run fills blank metadata
(entry (7)) this would reach every new document. Re-measured on a
copy of the library (KI-89): every prose list gone (one falls through to a venue footnote, still
wrong), no correct list lost, and three documents whose title had been taken as their authors now
get the real list by falling through.

**Rejected.** A list of English function words: it is the measuring stick in KI-89, not a rule,
because it assumes a language. A run limit alone: *"Jane Doe, John Smith, on behalf of the X
Consortium"* has a run of four and is a correct list. A majority of lowercase words alone: *"Smit and
van der Berg"* has one. The three-word limit is how names are built (a connective plus *van der* or
*de la*), not a number fitted to this library. Cutting the author scan at the abstract: a layout
that prints the abstract before the names would lose its authors, and it was not measured.

**What it opens.** The stored rows stay: neither the re-run nor the enrichment runner writes None
over a value, so a *Re-run… › Metadata* clears no prose list the extractor now rejects; whether it
should is KI-89's open question. A self-citation line that begins with the right authors (ACM's
*"Name and Name. 2020. Title…"*) is not prose and is kept whole.

## 2026-11-05 (7) — The app's own index run fills title, authors, year and DOI (KI-87's library)

**What changed.** `ingest.main` gains a post-loop step, `_fill_missing_metadata`, right after the
catalogue's pass (`_apply_imported_metadata`). It takes the hashes the run indexed (new documents,
and documents re-read because their text changed), and `metadata_enrich.fill_missing_metadata`
runs the deterministic extractor over their cached text. It writes a column only while it is blank.
It works in batches of 100: one read of the four columns, one read of the texts that still have a
blank, one write. A document with nothing blank is not read. Like the steps beside it, it logs a
warning and never fails the ingest. `_persist_metadata` (the `scripts/enrich_metadata.py` runner)
now shares the column rule and, like the catalogue's fill-the-blanks mode, counts an
all-whitespace string as blank. Seven tests in `tests/integration/ingest/test_ingest_fills_metadata.py`:
0 documents, a fill, a re-index that keeps an extracted value and one that keeps a catalogue's, a
text that yields nothing, and a failing step. The fill and the two re-index tests fail without the
step, and the re-index ones also fail when it overwrites.

**Why.** The app's index run and *Add documents* never read metadata, so a library indexed only in
the app had no title, author or year on any document (KI-87's library; found inventorying the
library summary's signals, ROADMAP 112). No ADR: ADR-049 D3 already orders the slots (extractor <
catalogue < the user's edit) and already puts a post-loop metadata pass in `ingest.main`; this adds
the lowest slot under that order. KI-54's own measurement recommends running the extractor on blank
titles (14 right, 1 journal name, of 16). **Cost, measured** on a copy of the 2026-11-05 backup
library (104 documents, cached text median 71,704 characters, all four columns blanked, served
through `DOC_DATA_DIR`): **102 ms for all 104 (98.6 to 110.4 ms over 5 runs), about 1 ms per
document**, of which the extractor is 0.36 ms. It filled 102 titles, 76 author lists, 93 years and
26 DOIs. A second pass, with 84 documents still holding a blank, took 73 ms; with nothing blank,
0.9 ms. Extracting one PDF takes 14.7 s on average (`docs/performance.md`), so the step is under
0.01% of a first index; a re-read of 10,000 documents would add about 10 s. End to end, two public
PDFs (64 and 9 pages) indexed into an empty data home took 52.2 s, of which the step took 43.6 ms,
most of it the first import of its modules. On the copy the step's values match what the library
already stored from earlier runs on 97 of 104 titles, 97 of 104 author lists, and every year and
DOI.

**Rejected.** Filling every document on every run: a Re-index would change documents it did not
index, with nothing on the run's screen saying so. Replacing the extractor's earlier answer on a
re-read: replacing it is what *Re-run…* › *Metadata* is for. Calling `enrich_metadata` per document:
it opens a session per document, and with no bound on how many texts it loads, a re-read of the
whole library would hold every text at once. Extracting inside `process_document`: that puts it in
the locked ingest core and ahead of the catalogue's pass.

**What it opens.** A library already indexed in the app is not filled by a Re-index (its documents
are skipped as unchanged): one *Re-run…* › *Metadata* over them fills it. KI-54's journal-name title
now arrives at index time on about one new document in sixteen, to be corrected as an edit. **A
second defect, not filed yet:** the author picker takes a sentence of body text on 9 of the 76
author lists it returned on the copy (for *Pointer Networks*: "Nonetheless, these methods still
require…"). The stored library held 10 such lists, 8 of them the same, so the step brings the
defect forward to index time rather than adding it; a structural guard in `_extract_authors` is
its own fix.
`reingest._rerun_metadata` on an imported document re-applies the catalogue and leaves the fields it
lacks blank; the index step fills them. The library summary's *missing metadata* now means the
extractor found nothing, for documents indexed from here on. The spec's *Opens* line and a ROADMAP
row are the spec author's to update.

## 2026-11-05 (6) — Every component moves to the screen it shows (ADR-064 accepted, pass 1)

**What changed.** The user accepted ADR-064's seven choices as recommended. Pass 1, moves only: all
54 components left `apps/desktop/src/lib/` for `src/screens/<screen>/` (chat, library,
document-page, graph, manage-keywords, taxonomy, settings) or `src/shell/`, 28 of them renamed to
what the app shows (`LibraryBrowser` → `DocumentPage`, `ReingestDialog` → `RerunIngestion`,
`Sources` → `ManageFiles`, `LibraryTaxonomy` → `screens/taxonomy/Taxonomy`). Imports, component
tags and the comments naming a component follow; git records all 54 as renames (93–100%
similar). `src/layout.test.ts` holds the folder rules (no `.ts` in `screens/`/`shell/`, no
component in `lib/`, nothing in `lib/` or `shell/` imports a screen, a screen takes only `Icon` and
`ConfirmDialog` from `shell/`, every relative import resolves); it failed four of five on a copy with
planted violations. The security test's sink path, `docs/architecture.md` (the spine table's last
column, and a new section: every screen and dialog with its doors), GLOSSARY U-001/U-002 and the
new U-004, `docs/security.md` S2, ROADMAP row 111 and `apps/desktop/CLAUDE.md` follow.

**Why.** The user asked for a layout where a person who sees a screen finds its code in one place.
The rewrite ran first on a copy of `src/` with an import-resolution check, then for real once
`c2b2fef` left `apps/desktop/` clear.

**Rejected.** Splitting in the same commit: a file renamed and cut in one commit can fall under git's
50% rename threshold and lose its history, so pass 2 starts after this is committed. Renaming the
wire types that share a component's old name (`ConceptGraph`, `DocConnections`, `DocReferences`):
they mirror `apps/api/models/` and stay.

**What it opens.** Pass 2: the nine files of 690 lines or more, and App's one-domain state into rune
modules. ADRs, dated specs and archives keep the old paths as the record of their time.

## 2026-11-05 (5) — The library summary follows the user's seven recommendations (ROADMAP 112)

**What changed.** Six of the seven, in the wireframe (revision 2, four screens, rev 1001), the spec
(§ As built) and the screen. A count is navigation: each count row is its own target (hover,
chevron), a zero row is plain text, and *Show the N* buttons are gone. *Need attention* leads and
scrolls to *From indexing*; passages and pages are a context line. Meters read "N of M" with a thin
bar. Each *From indexing* row names its fix: files failed in this session's run open the indexing
screen, other files Settings › Documents, document issues the grid. "Not read" is said once: the
wire gains `metadata.with_any_field`, and at 0 the Metadata card is one line; references, figures
and keywords say *not looked for* while none is on record anywhere. Bars are horizontal, one hue,
value at the tip, square at the baseline. Driven again on invented libraries (60 documents; 140
with nothing read), at 1280 px dark and 375 px light, where the fix moves under the reason and a
meter's bar under its words (container queries).

**Why.** The user approved revision 1 and then sent the seven (*"Send all"*), on the condition that
the build follows ADR-063.

**Rejected.** *Years* sorted by count (recommendation 6): decades are an ordered scale and the dataviz
guide keeps an ordinal axis in order; *Types* are ranked. A lone *No year* bar when no document has a
year: a one-bar chart, so a sentence. Recommendation 7 (keep each count run for a trend): later work.

**What it opens.** The not-read rule is a heuristic on counts until a re-run records that it looked;
the canvas toolkit read revision 2 after the user's two open tabs had written revision 1 back into
the restarted canvas (T-023's "gives a restarted server back whatever it lacks"), so it was redrawn
on a fresh port.

## 2026-11-05 (4) — The library summary, built (ROADMAP 112)

**What changed.** The Library pane has a third layout, *Summary*, for the whole library or the open
folder, built to the wireframe the user approved with no markup. `library/summary.py` reads one fact
record per document and defines ten *issues* once (`no_text`, `flagged`, `missing_year`, …); the
counts sum them and `list_documents(issue=…)` selects by them, so *Show the N* opens exactly N, as a
new grid collection (`issue`). `GET /api/library/summary[?folder_id=]` is read-only and adds this
session's run failures; `GET /api/library/documents` takes optional `issue` and `folder_id`. Driven
in the preview on invented libraries (60 documents, then 0 with 6 files waiting): the three screens,
*Show the 7* → 7 documents with no year, a folder's summary and its door back, Settings opening at
Documents, 375 px in the light theme with no sideways scroll; dark at 1280 px.

**Why.** The user, 2026-11-05 (ROADMAP 112). The numbers say *on record* where the app never looks
(the spec's signal table): its indexing reads no metadata, references or figures.

**Rejected.** SQL predicates per issue: the effective-metadata rule (ADR-013) lives in Python, and a
second copy in SQL would be the drift this feature's one-definition rule exists to prevent. An `IN`
clause for the filter: SQLite's variable limit at 10,000 ids. Persisting *Summary* as the layout:
the Library would open on counts instead of documents. A per-document `EXISTS` on `citations`: no
index on the column.

**What it opens.** Measured at 10,000 invented documents: the route 0.68 s, but the unfiltered
document list the pane loads is 6.5 s and 4.2 MB (`docs/performance.md` § 3 and § 6). The
registry's "last look" cannot tell a folder emptied of every file from an old scan (the spec's known
limit). Indexing that reads metadata is its own session.

## 2026-11-05 (3) — The library summary, gate first: signals read, wireframe drawn, spec written (ROADMAP 112)

**What changed.** No code. `docs/wireframes/library-summary.md` (+ `.excalidraw`): three screens,
the library's summary, a folder's, and the same view at 0 documents, as a third layout (*Summary*)
of the Library pane beside *Grid* and *List*; drawn at rev 1060, gate `fallbacks 0 · inferred 0 ·
unnamedScreens 0 · orphans 0`, waiting on the user's markup. `docs/specs/feature-library-summary.md`
(draft): which signals exist and how far each can be trusted, with file:line; one read-only route,
`GET /api/library/summary[?folder_id=]`, logic in `library/summary.py`; an `issue` filter on the
document list for *Show the N*; how each number behaves at 0 and ~10,000 documents.

**Why.** The user, 2026-11-05: a summary per library or folder of what went wrong at indexing and
facts about the documents. Read first, it turned out that the app's own indexing and *Add documents*
never read metadata, references, figures or keywords (only a catalogue import brings metadata), so
"missing" there means "not on record", and a document parsed to no references looks exactly like
one never parsed. The drawing says so in its footnotes instead of presenting either as a fact.

**Rejected.** Counting through `GET /api/sources`: it scans and writes registry rows, so the summary
reads the rows as of the folder's last look instead. A per-document `EXISTS` over `citations`: the
column has no index; one grouped pass instead. Ratios on the wire: a count of 0 cannot be divided.

**What it opens.** Two follow-ups the screen exposes: indexing that reads metadata (milliseconds per
document), and a record that a *Re-run…* part ran, so *found none* splits from *never looked*. The
canvas reading numbers the pane column by column under a full-height sidebar: the toolkit's T-006,
a sighting added to its ledger; the wireframe doc states the intended order.
