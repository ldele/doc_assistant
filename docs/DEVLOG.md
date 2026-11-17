<!-- status: active · updated: 2026-11-16 · class: append-only -->

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
> **2026-11-05 (3) → 2026-11-13** in [`docs/archive/DEVLOG-archive-010.md`](archive/DEVLOG-archive-010.md)
> (rotated 2026-11-09: 11 entries, 2026-11-12: 11 more, and 2026-11-16: 11 more) ·
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
