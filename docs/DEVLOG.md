<!-- status: active · updated: 2026-11-02 · class: append-only -->

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
> **2026-10-24 → 2026-10-31 (2)** in [`docs/archive/DEVLOG-archive-009.md`](archive/DEVLOG-archive-009.md)
> (rotated 2026-10-31: 11 entries, and 2026-11-02: 11 more) ·
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

## 2026-11-02 (2) — ADR-060 accepted in a grill: who bills is a fact of its own

**What changed.** ADR-060's status: accepted by the user. An amendment holds the grill's ledger of 11
forks. `docs/decisions.md`'s line follows. No code.

**Why.** Seven choices stand as recommended. The user asked whether more effort on *paid* was worth
it, and checking found that a local Ollama serves cloud models from ollama.com, marked by
`remote_host` in `/api/tags` (Ollama's docs, fetched 2026-11-02). So where the text goes is now
worked out per model, and *who bills* is a separate fact: a preset's vendor, nobody, Ollama for a
remote model, or a *does not bill me* tick on a server of your own. The guard trips on billed or not
known.

**Rejected.** Real prices per turn: they need a price catalogue and usage reports. `temperature` per
preset: one fact per vendor to keep current, for reviews that are unmeasured anyway. Editable preset
addresses. Exported vendor keys. A model list kept on disk.

**What it opens.** Piece 2's *stays on this computer* is wrong today for an Ollama cloud model; the
build fixes it. The build confirms OpenAI's address and Mistral's model-list path first, and checks
`remote_host` on a real cloud model, which needs an Ollama sign-in.

## 2026-11-02 — ADR-060 proposed: other answer providers through one OpenAI-compatible adapter (0.7.2, piece 3)

**What changed.** `docs/decisions/ADR-060-other-answer-providers-through-one-compatible-adapter.md`,
proposed, and its line in `docs/decisions.md`. No code.

**Why.** The user asked for more providers than Claude and Ollama, and the norm is an
OpenAI-compatible `/v1/chat/completions` with `/v1/models`. The ADR decides nine things: the client,
the presets, the route that takes a secret, where the text goes, what counts as paid, what is sent,
an unknown context window, model lists, and the tests at $0. Its worked example follows one question
through Groq and then through a server of your own.

**Rejected.** `langchain-openai`: its 1.7.3 needs `langchain-core>=1.7` against the locked 1.4.9, and
brings `openai`, which is on `httpx2`, and `tiktoken`, which fetches its files on first use (PyPI,
2026-11-02). A presets file: the build freezes `doc_assistant` as code only. Looking up a name to
decide where the text goes. Ollama through `/v1`: its docs say the context size cannot be set there,
which is KI-66 again.

**Found.** Eight places name the two providers, not five. `app_settings.set_llm_selection:107`
refuses any other name. `helpers._is_local` decides *no metered cost* from the name. The cost chip
prices every non-Ollama turn at Haiku's rates. `pipeline.build_chat_model` builds an Ollama model
for any name that is not `anthropic`. `docs/decisions.md` has no line for ADR-058 or ADR-059.

**What it opens.** The user's decision on the nine choices (`grill-me` if wanted), then one or two
build sessions. A paid call only comes last, at the user's word.

## 2026-11-01 (5) — Getting started is two rows; models live in Provider & model (0.7.2, piece 2)

**What changed.** New `GettingStarted.svelte`: Model and Documents, each a step from `/api/setup` with
a status and a link to its tab, and no form. `ProviderSetup.svelte` moved under Provider & model and
gained the *Answering with* card and a provider list grouped by where the text goes, with one form
open at a time. Save, verify and remove, the `.env` note, the Ollama probe and *Expect fewer
citations* all came across. The select and Apply are gone: each form has a free-text model box behind
*Use for answers*. `readiness.py` words the rows: `SetupStep` gains `status` and `offline`, and
`ProviderReadiness` gains `name`, `group` and `destination`, mirrored in `models/setup.py` and
`types/setup.ts`. `set_llm_selection`'s refusal now names Provider & model. ADR-034 amended; README,
QUICKSTART, setup, DEMO and `.env.example` send a reader to the right tab.

**Why.** The user: "Gettings Started vs Provider/Model is still confusing on initial setup". Both tabs
held provider setup and each pointed at the other.

**Decided while building.** *Engine online* for Claude means a key is in place, since ADR-034 does not
probe it; the amendment says so. Where the text goes comes from the address, so an Ollama on another
machine reads "go to" that host and is listed under *Your own server*, and `ANTHROPIC_BASE_URL` is
honoured. Only a chosen engine that does not answer is red: an Ollama with no models and a mismatched
model read *Not set* with the reason. The row shows the model id, as the status bar does: the drawing's
"Claude Haiku 4.5" is marked as an example, and a table of display names would go stale. The drawing's
line about every cloud provider having one form waits for those providers. `costLabel` and
`displaySteps` went with their only callers. The Model row's detail joined an alternative straight on
to a sentence with no full stop; fixed.

**Rejected.** A display-name mapper; a probe for Claude; warning rows or extra buttons on Getting
started; keeping the provider select as a second way in.

**Checked.** On new empty data homes, the repository's `.env` hidden and a local verifier, so no request
left the machine. `data_home` read through the page before each write. Driven: a fresh install from
*Finish setup…* with no key field on Getting started, a refused key, a saved key, removal, Ollama
chosen, a folder indexed, and a question answered with citations on `llama3.1:8b`; green; Ollama at a
dead port, red, the link opening its form. Light, dark and 375 px, no sideways scroll, 0 console errors
in the two clean final tabs. Real `data/` unchanged. **Mocked:** the `.env` note, by one in-page
response, because the check hid `.env`. **Not driven:** the packaged build; Ollama coming back under
Re-check; *Answering with* for an Ollama off this computer, which has tests.

**What it opens.** Piece 3 adds providers: `PROVIDERS` and `group` place a row with no change to the
list, and the ADR decides the form. `step.action` for Documents has no reader left in the app; it stays
on the wire because the API's contract and tests rest on it.

## 2026-11-01 (4) — Claude models from Opus 4.7 on answer and review

**What changed.** `llm.is_older_claude` tells the models before Opus 4.7 from the rest. On the rest,
`AnthropicClient.complete` sends no `temperature`, asks for `output_config.effort: "low"` and adds
`THINKING_ROOM` (4,096) to `max_tokens`; `build_chat_model` adds the same room to the chat's 1,024.
`llm.content_text` reads the text out of a string or a list of blocks of any shape, and the three
chat reads in `pipeline.py` (`rewrite`, `stream_answer`, `expand_query`) and
`_extract_anthropic_text` all use it. Haiku 4.5 is untouched: same request, temperature 0, 1,024.
Tests: `tests/unit/test_newer_claude_models.py`, the real `ChatAnthropic` over a fake stream of the
SDK's own event types; the three chat reads failed there before the fix. 2,983 tests pass.

**Why.** Asked whether the parked `temperature` patch could wait, the check found a gap it does not
cover. The patch keys on the API version; this keys on the model, and the API reference lists
`temperature` as a 400 under `2023-06-01` on Opus 4.7 and later, Sonnet 5 and later, Haiku 5.5 and
Fable. The model field is free text, and the reviewer follows it unless `REVIEWER_PROVIDER` is set
in the environment, which an installed copy has no `.env` to do. Tracing the fix found two more
faults behind the first. From Claude Opus 5 a reply opens with a thinking block: LangChain emits it
as list-shaped chunks and `controller.py:631` does `full_answer += tok`, a `TypeError`, so the chat
fails before the reviewer runs. `_extract_anthropic_text` read `content[0]` and returned the
response's repr. And thinking spends `max_tokens`, so a 400-token review could end with no text.

**Rejected.** The reverse list, naming the models that refuse: a new model would be a 400 until
someone added it, and sending no `temperature` is valid on every model, so an unknown name takes that
side. Trying with `temperature` and retrying on a 400: it depends on an error's wording and spends a
round trip on every call. Dropping `temperature` everywhere: it moves the reviewer and judge on Haiku
4.5 from 0 to 1.0, and their numbers stop being comparable. Turning thinking off: Opus 5.5 and Fable
refuse that, and Sonnet 5.5 needs a different value, so it would be a table per model.

**What it opens.** **Not checked against the API:** no paid call was made. The model list comes from
the API reference, and `THINKING_ROOM` is a judgement: if a model thinks past it, the answer is cut
at `max_tokens` and a review reports it could not parse. Chat runs at the model's default effort.
`figures.py` forces a tool, which the reference lists as a 400 on Opus 5.5, Sonnet 5.5 and Fable 5.1;
only `FIGURE_VLM_MODEL` reaches it, so it stays. The parked SDK 2.0 patch no longer applies to
`llm.py`; when that bump comes, its version test has to combine with `is_older_claude`.

## 2026-11-01 (3) — Wireframes approved: indexing, Getting started, Provider & model

**What changed.** `docs/wireframes/indexing.excalidraw` and
`docs/wireframes/getting-started-and-providers.excalidraw`, each with its reading in an `.md` beside
it. No code.

**Why.** The drawing comes before the code. The indexing screen lists every file with its status and
adds pause and stop. Getting started and Provider & model overlapped: both held provider setup, so a
first run had two places to look.

**Decided in the drawings.** Getting started is two rows, Model and Documents, each linking to its
tab; ready means both green, with the model, *Engine online* and the documents indexed. Provider &
model holds everything about answer models and says where the text goes. *Stop* has no confirmation,
because the next run skips what finished.

**Rejected.** Warning rows and extra buttons in Getting started: the user wants other options only if
they are needed. A confirmation on *Stop*: it loses nothing. One adapter per vendor: most providers now
document the same OpenAI-compatible interface, so presets are data over one adapter.

**Checked.** Indexing at rev 858 and Getting started and Provider & model at rev 176, each
`fallbacks 0 · inferred 0 · unnamedScreens 0 · orphans 0`, both approved by the user with no markup on
the canvas.

**What it opens.** An ADR for the adapter, extending ADR-011 and ADR-034. Per-file status, pause and
stop in the backend. Nothing here is built.

## 2026-11-01 (2) — Settings closes during a run (0.7.2, piece 1)

**What changed.** `Settings.svelte` and `Sources.svelte` drop their private `pollUntilDone` and read
the shared `ingestRun`. `busy` is derived in both: `indexLocked(requesting, status)` in
`core/ingest.ts`, true while a panel's own request is out or any run is in flight, so the folder
input, Browse and a second run stay guarded. Escape, the backdrop and the close button no longer
read it. `useRunEnd` in `core/ingest.svelte.ts` tells a mounted panel a run ended. Both panels call
`noteIngestStarted`, which now takes the run's folder. `initialSection` opens Documents during a
run. Opened mid-run, focus goes to the panel, because the input is disabled. The sentence now says
Settings can be closed. Settings reports `lostContact`, which used to come from its own poll. Tests:
`indexLocked` and the landing section, 331 in all.

**Why.** The user's report: once a run starts, Settings cannot be closed. `busy` covered all of
`indexFolder`, which awaited the poll. Freeing the three routes alone would have left three faults,
all gone by reading one run. A reopened panel held no run, so it showed an idle form that offered a
second run the API refuses with a 409. A run shorter than the 4 s idle poll ended unseen, because
Settings never told the watcher, and with the panel closed nothing refreshed the library. A first
run reopened on Getting started, which does not show it. `onCorpusChanged` no longer carries ingest
completion: App's `completedRuns` effect refreshes the library and chunk count with or without
Settings.

**Rejected.** Keeping both polls and adopting the run on mount: a second poller per panel, the
disagreement `core/ingest.svelte.ts` exists to prevent, and a run from the other panel would stay
unseen. A "started here" flag so Sources keeps its own lines: it forgets on reopen, and its result
would sit beside Settings' identical one. Sources sits directly under Settings' line, and its own
line was below the file list, often off screen, so it goes.

**Checked.** `npm run check` 0 errors; `npm test` 331 passed. Driven in the app in a browser preview
over a new data home and generated documents, four runs: 60 documents started with Enter, 240 with
a click on *Re-index*, one selected file closed at once, one started by a direct `POST /api/ingest`
with Settings left open. Escape, the backdrop and the close button each closed Settings mid-run at
1280 px in dark; Escape again in light, the close button again at 375 px. Escape works with focus on
the control that has just been disabled. The chip kept counting, and reopening showed the run on
Documents with focus on the panel. With Settings closed, the chunk count, the library and the setup
card refreshed. The outside run locked the open panel within one idle poll and ended in its result
line with fresh corpus numbers. No console error came from the app, and the real `data/` kept its
10,080 files at the same size and mtime. A new data home rather than a copy of the library because a
Re-index is a full ingest: `cleanup_orphans_chroma(also_clean_cache=True)` unlinks the cache paths
stored in Chroma, which on a copy name the real files. **Not driven:** the lost-contact line, Browse
(a browser does not render it), and the packaged build.

**What it opens.** Piece 3 replaces this line with the indexing screen, and `useRunEnd` is what it
needs for a result. A status poll already in flight when the POST lands can read the previous
`done` after `noteIngestStarted`, which fires a harmless extra refresh: the window is milliseconds,
the next poll corrects it, and the add sheet has the same gap. While Settings is open it covers the
chip on a wide window, so the count is only visible once it closes.

## 2026-11-01 — Letter-spaced text closes at extraction, with a record and a control (ADR-056, ROADMAP 108)

**What changed.** Built 2026-10-23 and 2026-10-24 in a side session and parked through the 0.7
release; this entry is the account of all of it, and of what landing changed.

*The rule.* `extractors.py`: `letter_spacing_joins` reads one PyMuPDF `rawdict` line for runs of
one-glyph tokens whose spaces the page does not show (the spaces PyMuPDF added inside letter-spaced
text, or space glyphs between letters that touch); `apply_letter_spacing` writes each run into the
page's markdown; `extract_pdf_pymupdf` runs both per page and skips a page whose glyphs cannot be
read. It leaves alone a page one image covers (`_SCANNED_PAGE_MIN`), any run in a formula, picture or
table box, and maths; a URL closes as kind `url`. *The ignore list.* `config.REJOIN_IGNORE_PATH`
(`rejoin_ignore.json` in the data home): per PDF, by sha256 with its size, the runs to keep, `*`, and
the user's `spacing`, applied only when it moves spaces alone. `document_fingerprint` adds a
document's entries to its cache fingerprint, so an edit re-extracts that document alone. *The
record.* `write_cache` writes `<cache>.md.spacing.json` between the markdown and the fingerprint,
each run with its state and its span in the text; a document without runs has no file. *The control.*
`library/spacing.py`, `/api/library/documents/{id}/spacing` (+ `keep`, `edit`, `remove`, `keep-all`),
`/api/library/spacing` (+ `remove`, `restore`), `?clip=` on the page route; `DocSpacing.svelte`,
`SpacingEditor.svelte`, `SpacingIgnoreList.svelte` and the pure `spacing.ts`; the re-run picker takes
parts to tick on open. *The CLI.* `scripts/list_letter_spacing.py`, read-only.

*Found by reading it again as new, and fixed, each with a test* (ADR-056, amendment 2026-11-01): a
bold or italic run never closed, because the pattern wanted whitespace where markdown writes `**` or
`_`; a kept or maths run recorded a form the text does not hold, so it had no span; the block did
not follow a finished re-run; `{"documents": null}` raised in the reader every freshness check
calls, and an entry of the wrong shape was a 500; a removed document left its record; `list_entries`
read every document's row and, like the CLI, missed a moved file; Settings could say "the 0
documents' text".

**Why.** ADR-056: a space the PDF does not hold, or holds with no visible width, is this project's
reading error, not the document's content, so it is fixed where it is made (as KI-64 was). The user
asked for points 7, 9 and 10 with the control from the sketch. Point 8 (keywords) waits for ADR-055,
and its span is in the record now because the record must be right before the library is
re-extracted.

**Checked.** The full suite, 2,937 passed; ruff, `mypy src` and CI's `mypy src/ apps/ --platform linux` clean; svelte-check 0 errors and 0 warnings, the 327 frontend tests pass. Gate 2 on the final rule, all 104 documents, against a copy of the live cache:
102 byte-identical, 2 changed by deleted spaces only (`cajal-lecture` 479, `s41598-025-33686-0` 306), no errors, as at every earlier run (those two compared with the live cache itself, the drive having re-extracted their copies). 36 runs recorded in 4 documents, 33 of them in the text, each at a distinct span that holds its text. The bold and italic fix moves nothing here: the same 3 runs still never reach the text.
Driven in the app on a scratch copy with every stored path pointed at it (`GET /api/settings` read
first, and a before/after listing of the real `data/`: unchanged): a document from before this
change shows no block; a real *Text & chunks* re-run wrote the text and the record, and the block
appeared without reopening the document; *Keep spaced*, *Edit spacing* and Save, *Show on page*,
*Keep every run spaced*, *All entries in Settings*, *Show runs*, *Remove all* and *Undo*; a second
re-run from *Re-run text now…*, which opens with *Text & chunks* ticked, wrote the user's spacing into
the text and kept the running head spaced, only spaces differing from the live cache. Dark, light and
375 px; no failed request. Earlier, in the side session: 48 joins reviewed by the user (33 places: 6
close, 27 not text, 0 keep spaced), which the four context rules reproduce on 31 of 33 decisions.

**Rejected.** The rule in a module of its own: the fingerprint walks `extractors` only, so a change
to it would re-extract nothing (KI-40). A set for the closing marks: its repr order follows the hash
seed, and the fingerprint hashes the repr. Skipping page headers: both letter-spaced runs of the
document that asked for this sit in header boxes. A threshold under that document's tracking for the
label-row gray zone: fitted to one library. `scrollIntoView` for *All entries in Settings*: it moves
every scrolling ancestor. An empty record file per document: files that say nothing.

**Not done, on purpose.** The count per document in `ingestion_events.notes` (ADR-056, point 3): the
store writes the event from the text alone, so the ingest log carries it. `data/rejoin_ignore.json`
is not written: this library's list starts empty. The library is not re-extracted: that is the
user's, with its backups, rebuilds and the private eval.

**Costs once.** The ignore entries go through `extract_to_markdown`, so every format's fingerprint
moves (checked on all six), and the first full ingest after this reads every document, about 24
minutes here. A document whose text comes out the same is not embedded again.

**What it opens.** ROADMAP 108: the re-extraction (gates 3 and 6), more letter-spaced PDFs (gate 4,
each download the user's call), and point 8 after ADR-055. Re-extracting before point 8 brings the
running head's keyword on the next keyword rebuild. The other document blocks (figures, references)
read their document once too, so they may also lag a finished re-run until it is reopened; not
checked here.

## 2026-10-31 (6) — The roadmap cleaned up; the order set for the next moves

**What changed.** `docs/ROADMAP.md` went from 90 rows to 86. Ten finished rows moved verbatim to
`docs/archive/ROADMAP-done-002.md`, 12 long rows were cut to what is left with their old text kept
there, and rows 101–106 were added. Row 65, the MCP server, is unparked and sits in F3. The table is
sorted by feature. Statuses were checked against the code: rows 34 and 86 closed, EX1's gate has been
closed since 2026-08-08, row 43 waits on KL3 only. `AGENTS.md` and `docs/OVERVIEW.md` §5 say 0.7.1 is
published.

**Why.** The user's order: the short-term work, then MCP and the chat, then extraction, the Library
and every instance of what the user chose. The file had also outgrown its own rule that a row's why
lives in its spec: one row ran to 6,145 characters.

**Rejected.** Release entries in the ROADMAP: sessions and release criteria stay in a local plan.
Scrubbing the archived rows of the private library's figures: an archive is verbatim (ADR-043).

**Opens.** ADR-055's parked patch takes row 107. KL2 and row 25 move behind both releases, unless
row 106's spec needs row 25.

---

## 2026-10-31 (5) — Preflight reads true after the tag

**What changed.** `check_checklists_refreshed` takes the version and leaves the release's own tag
out of `git describe`, with three tests. On this repository at `v0.7.1` it now compares against
`v0.7.0` and passes. `docs/RELEASE.md` §6 says so.

**Why.** Once the tag existed, the check compared the release with itself and failed, which is the
false alarm of v0.7.0.

**Rejected.** Keeping the runbook's "pre-tag only" rule: a check that lies after the tag is a trap
for whoever reads it there.

**Opens.** Nothing.

---

## 2026-10-31 (4) — Settings' Ollama card states its trade-off in words

**What changed.** *Expect fewer citations* in `ProviderSetup.svelte` no longer quotes the run's size
or its rates. It says local models carry fewer inline citations than Claude Haiku 4.5, and that this
predates 0.7.0's fix. The comment above it says why there are no figures.

**Why.** It was the one sentence the app shows that came from the private library. It stays in the
0.7.1 installer, and this change ships with the next build, by the user's choice over a 0.7.2.

**Rejected.** Deleting the warning: a local user still meets many *uncited* claims and needs the
reason where the engine is chosen.

**Opens.** A re-measure for KI-36 can no longer go on the card as figures from that library.

---

## 2026-10-31 (3) — 0.7.1 published; its notes say nothing about the private library

**What changed.** 0.7.1 is published, with the installer from `83c9af1`. The release notes and the
CHANGELOG's 0.7.0 section lost every figure, name and example taken from the private library. Each
limit stays, in words, and the notes' CHANGELOG link now reads `main`.

**Why.** The library is private, so a release says nothing about it. The clean-machine gate's three
arXiv papers are public and its numbers stay.

**Rejected.** Moving `v0.7.1` so its own CHANGELOG matches: a pushed tag is not moved
(`docs/RELEASE.md` §6), so that copy keeps the old text.

**Opens.** Settings' Ollama card still names the size of that library and its citation rates, in
the published installer too. Older CHANGELOG sections, the README and `docs/OVERVIEW.md` were left.

---
