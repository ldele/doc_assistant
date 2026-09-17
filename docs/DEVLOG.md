<!-- status: active · updated: 2026-09-17 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps the newest 20 entries** (`devlog_max_entries = 20` in `scripts/conventions.toml`,
> cpc rule 13b, user standard 2026-09-10; `tests/unit/test_doc_sizes.py` pins the same number).
> Rotate with `python tools/conventions/rungate.py rotate --root . --file devlog --write` (the shim —
> calling `tools/conventions/cpc/rotate.py` directly fails under `.venv`; corrected 2026-09-16) — it moves the
> oldest entries **verbatim** into the highest-numbered archive and verifies the bytes; then update
> the range below by hand (cpc ticket T-003). A day may be split across two files at the cut.
> Older entries, newest-first, unedited:
> **2026-08-12 (1) → 2026-09-01 (4)** in [`docs/archive/DEVLOG-archive-006.md`](archive/DEVLOG-archive-006.md)
> (rotated 2026-09-04, 2026-09-10, four times on 2026-09-16 and on 2026-09-17) ·
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

## 2026-09-17 (3) — Security S-3: answers are sanitised before they become HTML, and the dev loop gets a CSP — not the way the plan said

**What changed.** `Markdown.svelte` runs `marked` output through DOMPurify (new dependency
`dompurify` ^3.4) before `{@html}`, forbidding `style`, `form`, `input`, `button`, `textarea` and
`select`; citation buttons are still added afterwards, on the DOM. The Vite dev server sends a
Content-Security-Policy built from `tauri.conf.json`'s production string by
`src/lib/core/devCsp.ts`, which adds only `style-src 'unsafe-inline'` (Vite injects each stylesheet
as a style element), the HMR socket and Tauri's IPC endpoints — `script-src` is never loosened.
Tests: `devCsp.test.ts` (3 — every production value kept, script never loosened, only the three
additions); `test_desktop_security_config.py` +2 (the only `{@html}` sink renders DOMPurify output;
the dev header is derived from the production policy). Walkthrough §3 gains the hostile-markup row.

**Why.** S2: a passage quoted from the corpus carries its markup into the answer, and `marked` ≥ 14
passes `javascript:` links. **The plan's second half was wrong.** It said to set `devCsp`. Read in
the Tauri 2.11.3 source: on desktop `tauri dev` loads `devUrl` directly, and the policy (`devCsp`,
else `csp`) is injected only into assets Tauri serves itself (`manager/mod.rs` `csp()` →
`get_asset`; the dev proxy is mobile-only). `devCsp` would have changed nothing.

**Verified live, $0** (browser pane over the Vite dev server and the API): the real `Markdown`
component mounted with an `<img onerror>`, a `javascript:` link, a script, a form and a style block
rendered all five removed, kept the ordinary link and the `[1]` citation button, and left the
probe variable untouched; with DOMPurify bypassed, the dev policy alone blocked an inline `onerror`
(`script-src-attr`); the app raised no violation of its own across Chat, Graph and the taxonomy
modal, HMR connected, every `/api` call 200. **Not verified:** the Tauri window under `just app`
(IPC with the new header) and the installed build — both walkthrough rows.

**Rejected.** `devCsp` (a no-op on desktop, above). A `<meta http-equiv>` policy in `index.html` — it
would also apply in the built app on top of Tauri's header and block Tauri's nonce'd scripts.
Sanitising in a plain `.ts` module for `node:test` — DOMPurify needs a DOM, and jsdom for one call
breaks the frontend's zero-test-dependency rule. An `ALLOWED_TAGS` allow-list — markdown emits a
wide tag set, and a missed tag silently drops content.

**What it opens.** S-4 (`form-action`, `base-uri`, `object-src`) is next. S-11 can read
`DOMPurify.removed` for its `security_markup_sanitised` event.

---

## 2026-09-17 (2) — Row 53: a merge keeps what you curated and can be undone; the preview and the merge share one definition — the only harmless one measured

**What changed.** New `knowledge/concept_merge.py` is the merge's write seam (`apply_merges`,
`undo_merge`, `list_merges`), replacing fold-and-delete. A merge refuses ids that are not both
concepts, and a plan whose placements would close a taxonomy cycle — decided in memory before the
first write — then moves to the survivor: surface forms; definition and graph membership when it
lacks them; every `concept_hierarchy` placement, re-pointed through `add_hierarchy_edge` (an edge
between the two is dropped, not made a self-edge); gap triage (the survivor's own verdict wins); and
stochastic gap suggestions. It deletes the row and writes a `ConceptMerge` record (new additive
table `concept_merges`) holding what `undo_merge` needs. `apply_plan` returns `(n_demoted,
MergeOutcome)`; `curate_concepts` reports refused merges and the largest merge group, and gains
`--merges` and `--undo-merge ID [--apply]`. **One definition:** `concept_curation.dedup_pairs`
(keyed by id; `merge_text` = label + definition) serves both `suggest_concepts --near` and `--dedup`,
defaulting to new `CONCEPT_MERGE_MODEL` bge-base and `CONCEPT_MERGE_COSINE` 0.85 → **0.90**; the
preview also stops listing artifact labels. Tests: `test_concept_merges.py` (9, real SQLite with
foreign keys on; non-vacuous — patching `_repoint` to move nothing fails two of them),
`test_the_merge_preview_is_the_merge`, the `apply_plan` tests.

**Why.** REVIEW 2026-09-16 C-1/C-2: `concept_hierarchy` cascades on delete, so a merge destroyed the
dropped concept's placements, orphaned its triage and left no record — the one curation path that
deletes, against ADR-028 and ADR-018. Row 51 writes curated `is_a` edges next, so this came first.

**Measured before choosing the definition** ($0, read-only;
`tests/eval/baselines/concept_merge_cosine_2026-09-17.md`). My first cut unified on the preview's
settings. On the 357 real labels SPECTER2's median pair scores 0.842: at 0.85 the merge would delete
354 concepts into one group, at 0.90 309. bge-base at the old merge's 0.90 merges none; at 0.85, 34
pairs — 7 duplicates, 21 narrower terms, 6 related, by my first reading. So the shared definition
is the old merge's: `--apply` behaves as before, the preview is finally honest, and no threshold is
claimed.

**Rejected.** A savepoint per merge — pysqlite's SAVEPOINT handling needs a driver workaround the
session lacks. Demoting the dropped concept behind a `merged_into` column — every reader of
`concepts` would need a new filter. Keeping SPECTER2 because `config.py` says bge "compresses
same-domain concepts" — on labels the reverse holds. Settling on 0.85 or 0.87 for bge — the pairs say
a threshold is the wrong tool.

**What it opens.** Row 53 (3): the user's hand score of the 34 pairs, and the design it points at —
a surface-form fold for real duplicates, per-pair review for the rest. The 21 narrower pairs are
`is_a` candidates for row 51. Nothing was merged on the live library.

---

## 2026-09-17 (1) — Row 54: no write can put a field node on the graph, and every coverage number counts what it describes

**What changed.** `taxonomy.py` gains `presence_query()` (the kind guard as a query to narrow) and
`presence_node(session, id)` (its write side). Routed through them: `concept_skeleton.load_concepts`
— the graph's own vocabulary read, which filtered on `graph_include` alone — `set_graph_include`,
`delete_concept`, `rename_concept`, the label get-or-create in `add_concept` and `promote_keyword`,
`load_glossary`, `list_keyword_candidates`; the family read and all five family writes by id in
`library/keywords.py`; `concept_curation.load_concepts` and `rank_keyword_candidates`.
**Denominators:** graph coverage counts cited ∩ shown documents over the non-archived count
`library.count_documents` reports; the coverage sentence names the skeleton's concept count, not the
vocabulary's; the taxonomy header counts the graph vocabulary, and its "not yet placed" is exactly
`unplaced_concepts()` — auto-propose's input — shown as "N graph concepts". Tests: 3 kind-guard tests
(refused by `set_graph_include` and absent from `load_concepts` even when already flagged; the five
family writes refuse a field and its placement survives a delete; `add_concept` never adopts a field
with the same label) and 2 parity tests asserting against the functions that act
(`count_documents`, `unplaced_concepts`), not a fixture; the staleness test that pinned the numerator
bug (expected 2) now expects 1.

**Why.** REVIEW 2026-09-16 C-4/C-5/C-6. ADR-028 D4's guard lived on the reads that remembered it; a
family delete by a field's id would have cascaded every placement under that field. Graph coverage
counted deleted documents; the taxonomy header said "344 not yet placed" (357 families) while
auto-propose had nothing left to place.

**Verified live, $0.** `/api/taxonomy`: 13 concepts · 98 documents · 0 unplaced (was 357 · 344); the
header renders "13 graph concepts · 98 documents · 236 fields"; the graph still reads "Covers 30 of
your 98 documents" (no document on this library is deleted or archived).

**Rejected.** A database constraint on `graph_include` — a migration for a rule the write seam holds,
and it would not cover renames or deletes. Keeping `n_concepts_total` at every family and fixing only
"not yet placed" — "357 concepts · 0 not yet placed" reads as all placed. Raising instead of returning
None/False — each function keeps its existing contract.

**What it opens.** `taxonomy_view` still filters kinds on networkx attributes (a graph read, not a
query — left). `unclassified_documents` still includes archived documents (propose input, harmless).

---

## 2026-09-16 (5) — KL1: the knowledge layer says what its signals are — thin bridges get a real definition, and "unsourced claims" turn out to be about your answers

**What changed.** Sequence row 2's main work, from a read of the code and the live library before
any edit. **A1 (wording):** the opt-in stance chips read `contested? (experimental)` /
`superseded? (experimental)` in the source card and in the controller's Sources block (was
"contested in corpus"); the Settings sandbox toggle says "not a corpus measurement" and falls back to
`false` like the backend; `unsourced_claim` is shown as **"Uncited in answers"**; GLOSSARY C-007,
`how-answers-work.md`, README and the `ConceptGraphEdgePayload` docstring (which claimed `relation`
was empty on every edge — 19 of 20 carry it) stop implying a measurement. **A4:** most of RG-014's
grades were already encoded (single_source leads, under_connected hidden). What remained:
`detect_thin_bridges` now counts a bridge only when both sides keep ≥ 2 concepts and flags the
smaller side's endpoint (both on a tie), computed in one linear pass over the tree of 2-edge-
connected blocks; and the gap list's "N open" counts only rows it can show. **A3:**
`synthesis.is_claim_unit` rejects pieces that assert nothing — a markdown heading, a sources/references
block, a piece that is empty or ends in `:` once a trailing list enumerator is removed — applied in
`detect_unsourced_claims` only. **A5:** `knowledge-layer.md` §6 re-read: thin_bridge structural,
unsourced_claim about answers, new rows for coverage, stored-gap staleness, placement, merges; §6b
notes why ADR-041 option 6 is not buildable as written. **A2 → new row KL1b** (decision first).
**New rows 90** (the unreadable "According to Source 6: file.pdf" citation form) **and 91** (say when
stored gaps predate the graph). Tests: 5 thin-bridge tests replace the one that pinned both-ends
flagging; `is_claim_unit` (3) and the gap floor (1); two `test_chat_controller` strings.

**Why.** PLAN 2026-08-03 Phase A: a lying surface is worse than an absent one. **The finding that
reshaped it:** the "claim layer" A2 and `unsourced_claim` both rest on is the assistant's own answer
sentences — 1,911 `AnswerClaim` rows, uncited 72% of the time on `llama3.1:8b` against 21% on Haiku —
not claims the corpus makes. So "Uncited in answers" is the honest label, and option 6 would measure
query history × model.

**Measured, read-only, $0.** Of 1,239 `unsupported` pieces, `is_claim_unit` rejects 306 (230 bare
enumerators, 63 colon lead-ins, 9 headings, 4 Sources blocks); on the concept-attributed links the
gap floor actually counts, 153 → 139 over the same 8 concepts — headings were never the main
contamination on this box, the unreadable citation form is (row 90). A dry `build_gaps` on today's
skeleton: isolated 3 · single_source 3 · under_connected 3 · unsourced_claim 8 · **thin_bridge 0**
(the 4 stored rows were all dead-end edges). The stored rows are stale anyway: 16 of 18 came from
graph `c97c…`, two versions before the current `027b…` (row 91).

**Rejected.** Filtering inside `segment_claims` — renumbers claims on every new answer and moves the
RG-012 citation markers. Word-count or phrase-list rules for "not a claim" (the audit's heuristic
had `< 6 words`, "Here is") — tuned to one model's habits. Flagging the lower-*degree* endpoint of a
bridge — degree is the vocabulary-sparsity signal RG-014 graded noise; side size is structural.
Relabelling the chips per ADR-040 option 4 ("mixed stance") — ADR-040 blocks the surfacing choice
behind the Node-B rebuild; this only removes the false "in corpus". Building A2 — see KL1b. Running
`build_gaps --apply` — a DB write, the user's call.

**What it opens.** The gap rebuild ($0, `python -m scripts.build_gaps --apply`) to make the stored
rows match. KL1b's decision. Row 90 before any "Uncited in answers" count is quoted. The colon rule
drops ~63 lead-ins, a few of which make a weak assertion ("The authors propose two main types:");
"Liu et al." fragments from abbreviation splits still count.

---

## 2026-09-16 (4) — Security S-2: one add checks at most 50,000 files, and the uploader states the limits it enforces

**What changed.** `library.add.expand_paths` counts while it walks and raises
`AddBatchTooLargeError` past `config.MAX_ADD_FILES` (50,000, `DOC_MAX_ADD_FILES`) — before sorting, so
a pick of a whole drive costs the walk up to the limit; `apply_add` refuses a longer explicit path
list before touching anything; both routes return a 400 whose `detail` is the sentence, which the
review sheet already shows as its alert. `GET /api/documents/accepts` serves `extensions`,
`max_file_bytes`, `max_archive_bytes`, `max_files_per_add` from `library.add.accepted_input()`.
The frontend loads it once when the API is up (`accept.svelte.ts` `loadAccepts`) and the three
places that listed formats — the Add documents dialog, the empty-library card, the drop overlay —
now show the served formats line and **"Up to 1 GB per file"** (`lib/library/formats.ts`, 5 tests).
Settings' formats line derives from the same registry. README Limitations, `docs/usage.md`,
`.env.example` (three knobs), `security.md` (S-2 done, S-3 next), ROADMAP row 60, the walkthrough
§1 (stated-limit and walk-cap rows), `feature-add-documents.md` constraint 6 amended, CHANGELOG.

**Why.** `docs/security.md` S3: `inspect` walked a picked folder with no cap on a request thread.
And the user asked that the 1 GB limit S-1 introduced be stated in the uploader and the docs.

**Choices.** N = 50,000: 5× the 10,000-document contract, since real folders hold files that are not
documents and Zotero sends one path per attachment through the same route; ~9 s at the measured
~11,000 files/s. **No byte cap (M)**: `inspect` barely reads contents and S-1 caps each file. The
numbers come from the API, never from a frontend constant, because all three limits are env knobs.
No env-var name in the UI — the docs name them.

**Verified.** `test_library_add.py` +6, `test_api_documents_inspect.py` +3 (400 names the cap; `/add`
refuses; `/accepts` follows a monkeypatched cap). Live preview: `/api/documents/accepts` returns
1073741824 / 50000 / nine extensions; the app requests it once per load; the state module holds it.
**The live check caught a bug:** the empty-library line lower-cased the whole limit sentence and read
"up to 1 gb per file" — fixed with `limitPhrase`, pinned by a test. **Not verified:** the rendered
lines themselves — all three surfaces exist only in the desktop window (`canAccept()` is false in a
browser); the walkthrough row covers them.

**Rejected.** Hard-coding "1 GB" in the frontend (wrong the moment the env knob is set). Adding the
limit to `/api/setup` (a readiness endpoint). An object-shaped 400 (`errorDetail` would
`JSON.stringify` it). A folder-depth limit (the grill chose full recursion, branch 3).

**What it opens.** `docs/QUICKSTART.md` §3 still describes adding documents through Settings (row
81). S-3 is next.

---

## 2026-09-16 (3) — Security S-1: oversized files and zip bombs are refused before they are opened, without re-extracting the corpus

**What changed.** `extractors.ingest_refusal(path)` returns a sentence, or `None`: a file over
`MAX_INGEST_FILE_BYTES` (1 GB, `DOC_MAX_INGEST_BYTES`); for EPUB/DOCX/ODT, an archive that cannot be
read, that declares more than `MAX_ARCHIVE_EXPANDED_BYTES` expanded (1 GB, `DOC_MAX_ARCHIVE_BYTES`),
or that holds an entry at ≥ `MAX_ARCHIVE_ENTRY_RATIO` (1000:1). It reads the size and the central
directory only. Two callers: `get_format_status`, so the add review sheet shows the sentence
verbatim as an `unsupported` row; and `ingest.cache.load_or_extract`, which raises
`IngestRefusedError` (a `ValueError`) just before extraction, so a file that reaches ingest without
the sheet is refused per file through the existing `document_error` path. `tests/unit/
test_ingest_size_caps.py` (12): a file over the cap and a synthetic zip bomb are each refused with a
sentence in the review sheet; an archive over its expanded cap; a damaged archive; real DOCX, ODT
and EPUB are not refused; extraction never runs on a bomb; the check is outside every format's
fingerprint closure; and CPython's `zipfile` truncating an entry at its declared size. Walkthrough §1
gains the row, `security.md` S-1 → done and S-2 next, ROADMAP row 60 → S-2, CHANGELOG `[Unreleased]`.

**Why.** `docs/security.md` S4: EPUB, DOCX and ODT are zip archives opened with no decompressed-size
accounting, so one file in the corpus could exhaust memory — the cheapest local denial of service.

**Two facts that shaped it.** (1) **Placement was forced by KI-48.** The extraction fingerprint
hashes everything reachable from `extract_to_markdown`; a check inside it would have marked all 98
cached documents stale and re-extracted the corpus. The fingerprints of all eight formats were
recorded before and after the change and are identical, and a test now asserts `ingest_refusal`
stays out of the closure. (2) **The declared sizes can be trusted as a bound.** Read in the CPython
source, `ZipExtFile._read1` ends `data = data[:self._left]`: output stops at the declared size
whatever the stream holds, and ebooklib, python-docx and odfpy all read through `zipfile`. So an
archive that lies about its sizes is truncated and fails its CRC — pinned by a test that forges a
central-directory size. The ratio check therefore catches the bomb *signature*; the expanded-total
check is what bounds memory.

**The numbers are structural, not tuned.** 1 GB sits ~33× above the largest document in this library
(31 MiB, 98 PDFs); deflate cannot compress real data past ~1032:1 — only a constant stream reaches
it — so 1000:1 is not a corpus constant. Both byte caps are env knobs; the ratio is not.

**Rejected.** The check inside `extract_to_markdown` (a corpus-wide re-extraction, KI-48). A new
`refused` verdict for the sheet — the frontend already renders the advisory verbatim beside a
warning icon with no fixed label, so a new verdict would be a wire and UI change for no visible
difference. Streaming decompression accounting — unnecessary once `zipfile`'s truncation is pinned.
A page cap — PyMuPDF opens a large PDF lazily and page count is not the memory risk the finding
names; noted as residual in `security.md` S4.

**What it opens.** Residual in S4: a central directory with millions of entries is bounded only by
the 1 GB file cap. S-11 (security events in the log) will give this control its own event; today a
refusal is visible only as the sentence and as `document_error`. Next security step: S-2, the walk
cap on `inspect`.

---

## 2026-09-16 (2) — Row 46: the ship gate asks three questions, reads its verdict with the app's own parser, and moves into the repo

**What changed.** **The harness is tracked** at `scripts/rg012/` — `rg012-run.ps1` and
`rg012-tier2.wsb`, which now maps that repo folder into the sandbox (it ran an untracked copy under
`C:\rg012-host\script`, invisible to diff review and to tests; the host `.wsb` was replaced and the
original kept beside it as `.bak-20260916-single-turn`). **Three turns:** one question per corpus
document, each in its own session so no answer is shaped by another's history; one
`out\run-<stamp>\` per run holding the log and `turn-N-stream.txt` / `turn-N-result.json`.
**Two verdicts:** `release_preflight`'s `rg012` check became `rg012_packaging` (Python not on PATH,
ingest produced chunks, every turn answered) and `rg012_citation` (≥1 turn cited, none *tried and
failed*, at least `RG012_MIN_TURNS` = 3 turns), and the citation half is judged on the host with
`synthesis.audit_citations` over the saved answers — each turn named *cited / unresolved / uncited /
missing*. The script's own lines are now labelled `(estimate)`. **The newest run on the artifact
decides**, and earlier runs on it are printed with their verdicts. **A reader bug fixed on the way:**
the harness appended to one log, the 2026-08-15 archive holds two runs, and the preflight read the
first installer line and *any* `TIER-2: PASS` in the file — a failed re-run after a pass would have
read as a pass. It now judges the last run in a log. Tests: 15 new in `test_release_preflight.py` (17 cases), and a
new `test_rg012_harness.py` pinning ASCII-only, the log lines and file names the script writes
against the ones the preflight reads, ≥3 distinct questions, one session per turn, and that the
`.wsb` launches the tracked script. `docs/RELEASE.md` §5 and `docs/desktop-packaging.md` §5 say how
to run it and how to read it.

**Why.** RIGOR_TODO RG-012 (2026-08-14): the byte-identical 0.5.1 installer failed its single cited
turn once in four runs, because `llama3.1:8b` cites all-or-nothing per answer (KI-36). A blocks-ship
gate that fails ~1 run in 4 on a healthy artifact trains its operator to re-run until green. Option
2 ($0, local) over option 1 (a paid Haiku turn) per the roadmap row.

**Verified without a sandbox run.** (1) `audit_citations` over the ten archived runs reproduces every
recorded verdict, including the two FAIL kinds: 08-06 run 2's `[Source 1: file.pdf]` → *unresolved*,
08-14 run 1 → *uncited*. (2) The tracked script is 0 non-ASCII bytes and 0 parse errors under
Windows PowerShell 5.1.26100. (3) Sections 5–6 of the tracked script were run verbatim on the host
with `Invoke-WebRequest` stubbed to replay archived streams — cited/uncited/unresolved gave
packaging PASS + citation FAIL; a timed-out turn gave packaging FAIL + "could not judge" — and the
preflight read those PowerShell-written, BOM-carrying files. That run also caught the estimate
claiming a citation PASS with a turn missing; it now says `NOT JUDGED`. (4) The live preflight on
this box reads the 0.6.0 run as `rg012_packaging PASS` / `rg012_citation FAIL — only 1 turn`, which
is the honest reading of that record.

**Rejected.** Keeping the verdict in PowerShell with a loop around the old regex — it leaves the
KI-35 shape (a restated contract) in place and untestable. Option 1, a paid turn — makes the ship
gate a billed path (KI-4). Accepting single-turn runs as a pass — a stale copy of the harness would
quietly bring the coin flip back. Letting any PASS on the artifact win, as before — that *is*
re-running until green. Keeping the harness local-only — the gate that decides a release was the one
piece of release tooling no review could see.

**What it opens.** Row 46 closes with **one sandbox run** of the new harness on the 0.6.0 installer:
only that shows three turns completing inside the timeouts on a clean box (needs Ollama reachable
beyond loopback — a host change, the user's call). RG-012 itself closes when two consecutive release
gates agree. `rg012-diag.ps1` / `rg012-ingest.ps1` stay untracked in `C:\rg012-host\script\`
(August diagnostics, not part of the gate).

---

## 2026-09-16 — After the break: the plan is made true again, fourteen dropped follow-ups become rows, and the crossover's merge row turns out to aim at the wrong knob

**What changed.** A docs pass after a post-release break; no code. **ROADMAP:** rows 35 · 37 · 47 ·
73 · 74 cited `docs/ui-checklist.md` §2/§3, sections that left that file on 2026-09-10 — repointed
at the rows they came from in the frozen `.claude/ui-checklist-archive-001.md`. Rows **53 · 54 ·
51 re-scoped** after the 2026-09-07 crossover review was checked against the code and the live
library: `CONCEPT_MERGE_COSINE` only feeds the read-only `suggest_concepts --near`, while the merge
that deletes rows (`curate_concepts --dedup --apply`) uses a hard-coded 0.9, another embedder and
another input; that merge deletes the dropped concept and `concept_hierarchy`'s cascade takes its
curated placements with it; `set_graph_include` accepts a field node; graph coverage counts deleted
documents; and nothing can write an `is_a` edge except a raw API call. So 53 is now *merges made
safe, then one threshold measured*, 54 adds the kind guard, 51 needs an `is_a` write path; the
sequence order is unchanged. KL1 also refreshes `knowledge-layer.md` §6 (presence is 534 chunk
keys, not 1,781). **Fourteen new rows (76–89)** for follow-ups that had dropped out of every plan
between the release and the restructure — the user's own 2026-08-24 asks (range selection in chat
select mode, draggable dialogs), the Windows CI job + shared `empty_library` fixture KI-58 said row
60 carried and it did not, parallel extraction never verified in the installed build, the README
GIF / DEMO refresh, the process-scoped `OLLAMA_HOST` route, the stale frozen sidecar that `tauri dev`
spawns, the 2026-09-10 review's hygiene moves, and three user decisions (`docs/sprints/`, the
encoding guard for local docs, an unfilled `.claude/NORTH_STAR.md`). Rows 29 and 75 absorbed two
small UI loose ends. **KI-48** closed — fixed 2026-08-25 (per-format fingerprint) but headed OPEN
for three weeks; body moved verbatim to `docs/archive/KNOWN_ISSUES-resolved-002.md`. **CHANGELOG:**
`## [Unreleased]` added; 0.6.0 dated 2026-09-04, the tag date, like every earlier release (it said
the build date). **Conventions:** `scripts/conventions.toml` no longer claims the cpc 1.2.3 key set
as the vendored version (the drop is 1.8.0, re-vendored 2026-08-22 with no record here); the rotate
command in this header and in that file is the shim form. `docs/local-only.md` lists four gitignored
paths the docs cite. **cpc:** tickets T-013 (generate/keypoint extras spawn without the gate env),
T-014 (keypoints route to a skill the model cannot invoke), T-015 (a re-vendor leaves no trace in
the consumer), T-016 (lift `release_preflight` — the user's 2026-09-02 ask); T-011/T-012's version
corrected before their first commit. All uncommitted in the cpc checkout.

**Why.** The user asked what was pending after the break and whether the crossover review held
anything for the concept branch. Three independent read-only audits plus a re-read at source of
every claim that decides a row; the full record, with file:line, is
`docs/reviews/REVIEW_2026-09-16_post-break-state.md` (local).

**Rejected.** Sweeping `CONCEPT_MERGE_COSINE` as row 53 said — it would have measured a preview no
merge reads. Re-ordering the user's sequence to put merge safety first — the destructive path is a
manual `--apply` nobody has run, so it only has to land before 51 writes curated edges, which the
existing order already guarantees. Fixing the orphans inline (the OLLAMA route, the EPUB parser, the
sidecar comment) — the user asked for rows, and a docs pass that also changes code is two commits
pretending to be one. Re-vendoring cpc — its HEAD carries unreleased fixes and would stamp 1.9.1.
Pruning `build/` `dist/` and the backups — deletes data; row 88 is the user's call.

**What it opens.** Session 1 of the sequence is unchanged (row 46 + S-1). Row 53 may take two
sessions; the sequence row says what moves if it does. The ~5 GB prune, the Docker upgrade and the
three decisions in row 89 wait on the user.

---

## 2026-09-10 (2) — Working docs get folders, the DEVLOG keeps twenty, the release checklists are stale by default, and security becomes one step per session

**What changed.** Second session of the day, on the user's reading of the review. **Folders
(ADR-051):** the five live plans moved to `docs/plans/`, the two reviews to `docs/reviews/`; each
folder has a tracked README; the dated files stay local-only by *pattern* (a folder-level ignore
would have taken the README with it); fifteen living files repointed (ADRs 028/030–033/046, the
add-documents spec, ROADMAP, ui-checklist, knowledge-layer, local-only, README, the three local
trackers) — append-only logs and the CHANGELOG keep their old paths as history. **Entry budgets:**
`devlog_max_entries = 20` in `scripts/conventions.toml`, `cpc-rotate --file devlog --write` moved
56 entries (2026-08-15 → 2026-08-30 (8)) into archive 006 byte-verified, and
`tests/unit/test_doc_sizes.py` pins the same number so a clone or CI sees it. **Release:** the two
checklists are assumed stale — `docs/RELEASE.md` §0 is now the first step (list what shipped, add
rows, bump), and `release_preflight` gained `check_checklists_refreshed` (both files must carry a
commit or an uncommitted edit since the previous tag; six tests, a pinned file list). **Security:**
`docs/security.md` restructured — §0 how the file is worked, §4 an ordered plan S-1 … S-12 sized
one step per session with a `done when` per step, §6 the periodic full check (eleven lenses, each
with its command) whose log is `.claude/REVIEWS.md` row 7; S-12 (Dependabot + SHA pins) marked as
the user's call. **Roadmap:** re-sequenced to the user's order — gaps first with one security step
per session (row 60 is now the standing "current step" row), the knowledge layer made honest
(KL1 → 53 + 54 → 51 → KL4 → KL2), then row 25, then a **graph re-pass** (new row 75: 52 · 57 ·
50 · 43 · a UX pass); a twelve-session sequence table; row 62 done → archive. **cpc:** tickets
T-011 (the folders) and T-012 (the four consumer-grown conventions) opened with `cpc-ticket` in
the cpc checkout — uncommitted there, for the next cpc session.

**Why.** Each is a user decision from the second session: folders for plans and reviews as the
standard going forward; twenty entries as the DEVLOG standard; checklists that "would never
otherwise be up to date"; security worked a little at a time with a thorough check at the end and
a standard place to log it; and the order gaps → knowledge → ingestion → graph.

**Rejected.** Ignoring the two folders wholesale (kills the tracked README); judging checklist
freshness by the `updated:` header (bumpable without a refresh — git history, like
`artifact_fresh`); hand-cutting the DEVLOG on a day boundary (the tool rotates by count and
verifies bytes; the split day is recorded in both headers); a separate security log file (the
REVIEWS ledger already has the shape and the "not covered" discipline); editing cpc's CONVENTIONS
directly (a ticket is the channel the ledger asks for).

**What it opens.** The DEVLOG header's archive range is still updated by hand (cpc T-003). Session
1 of the sequence is row 46 + security step S-1 (ingest size caps). `docs/sprints/` is laid by cpc
and unused — the T-011 answer may settle it. Stamps: the first draft of this session said
2026-09-11 in fourteen files; corrected to 2026-09-10 before staging.

---

## 2026-09-10 — Whole-project review: the roadmap goes per feature, the security floor is written down, and a ninth version file turns up at 0.4.2

**What changed.** A review session, not a feature session. Three read-only audits (repo health,
security surface, file architecture) and a read of every coordination doc produced
`docs/REVIEW_2026-09-10_project-review.md` (local, ADR-029) and these tracked changes:
`docs/ROADMAP.md` restructured into one table of **open** rows with a `Feature` column and twelve
feature sections — the 87 done rows moved **verbatim** to `docs/archive/ROADMAP-done-001.md`; the
UI checklist's queue and verification debt folded into it (`docs/ui-checklist.md` keeps only the
per-feature gate); three closed working docs archived under the new, gitignored
`docs/archive/local/`; `docs/security.md` (threat model · what holds · eleven ranked findings each
with a roadmap row · the deterministic floor · a cpc gate proposal); `docs/release-ux-checklist.md`
(seven surfaces to drive in the installed build, with a record table whose "not driven" column is
mandatory) wired into `docs/RELEASE.md` §5b and the gates table; `docs/architecture.md` corrected
(the taxonomy layer is *built*, ~2,600 lines; the keyword arm is the on-disk FTS5 index, not the
retired in-RAM BM25; the legacy `tests/eval/run_eval.py` dropped from the layout).
Code and config: the lint/type/SAST gates now cover `apps/` and `scripts/` in CI and pre-commit
(they were `src/`-only — the HTTP boundary was gated by nothing; two fixes for mypy: the
`sse_starlette` import moved to the package root, and `routers/updates.py` names its `UpdateState`);
`[tool.bandit]` exists so `-c pyproject.toml` means something; `npm audit --audit-level=high` is a
frontend CI step and its one high (`nanoid` < 3.3.18) is fixed in the lock;
`tests/unit/test_desktop_security_config.py` pins the Tauri CSP (set, never unsafe, network only to
the sidecar) and the exact capability set; `docker-compose.yml` binds the port to loopback;
`scripts/conventions.toml` registers `gh run list --branch main --limit 1` in the session-start floor
and the security floor + walkthrough as sprint-close checklist lines; the dead `langchain.*` mypy
override is gone; `docs/features/` (a template nobody used, not laid by cpc) is removed. Local:
`.claude/CONTEXT.md` 316 → 245 lines (the July narrative and the resolved questions moved verbatim),
`.claude/KNOWN_ISSUES.md` 1,726 → 867 (18 closed bodies → archive 002, index rows auto-derived from
headings — refine when next touched; **KI-58 filed**: the CI-blind shape), `.claude/RIGOR_TODO.md`
history block archived and its "gate fails" line corrected (no gate exists).

**Why.** The planning surfaces disagreed (the ROADMAP said row 24 next, the UI checklist said P2;
87 of 108 rows were done and restated the DEVLOG); the phase framing had stopped describing the
work; security had never been written down as a threat model, so every finding would have been
"critical" or ignored; and the session-start protocol never read CI, which is how `main` stayed red
for seven pushes.

**Found on the way, and the one that matters:** `apps/desktop/package-lock.json` records the
project version twice and npm rewrites it only when npm runs — it sat at **0.4.2 through v0.5.0,
v0.5.1 and v0.6.0**. Same shape as the Cargo files: a version-carrying file nothing was looking at.
`release_preflight.collect_versions` reads both fields now, the pinned-file-list test carries them,
and `docs/RELEASE.md` §1 says nine places, not eight.

**Rejected.** Making `pip-audit` blocking today (66 advisories; it would redden every run until
row 61 triages them — the CI step stays advisory and the `sprint-close` checklist names it);
adding Dependabot and SHA-pinning the actions (bot PRs are the user's call); DOMPurify, the CSP
`form-action`/`base-uri`/`object-src` additions, `TrustedHostMiddleware` and the prompt fence
(each a code change that needs the installed build or an eval run to verify — rows 60/61, not a
review session); rotating the DEVLOG now (3,905 of 4,000 lines — due at the next close);
hand-writing the 18 KI index rows (verbatim archive first, judgment when each is next touched).

**What it opens.** Row 46 before the next release (the citation gate is a coin flip); a Windows CI
job (KI-58's other half — the shipped platform is the one no job runs on); the `pins.py` reachability
question; the 22 headerless specs; `docs/sprints/` is laid by cpc and unused since 07-18 — decide.
The review's §11 lists what it did not cover; rows 2 and 3 of `REVIEWS.md` stay `never`.

---

## 2026-09-07 (3) — CI on `main` was red for five days and seven pushes, including the release; two test-only fixes

**What changed.** `tests/unit/test_source_view.py`: the two "unknown document" tests get an
`empty_library` fixture — a temp SQLite with `Base.metadata.create_all` swapped into
`db.session._engine/_SessionLocal` via `monkeypatch`. `tests/integration/test_external_metadata.py::
test_matching_survives_a_differently_written_path`: the "differently written" variant is now
`parent / "sub" / ".." / name` (upper-cased on Windows) instead of a `/`→`\` swap. Both files,
+41/−4, test-only; shipped code untouched.

**Why.** `gh run list --branch main` shows every run since `3230703` (2026-09-02) red at `pytest
with coverage`, last green `5405d44` (2026-08-28) — through the merge, the tag and the docs commit
this morning — while the suite was 2,389/0 on the Windows dev box. Two causes. (1) The source-view
tests read the real `session_scope()`: locally `data/library.db` has the schema so the lookup
misses and passes; on CI `SQLITE_URL` names a file SQLAlchemy creates empty on first connect, so
`no such table: documents`. Reproduced here with `DOC_DATA_DIR=<empty dir>` on the old file (both
fail with exactly that error) and the fixed file (24/24). (2) `pathkey` is
`normcase(abspath(...))`. The old variant, on POSIX, produced a relative filename with literal
backslashes — a different file, so the catalogue entry never matched, red since the test was
written on 2026-08-31; on Windows `str(Path)` contains no `/`, so the swap never fired and the
test passed without testing anything. `abspath` collapses `..` lexically on both `posixpath` and
`ntpath` (checked), so the new variant exercises the normalisation everywhere.

**Rejected.** *An autouse DB-isolation fixture in `tests/conftest.py`* — the right end state, but
it touches every test that reaches the database, which is not a fix to make blind on the day the
suite is discovered to differ by platform. *`skipif(os.name != "nt")` on the path test* — hides
the platform difference instead of testing it. *Dropping the case half* — kept on Windows only,
because `normcase` folds case only there and a case-different path on ext4 *is* a different file.

**What it opens.** Nobody looked at Actions for five days while three baton entries said "every
gate green locally": the third instance of the CI-blind shape (uv.lock at 0.4.0 · the container
until 2026-09-04 · this) and worth a KI with a session-start check (`gh run list --branch main
--limit 1`). The two DB-isolating fixtures (`env` in the integration file, `empty_library` here)
want a shared one in `conftest.py`.

---

## 2026-09-07 (2) — v0.6.0 published; a README tell pass; §7b learns what the publish command actually does

**What changed.** The GitHub release for `v0.6.0` exists and is Latest: id 383972486, asset
`Provenote_0.6.0_x64-setup.exe` at 1,648,765,716 bytes with GitHub's own digest
`sha256:2280…5bc4` equal to the local hash, body = `.claude/release-notes-0.6.0.md` (11,099
chars, ends in the SHA-256 line). `releases/latest` now answers 0.6.0, so ADR-044's update check
on a 0.5.1 install has something to compare against. `README.md`: a pass for writing tells — the
contrastive slogans ("measured, not asserted", "estimates, not promises", "costs one paper, not
the library"), the aphorisms ("megabytes tell you almost nothing"), three "deliberately", one
"surprisingly", and two headline phrases added earlier today ("Your files, added your way", "the
answer can show you its source"); every number and the bold-lead structure stayed. `docs/RELEASE.md`
§7b: the command now carries the asset and `--verify-tag`, and the section says where the notes
live, why they need a status header, and that the command drafts, uploads, then publishes.

**Why.** Two `gh release create` runs raced: one from this session, one started a minute earlier
from the Run button on the command shown in the previous message. The earlier one drafted,
uploaded and published at 09:36:44Z; this session's uploaded a second copy to its own draft and
failed the final flip with HTTP 422 "Release.tag_name already exists", after which gh deleted its
draft — zero drafts remain, one release object, correct. The runbook did not say the command drafts
first, so a draft with zero assets read as a failure mid-way, and nothing said to check
`gh release list` before running it a second time.

**Rejected.** Killing the second process once both were seen — one had already published by then,
and gh's own cleanup handled the loser. Editing the entry below to say "published" — append-only;
this entry is the correction.

**What it opens.** `README.md`, `docs/DEVLOG.md`, `docs/RELEASE.md` are uncommitted. The GIF
(storyboards in the baton) and the README alt text that goes with it. `docs/DEMO.md` and the KI-48
heading, as noted below.

---

## 2026-09-07 — 0.6.0 release notes and README brought level with the tag; the release object itself is the user's command

**What changed.** `README.md` re-read against the tagged code, the way `docs/RELEASE.md` §2 asks
of the CHANGELOG — and two Limitations bullets were false. The OCR one said a pure scan "is
unreachable" and that OCR is "deliberately not built"; KI-47 measured the opposite (PyMuPDF uses any
`tesseract` on PATH, 0 → 34,600 characters on the same file), and the bullet now carries the
corrected 0.6.0 text. The reference-links one still described the surname+year matcher, "4 links
where 16 are stored", and named KI-45 as *next* — 0.6.0 requires title agreement (16 → 41 links,
DEVLOG 2026-08-26), so it now says that, plus the real remaining limit: links are computed at first
read and not revisited (`extract_citations --reresolve` refreshes them; not a button). The first
ingest is no longer "single-threaded" (two workers by default). **Added:** a Windows-installer route
at the top of Quick start — the README had no download pointer at all through two releases that
shipped an installer; the source-pane / *In context* bullet; the add-documents / Zotero /
per-part re-run bullet; the graph's coverage line; and a condition under the indexing-time table,
whose two OCR rows are only true with `tesseract` on PATH. Status → v0.6.0, 2,389 tests.

**The release notes** are drafted in the 0.5.1 body's shape at `.claude/release-notes-0.6.0.md`
(gitignored, like the 0.5.1 body was never committed). They carry the corrected OCR limit and say
in one sentence that the tagged `CHANGELOG.md` has it wrong; state the RG-012 verdict in two halves
(packaging strong; citation one sample — 1 failure in 4 byte-identical runs on 0.5.1, RIGOR_TODO
2026-08-14); and explain why a 0.5.1 library shows every document as *changed* after upgrading
(`is_cache_fresh` compares the recorded fingerprint against the per-format one 0.6.0 computes, so
every 0.5.1 cache reads stale — and ADR-047 means that re-read no longer orphans sidecars). The
upgrade path itself was not gated and the notes say so. Gates re-run at HEAD, which equals the tag
on `src apps scripts tests`: pytest **2,389/0** (9:33) · node:test 257/257 · mypy 98 files clean ·
svelte-check 219 files 0/0 · `docs_check --strict` OK · docs-encoding guard 5/5. Installer
SHA-256 `2280180840548023735044020e9a0e05ec418d06522909af66dbe4db9a5c5bc4`, 1,648,765,716 bytes.

**Why.** `docs/RELEASE.md` §7b: since ADR-044 the app reads `releases/latest`, so a pushed tag with
no release object is invisible to every install — 0.5.1 users cannot learn 0.6.0 exists until the
object is there, and `gh release list` shows only 0.5.1 and 0.4.2. The README drifted the same way
the CHANGELOG had: its Limitations were written for 0.5.0 and nobody re-read them at 0.6.0.

**Rejected.** *Re-tagging onto `b5ad5de`* to carry the corrected CHANGELOG — the tag is public and
peels to the source the installer was built from; moving it breaks the tested-equals-tagged diff §6
depends on. The correction lives in the release body, which is what people read. *Creating the
release from the agent session* — `gh release create` was refused by the permission classifier as
a publish, which is the right call: §7 makes publishing the one deliberate, irreversible step. The
exact command is in the baton. *Committing the notes under `docs/`* — the release page is the
body's home and the CHANGELOG is the durable record; a third copy rots.

**What it opens.** The release is one command away. The README GIF is a release stale (the 0.5.0
storyboard: no source pane, no add-documents) — three slideshow storyboards are in the baton, per
the user's ask. `docs/DEMO.md` still calls Connections *scored* (ranked since 0.5.1) and does not
mention the source pane — not touched. `KNOWN_ISSUES.md` still heads KI-48 as OPEN while DEVLOG
2026-08-25 (3) fixed it at the cause; the heading wants reconciling.

---

## 2026-09-04 (2) — CI builds the container, and checks the two things a green build does not prove

**What changed.** A third job in `.github/workflows/ci.yml`, alongside `ci` and `frontend`: free
~10 GB on the runner, build the image through buildx with the GitHub Actions cache, then assert
**(a)** torch is the `+cpu` wheel with zero `nvidia-*` distributions and **(b)** `apps.api` and
`doc_assistant` import inside the image.

**Why.** Nothing referenced the Dockerfile between `a052703` (2026-08-01) and today, so the
container was the one of this project's three shipping paths — desktop installer, source checkout,
headless image — that no gate touched. The pinned `ghcr.io/astral-sh/uv:0.12.1` base had never been
exercised on any machine (the dev box runs uv 0.11.14), and a `uv.lock` that had drifted would have
failed `uv sync --locked` in the image and surfaced only when somebody needed the container. Which
is exactly how it came up: the user asked whether Docker still worked, and the honest answer was
that nothing had checked since August.

**The two assertions are the job, not the build.** A green build says the layers assembled. It does
not say the image is the right one: `pip install ".[cpu]"` ignores `[tool.uv.sources]`, resolves
torch from PyPI, and that Linux wheel bundles CUDA — several GB of `nvidia-*` in an image with no
GPU, which still builds and still runs. KI-34 is the standing version of this lesson at the desktop
end: an artifact that started cleanly, served `/api/health`, reported a healthy chunk count, and
could not read a single PDF.

**Two bugs found in this job while writing it, both by running it rather than reading it.**

1. The nvidia count was `ls /app/.venv/lib/python*/site-packages | grep -c "^nvidia" || true`, which
   prints `0` — a **pass** — when the glob matches nothing at all. Relocating the venv would have
   turned the check off silently instead of failing it. It now asks `importlib.metadata` inside the
   image, and that the scan is not blind is itself checked: pointed at `torch`, it fails with
   `packages found: ['torch']`.
2. Rewriting it to a single line made the whole workflow **unparseable YAML** — a plain scalar
   cannot contain `": "`, and the f-string is `f"nvidia packages in the image: {nv}"`. It parsed
   before the edit and not after; only re-validating caught it. Both `run:` steps are block scalars
   now, with the reason recorded inline.

**Verified by extracting the commands from the parsed YAML and executing those exact strings**
against the built image, rather than retyping them: `torch 2.12.0+cpu | nvidia packages: 0` and
`apps.api ok; doc_assistant 0.6.0`.

**Rejected.** *Booting to a green `/api/health`* — first run downloads the embedder and reranker,
which is why the Dockerfile's `HEALTHCHECK` carries a 300 s start period; that belongs in the
release gate, not on every push. *A path-filtered trigger* — the Dockerfile's inputs are
`pyproject.toml`, `uv.lock`, `src/`, `apps/api/` and `scripts/`, which is most of the repo, so the
filter would have saved nothing and hidden the cases it did skip. *Plain `docker build` with no
buildx cache* — the dependency layer is ~8 minutes and the Dockerfile already orders its `COPY`s to
make it cacheable; not using that would have made the job the slowest thing in CI for no reason.

**What it opens.** The image is ~6.3 GB and the GHA cache is capped at 10 GB per repository, so the
`mode=max` export may thrash once other caches compete. If it does, the fix is `mode=min` or
dropping the cache export and paying the eight minutes. Left as-is because the first failure will
say so plainly, and guessing at it now would be tuning against an imagined problem.

---

## 2026-09-04 — The 0.6.0 known limits, checked line by line: one was inverted, one stale, and one I broke

**What changed.** Three bullets under `## [0.6.0] → Known limits` in `CHANGELOG.md` (`6da294b`).
The 0.5.1 section was deliberately left alone — a released section records what was true then, so
the exception belongs in the 0.6.0 entry, which is where somebody deciding whether to install 0.6.0
reads.

**Why.** `docs/RELEASE.md` §2 makes this a judgment step and names the failure it guards: the 0.4.1
draft claimed a clean-machine install was unverified for three days after it had been verified — "a
limit that silently becomes a lie". Checking all eight bullets against the code rather than
re-reading them found six sound and two not — and produced a third error of my own, recorded below
because it is the sharpest instance of this session's recurring failure.

- **The OCR limit is TRUE, and I broke it before restoring it — the most useful thing in this
  entry.** I read it as fiction and rewrote it to say the app has no OCR. Four "confirmations"
  agreed: no OCR package in `pyproject.toml` or `uv.lock`; `tesseract` never in `src/` in the whole
  history (`git log -S`); `pymupdf4llm.to_markdown` takes no `ocr` argument and the library's own
  `check_ocr` import is commented out; ADR-039's Context says a scan "extracts to nothing". Every
  one of those is about **code this repo can read**, and the mechanism is not in this repo's code:
  PyMuPDF discovers a `tesseract` binary on PATH by itself. `KI-47` had the measurement all along —
  the same scan yielding **0 characters on 2026-08-08 and 34,600 on 2026-08-19**, nothing in the
  repo changed — and `C:\Program Files\Tesseract-OCR\tesseract.EXE` v5.4.0 is on this box's PATH
  right now. ADR-039's Context is not counter-evidence either: it was written 2026-08-01, eighteen
  days before the behaviour was found. The corrected bullet now carries the measurement, so it is
  stronger than what it replaced.
- **The moved-file limit was inverted.** It led with "is treated as a new document", while its own
  next sentence said the content is recognised. `_existing_document_id` (`ingest/store.py:67`)
  matches on `doc_hash` **first** and falls back to path, so a moved file keeps its id, its figures
  and its corrections. Only path-keyed state is lost — exclusions live on registry rows keyed by
  `pathkey` — which is what the bullet now says.
- **The inherited update-check limit was resolved and nobody noticed.** 0.6.0 said "everything
  under 0.5.1 still applies"; 0.5.1 said the update check cannot compare until releases are cut.
  `gh release list` shows Provenote 0.5.1 (Latest) and 0.4.2 (Pre-release) published.

**Re-measured rather than assumed**, for the two limits carrying numbers: KI-57 is still open at
0.2% of pages corpus-wide, and the keyword figure is **97.0%** — 1,358 of 1,400 attached keywords
on exactly one document, across 98 documents — so 0.5.1's "97%" is still exactly right at a corpus
one document larger.

**Rejected.** *Editing the 0.5.1 section* — Keep a Changelog treats a released section as a record,
and rewriting it would falsify history to fix a forward reference. *Deleting the OCR bullet* — the
underlying limit is real and load-bearing (a scanned PDF is unreadable and marked broken); only the
mechanism was invented.

**What it opens.** A blanket "everything from the previous release still applies" inherits claims
without naming them, which is how the update-check line was re-shipped unread; naming each
carried-over limit would cost a few lines and make each one checkable.

The larger opening is the OCR mistake. **`.claude/KNOWN_ISSUES.md` was never consulted** during a
review whose entire job was "is this still true", and KI-47 answered the question directly, with a
measurement. Reading the code proves what the code does; it does not prove what the *running system*
does, and the gap between those is exactly where an undeclared external dependency lives. The
verification order should be: what did we already measure, then what does the code say — not the
reverse. This is the same fail-open shape as the four gates fixed on 2026-09-02, committed in a
release note by the person who had just spent two days cataloguing it.

---

## 2026-09-02 (3) — RG-012 passed on 0.6.0, and the preflight could not see it

**What changed.** One character class in `scripts/release_preflight.py`: `_CHOSEN` now reads
`([\d,]+(?:\.\d+)?)` where it read `([\d,]+)`. Five parametrised tests pin the parse.

**Why.** RG-012 Tier-2 passed on the 0.6.0 installer at 17:49 — clean Windows Sandbox, `python on
PATH? False`, 181 s silent install, health at 210 s, 3 PDFs to 322 chunks, a 14 s turn with 10
sources and 4 resolved citations, 0 unresolvable. `release_preflight` then reported **"no RG-012
run matches this installer — the gate ran against a DIFFERENT build"**, with the archive count
correctly up from 9 to 10. It had found the run and rejected it.

The harness logs its size as `[math]::Round($bytes/1MB, 1)`. Every installer before this one
happened to land on a whole number — 0.5.1 is 1572.0318 MiB, which renders as `1572` — so a pattern
accepting digits and commas matched for four releases running. 0.6.0 is 1572.3855 MiB, renders
`1572.4`, and the line stopped matching. The check had never been right; it had been lucky, with
roughly a one-in-ten chance of exposure per build.

**The failure direction is what makes this worth a log entry.** A parser that drops the record it
is looking for reports *absence of evidence* — and this check's absence message is an accusation
("the gate ran against a DIFFERENT build"). The rational response to it is to re-run a 20-minute
clean-machine gate that has already passed, or to override the check by hand. Both are worse than
the check not existing. This is the third instance today of one shape: `versions` never opened two
files, `artifact_fresh` compared the wrong quantity, `rg012` could not parse its own harness. In
each case the check *ran*, and what it silently failed to see was the thing it was for.

**Rejected.** *Loosening to `([^)]+)`* — it would parse, but the size is the one field that makes
the log line self-describing, and a pattern that accepts anything stops being a guard. *Making the
harness print an integer* — the harness is the record; changing what it writes to suit a reader is
backwards, and the archived logs would still not parse.

**On what the PASS is worth.** The packaging half is strong: it is the half that found KI-34, and
nothing else exercises the frozen artifact end to end. The citation half is one sample of a
measurement `.claude/RIGOR_TODO.md` reopened on 2026-08-14 as unreliable — a coin flip on
`llama3.1:8b`, where 0.5.1 failed once and passed twice on the same installer. Recorded in
`docs/desktop-packaging.md` §5 so the next reader does not take "4 resolved citations" for a
stability claim.

**Host state.** The run needs Ollama reachable from the sandbox. Rather than the documented
persistent `OLLAMA_HOST` user variable, this run set it **process-scoped** on a directly launched
`ollama serve`, so there was nothing persistent to revert — verified afterwards: both env scopes
empty, listener back to `127.0.0.1` only, gateway address refused, 9 models still served locally.
Worth preferring next time: the documented procedure leaves a variable that has to be remembered.

---

## 2026-09-02 (2) — `artifact_fresh` judges git history, not file mtimes

**What changed.** `check_artifact_fresh` no longer asks "is any tracked source file's mtime newer
than the artifact?". It asks `_newest_shipped_change()`, which splits the question in two:

- **committed** files are dated by the **committer date of the newest commit touching a shipped
  path** — never by the file on disk;
- **uncommitted** files are dated by mtime, which is the one place an mtime means what it looks
  like it means: a person edited the file, and it is not in history yet to be dated any other way.

`SOURCE_GLOBS` (dead — defined, never referenced) and `_newest_source` are gone. In their place
`SHIPPED_PATHS` names the fifteen paths the artifact is actually built from, and `_is_shipped`
matches them exactly: a `/`-terminated entry by prefix, a file entry only against itself.

**Why.** The old comparison demanded a rebuild after a plain `git checkout main`, which
re-materialises files with today's date and byte-identical content — `src/doc_assistant/__init__.py`
was blob `a789456…` at both the built commit and HEAD, and the preflight called it a source edit
(2026-09-02). A gate that cries wolf on a branch switch is a gate that gets overridden by hand,
which is how it stops working. Content makes an artifact stale; a checkout is not an edit.

The old path list was also short: it covered `src/`, `apps/api/` and `apps/desktop/src/` and
**not** the Rust shell, `tauri.conf.json`, the PyInstaller spec or `build_sidecar.py` — so an edit
to the spec, which is exactly where KI-34 lived, could not have marked the artifact stale. Same
failure as the version check's two missing Cargo files, so it gets the same guard: the list is
pinned by `test_the_shipped_path_list_is_pinned`, which also asserts every entry exists on disk.

**`Cargo.lock` is deliberately excluded, and it is a judgment call.** Cargo rewrites the lock
*while building*, so a release build necessarily ends with a lock newer than the artifact it just
produced; counting it would fail this check on every release, which is what happened at 0.6.0. Its
one release-relevant field (the crate version) is covered by `versions` instead. **Residual gap,
stated rather than hidden:** a dependency version changed in the lock without a rebuild is not
caught here. `uv.lock` stays *in* the list — nothing in the build rewrites it, so the asymmetry has
a reason.

**A bug found by probing rather than reasoning.** The first implementation read `git status
--porcelain` and sliced `line[3:]` for the path. `_run` ends in `.stdout.strip()`, which eats the
leading space of an unstaged ` M path`, shifting every offset by one: the path came out as
`rc/doc_assistant/__init__.py`, failed `is_file()`, and the entire uncommitted branch was a silent
no-op. One test caught it; two wrong guesses at the cause (a fatal pathspec, then a timestamp tie)
were both disproved by running the thing in isolation. It now splits on whitespace and never
depends on a column.

**Six behavioural tests**, in a throwaway git repo with `ROOT` monkeypatched: a docs commit does
not move the bar; an mtime-only bump does not (the regression test, which asserts its own premise —
that the bumped file *is* the newest thing on disk — so it cannot quietly stop testing anything);
an uncommitted shipped edit does; a committed shipped edit does; a `Cargo.lock` commit does not.

**Rejected.** *Recording the built commit in a stamp file at build time* — the correct answer in
the abstract, and still the better one if this ever needs to be exact. It was rejected here because
existing artifacts carry no stamp, so the check would have to degrade to "cannot tell" for the very
release that motivated the fix, and back-filling a stamp by hand is the "a PASS from a previous
build reads as evidence" hazard this file already warns about. *Keeping mtimes and special-casing
the checkout* — there is no way to tell a checkout from an edit by mtime, which is the whole point.

**What it opens.** `artifact_fresh` now trusts commit dates, so a rebased or amended history with
rewritten committer dates could in principle move the bar backwards. Committer dates are set at
commit time and a rebase rewrites them to "now", so this is monotonic in practice on one machine;
it would need revisiting if releases were ever cut from a rewritten branch.

---

## 2026-09-02 — The version check now reads the two Cargo files, and its file list is a test

**What changed.** `scripts/release_preflight.py`'s `versions` check went from five sources to
**seven**: `apps/desktop/src-tauri/Cargo.toml` (`[package] version`) and `Cargo.lock` (the
`doc-assistant-desktop` entry, found by name in the package list) now join the five it already
read. `collect_versions()` is split out of `check_versions()` so the *list of files* is importable
and therefore testable, and `docs/RELEASE.md` §1 grew from six rows to eight.

Three tests, in `tests/unit/test_release_preflight.py`:

- **the source list, pinned by equality** — adding a version-carrying file means adding it here;
- **no source may read as a sentinel** — `(not found)` and `(missing)` compare equal to each
  other, so seven simultaneously-broken readers would have "agreed";
- **a drift in any single file must FAIL**, parametrised over the source list rather than
  spot-checked, so a file added to the list gets its negative case for free.

**Why.** The `versions` check reported green while `Cargo.toml` and `Cargo.lock` held `0.4.1`
through **v0.4.2, v0.5.0 and v0.5.1** — three tagged releases (verified by reading each tag:
`git show vX.Y.Z:apps/desktop/src-tauri/Cargo.toml`). It never opened them, and neither did the
runbook table. This is the *inverse* of the `uv.lock` incident that created the check: not a file
someone forgot to edit, but a file nothing was looking at. An agreement check is worth exactly as
much as its file list, and until now that list existed only inside a function body.

Surfaced at 0.6.0 the hard way: the release build regenerated `Cargo.lock` from 0.4.1 to 0.6.0
*after* the release commit, and `tree_clean` — not `versions` — was what caught it.

**Verified by reverting the fix.** With the two Cargo sources removed from `collect_versions()`,
exactly three tests fail (the list test and both Cargo drift cases) and the other 14 pass. The
guard reproduces the historical bug rather than merely describing it.

**The re-lock command is verified too.** `docs/RELEASE.md` §1 now carries
`cargo update --manifest-path apps/desktop/src-tauri/Cargo.toml -p doc-assistant-desktop --offline`
— run against a deliberately desynced tree, exit 0, one line changed, no network. `cargo metadata`
was tried first and rejected on evidence: it wants metadata for every locked package including
Android-only ones this box has never downloaded, so it exits **101** under `--offline` (after
writing the lock) and needs the network without it; `--no-deps` exits 0 and updates nothing.

**Rejected.** *Deriving the file list from the runbook table* — a docs parser is a second thing to
break, and the table is prose. *Hand-editing `Cargo.lock`* — it is a lock; cargo overwrites it at
build time anyway, which is precisely the failure being fixed. *Extending the existing
"do they agree?" test* — it structurally cannot catch a missing source, which was the bug.

**What it opens.** `artifact_fresh` has the same shape of weakness one layer over: it compares
**mtimes**, so `git checkout main` re-materialising a byte-identical file (blob `a789456…` at both
`ef4a6d8` and `663c290`) fails it. Comparing `git diff <built-commit> HEAD` instead would say what
the check means. Not done here — it needs the built commit recorded next to the artifact, which is
a change to the build, not to the check.

---

## 2026-09-01 (6) — Two UI corrections from using the app: controls too small to find, and a dropdown painted by the OS

**What changed.** Both reported after driving the merged build in the native Tauri window.

1. **The source pane's header controls were too small to find.** 0.15rem of padding under a 0.7rem
   label, drawn in the muted `--fg-2`, inside a border that reads as part of the pane frame. Three
   changes, no redesign: a real hit target (**26px** row, steppers squared to 26x26 from 23px wide,
   close 29x26), the resting colour moved off `--fg-2` onto **`--fg`**, and hover that fills the
   button rather than only tinting the glyph. The active fit preset now takes the accent as a
   *fill* (`--accent` / `--accent-fg`) instead of tinting its text, so which mode is on is legible
   at a glance. Focus rings added on all three groups.
2. **The chat folder-scope dropdown did not match the app.** Its options and popup were painted by
   the user agent in the **OS** scheme — a light menu over a dark app.

**The second one was not a colour bug in that component.** `.scopepick select` sets
`background: none`, so the closed control was already correct; what was wrong is that **the app
never declared `color-scheme`**. Without it the UA paints every native widget in the system scheme
regardless of the page's palette. So the fix is one declaration per theme state in `app.css`
(`:root`, `[data-theme='dark']`, `[data-theme='light']`, and the `prefers-color-scheme` block) —
next to the palettes, not on the one control, because the same mismatch was in **all five**
`<select>`s and every native scrollbar fallback.

**Verified live** in the running app, both themes: `color-scheme` resolves `dark` / `dark` / `light`
across system-default, forced-dark and forced-light, and the scope `<select>` inherits it in all
three. Control contrast checked in both — active preset indigo-on-white in light, and the close
button now `#ece5d6` on dark where it was the muted `#a79e8b`.

**Rejected.** *Styling `option` backgrounds directly* — works in Chromium, does nothing for the
popup chrome or the scrollbars, and would need repeating in five places. *Hardcoding
`color-scheme: dark` on the control* — correct in one theme and wrong in the other.

**Gates.** node:test 257/257 · svelte-check 219/0. CSS-only plus one icon size; no logic touched.
**$0 — no model call.**

## 2026-09-01 (5) — Row 18 closed out: a citation now opens its page, and the two branches no test could reach were driven for real

**What changed.** Two gaps, both named in the 2026-09-01 (1) baton as unfinished.

1. **A chat citation can open its page.** The source card gains **Show the page**, which navigates
   to the Library and opens the pane where the passage is. `GET /api/library/chunk-page` now
   returns `{document_id, page}` rather than a bare page: a chat citation carries a `chunk_key` and
   **no document id**, and turning one into the other means reading the chunk store — so it happens
   on the server rather than by parsing the key's shape in the client, where the second copy of
   that contract would rot. New `library.locate_chunk` + `ChunkLocation`; `page_for_chunk` is now a
   thin wrapper on it.
   **It is offered for a figure too**, which the card's own comment had anticipated: a figure has
   no position in the text, so the page image is the only place it can be shown.
2. **The unavailable and text-only arms were driven**, against the live library, with a backup taken
   first — a file moved out from under a document, then a document's `format` flipped to `epub`.
   Both restored; the library matches `data/library.db.bak-20260901-183252-prearms` row for row.

**And that is where the two real defects were, neither of which a test could have caught** — the
corpus is 98/98 PDF with every file present, so no test fixture stands in for driving it:

- **The size and zoom controls rendered over a document that has no page.** "Fit page | Width |
  − 100% +" sat above the sentence *"The file is not where the library expects it"*. A dead
  control, and this project's own rule is that a dead control is worse than none. Now gated on a
  renderable page, with the close button taking the right-hand margin when they are absent.
- **Two pieces of copy that were wrong.** The backend said *"a epub document has no pages"* — an
  article cannot agree with a value read from the database, so it is now *"a document in EPUB
  format"*. And the pane said the extracted text was **below** while its own hint said **beside**:
  Chunks is beside the pane in the split layout and above it when stacked, so any direction is
  wrong half the time. Both directions dropped.

**A false defect, avoided by the project's own rule.** After the jump, the citation panel appeared
to stay open over the Library — the DOM still held it 2 s later and a screenshot showed it. It had
in fact closed; the node was mid-transition, exactly the stranding the baton warns about
(2026-08-31 (3)). **When the DOM and the state disagree, believe the state**: querying again showed
the card gone. Nothing was "fixed".

**A test that had to be rewritten, for a reason worth keeping.** The first version of the route
test monkeypatched `source_view.locate_chunk` and failed — the route resolves the name through the
**package** re-export (`from doc_assistant.library import locate_chunk`), which is a separate
binding. That is the trap in `src/doc_assistant/CLAUDE.md`, met from the other side. Rather than
patch the re-export, the fake chunk store now holds real rows, so the tests exercise the real
derivation — cache markers on disk, offset in the metadata, page derived.

**Verified live.** A mocked `/api/chat` turn (fabricated sources, a **real** `chunk_key`; no model
call, no cost — the discipline from the 2026-08 chat-UI note) → click the citation → **Show the
page** → the Library opens `03-Zuo2014_NBR_fconn.pdf` at **"Page 10 of 19 · cited here"**, the page
computed independently beforehand. Unavailable and text-only arms both render a sentence, no image,
no broken image, no page nav, no size controls.

**Gates.** pytest source-viewer suites **44/44** (+7) · node:test 257/257 · svelte-check 219/0 ·
mypy 98/0 · ruff + format clean · bandit 0 · `detect-secrets` clean against the baseline.
**$0 — no model call.**
