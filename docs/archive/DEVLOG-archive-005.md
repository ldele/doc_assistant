<!-- status: archived · updated: 2026-08-30 · class: append-only -->

# DEVLOG — archive 005 (2026-08-08 (1) → 2026-08-11 (4))

Older entries, moved verbatim from `docs/DEVLOG.md` on 2026-08-30 so the working log stays
about recent work. Newest-first, same format, unedited. Rotated because the live log had
reached 4,031 lines against the 4,000-line cap in `tests/unit/test_doc_sizes.py`, which fails
before it can grow further. Cut on a date boundary so no day is split across two files.

---
## 2026-08-11 (4) — a whitespace hook wanted to edit a CC-BY vocabulary (ADR-043), and chasing it found four docs already corrupted by the encoding hazard this project documents

**What changed.** `docs/decisions/ADR-043-received-content-is-preserved-verbatim.md` (new) +
its index row; `.pre-commit-config.yaml` gained one `exclude` with its reasoning inline; the two
Android adaptive-icon XMLs got their missing final newline; and **four tracked docs were repaired
from double-encoded text**: `docs/ROADMAP.md`, `docs/architecture.md`, `docs/knowledge-layer.md`,
`docs/decisions.md`.

**Why the hook was wrong, which is the ADR.** `pre-commit run --all-files` had been red for weeks on
`data/anzsrc-2020-for-20210429.ttl`. Read as a formatting nit it is an obvious fix. It is not one:
**28 of 28** trailing-whitespace hits sit **inside Turtle `"""…"""` literals** (`skos:definition`,
`skos:scopeNote`), so the hook was editing the text of a classification published by the Australian
Bureau of Statistics. The file is vendored under **CC BY 4.0**, `data/anzsrc_2020_for.json` cites it
in `_meta.source_file` as the artifact it was derived from, and **nothing parses it at runtime** —
no `.ttl`/`rdflib`/`turtle`/`skos` reference anywhere in `src/`, `scripts/`, `tests/`, `apps/`;
rdflib is not a dependency. So the edit bought nothing and cost the only property the file has.
Excluded, not obeyed. ADR-043 generalises it to two tests a formatter must pass — *did we author
it* and *is whitespace load-bearing* — and records the direction the user asked for: the same rule
applies to **ingest**, where it is currently bent on the user's own documents
(`strip_image_placeholders` KI-14, marker stripping KI-29, de-hyphenation RG-025), and where a
genuine judgement call should become a **user option** with an opinionated default.

**Then the hunt for that class of bug found the class already realised.** Surveying all **166**
tracked markdown files for the Windows encoding hazard the project names as non-negotiable #9 turned
up **four double-encoded files — exactly the four carrying a UTF-8 BOM**, which is the signature of
one tool having read UTF-8 as the ANSI codepage and re-saved: `ROADMAP.md` (228 occurrences),
`architecture.md` (113), `knowledge-layer.md` (60), `decisions.md` (12). Two of the four are linked
from the README, so this was public. `architecture.md`'s flow diagram rendered `â†“`/`â†’` instead
of `↓`/`→`, and **`knowledge-layer.md`'s trust table** — the one `AGENTS.md` tells every agent to
read *before believing any marker* — had a severity column of `âš ï¸` and `âŒ` instead of ⚠️ and ❌.

**The repair needed three attempts, and the first two failures are the interesting part.**
1. **Blanket `encode('cp1252').decode('utf-8')`** is the textbook inverse and is valid only if
   *every* non-ASCII character is damaged. A guard tested that per file and **refused three of the
   four** — they mix damaged text with correct characters (`→`), and the blanket form would have
   corrupted the good ones. Only `architecture.md` was uniformly damaged.
2. **Targeted runs, cp1252 only**, cleared all visible mojibake but left **9 control characters**:
   cp1252 leaves `0x81/0x8D/0x8F/0x90/0x9D` undefined, so any sequence containing them cannot
   round-trip through it. Those were emoji — `⚠️`'s U+FE0F variation selector and `❌`.
3. **Per-character byte mapping (cp1252, then latin-1)** finished it. `❌` (E2 9D 8C) mojibakes to
   `â` + U+009D + `Œ`, where U+009D is encodable **only** by latin-1 and `Œ` (U+0152) **only** by
   cp1252 — the run mixes codecs, so the mapping has to be per character, not per run.

**Verified, not assumed.** All four: **0 mojibake, 0 BOM, 0 control characters**. Every ASCII-only
line byte-identical and line counts stable in each pass (the check that proves only damaged
characters moved), and pre-existing correct `→` preserved rather than re-mangled. `pre-commit
--all-files` **all 11 hooks Passed, exit 0** — the actual goal, since a full run that is always red
trains people to ignore it. `docs_check --strict` 0/0.

**Rejected.** Stripping the `.ttl` literals (mutates a published dataset for nothing). Deleting the
`.ttl` (destroys the attribution chain its derived JSON cites). Fetching it on demand like the eval
corpus (that pattern exists for large, redistribution-fraught PDFs; neither applies to a 926 KB
CC-BY vocabulary). Rewriting `6484004` to split the release commit out of the `chore(hooks)` subject
— **it is already on `origin/main`**, so the history is public and stays as-is.

**What it opens.** ADR-043's user-facing half is direction with no design: how a normalisation record
is stored, surfaced, and paid for at 10,000 documents is unsolved, and de-hyphenation (RG-025) now
inherits a constraint it did not have — reversible and inspectable, not merely correct on average.
Also unresolved: nothing prevents the encoding damage recurring. Every one of those four files was
committed broken and no gate noticed; a `docs_check` rule for BOM + mojibake in tracked text would
have caught all four at the commit that introduced them.

---
## 2026-08-11 (3) — release prep for v0.5.0: the front docs stopped overstating the product, and the demo GIF stopped leaking the author's chat history

**What changed.** The reader-facing set (`README.md`, `evals/README.md`, `docs/DEMO.md`,
`CHANGELOG.md`), a re-recorded `docs/assets/provenote-demo.gif`, and the 0.4.2 → **0.5.0** bump
across all five places (`pyproject.toml`, `uv.lock` re-locked not hand-edited,
`apps/desktop/package.json`, `apps/desktop/src-tauri/tauri.conf.json`, `CHANGELOG.md`).

**Three of the doc fixes were defects, not staleness — all three flattered the product.**

1. **The README's benchmark table quoted `llm_judge` 3.894 ± 0.075.** That is the **2026-06-01**
   column, and `public_eval_2026-08-01.md` says in so many words to compare against 06-04 rather
   than 06-01, because the 06-01 baseline does not record its generator model (`.env` switched to
   `claude-haiku-4-5` on 06-02, *after* it was locked). The current committed run is **3.694 ±
   0.258** — i.e. the README was publishing the highest of three numbers and the least trustworthy
   one. Now 2026-08-01 across the board, with the two caveats that keep it honest: `citation_overlap`
   is saturated on a 10-paper corpus (no regression *at the available resolution*, not ranking
   parity), and the judge band only resolves changes larger than about ±0.5.
2. **`evals/README.md` still declared the chunk-size lock "unmeasured".** True on 2026-08-07, false
   from 2026-08-08 — RG-026 closed with two self-audited sweeps. Rewritten with the actual verdict:
   the lock holds, *and* un-beaten is not optimal (child `256/32` retrieves best at −45% input
   tokens; parent `3000/300` answers best; the control is the balanced point), plus the one
   experiment that would settle it — re-running the private grid on a strong generator, since the
   small-child answer penalty was measured through `llama3.1:8b` and Haiku does not reproduce it.
3. **The README sold the `contested` / `superseded trend` chips as a shipped feature** while its own
   Limitations section said they were withheld. They default **off** (KI-33, since 0.4.1). Both
   places now agree.

**Limitations was re-read line by line, per the runbook's rule that a stale limit becomes a lie.**
Three added: a scan with no text layer at all is unreachable (1 of 97); **most in-library reference
links are withheld** (4 presented where 16 are stored, because 12 were wrong — KI-45); epistemics
off by default. The local-model bullet gained the citation-coverage numbers (36% / 14% vs 81%),
which are what a reader actually feels, rather than only the taxonomy-precision figure.

**The GIF was re-recorded, and the interesting part is what it was showing.** The old cut framed the
chat with the sidebar open — which renders the **author's real chat history**, personal research
questions, into a public README. The recorder now collapses the sidebar for the chat beats and
*asserts* `aside.sidebar` is 0 px wide before filming. Two more guards, both from failures observed
this session: it **aborts unless the status bar reads `ollama/`** (`.env` is all-Anthropic, so an
unguarded run bills the API — KI-4), and every beat logs what it matched and exits non-zero on a
miss. `make_gif.py` now selects frames **by label instead of index**, because the storyboard gained
four beats and every hardcoded number shifted — the positional plan resolved those to
"skip (missing)" and would have written a silently shorter film while reporting success.
New storyboard adds the 0.5.0 surface: library filter → the five-block document view → figures →
a figure at full size. 960×600, 18 frames, 29.6 s, 0.97 MB, recorded on `llama3.1:8b` at $0.

**A bad first draw is also recorded, because it is a real product signal:** the first document
filmed showed **4 of its 5 figures as "no image"**. That is an honest empty state (caption found, no
image region located), not a bug — but corpus-wide **811 of 881 figures do carry an image**, so the
draw was unrepresentative. Chose the demo document by querying `figures.image_path` rather than by
eye, and the recorder now counts placeholders and reports a bad draw as a named miss.

**Why.** v0.4.2 is two releases behind the code (46 source files changed since that tag), and the
README's Status block still read v0.4.0 / 1,446 tests. A release is the moment the front docs are
read by people who cannot check them against the source.

**Rejected.** Updating `docs/performance.md`'s "97 documents / 33,105 chunks" to today's 36,574 —
those figures are measurement-bound, and re-labelling them would attribute results to a corpus that
was never measured. Committing the GIF toolkit into `scripts/` (third refusal: puppeteer-core and
Pillow would both be undeclared deps). Letting `pre-commit --all-files` keep its whitespace/EOF
rewrites of three unrelated checked-in files — reverted, so the release diff is only the release.

**What it opens.** `release_preflight` is green on `versions` and `changelog` and red only on
`tree_clean` (this commit) and **`artifact_fresh`** — the sidecar and installer still date from
2026-08-07 and must be rebuilt from the tagged commit before v0.5.0 can carry an artifact
(`docs/RELEASE.md` §4-5, then the RG-012 Tier-2 clean-machine gate). Separately, **KI-39's entry was
corrected**: it was fixed on 2026-08-06 and had read OPEN ever since; what remains open is the
RG-010 cold-start distribution, which is a measurement, not that defect.

---
## 2026-08-11 (2) — the UI checklist is prioritised instead of exhaustive (90 KB → 20 KB), and the 2026-08-11 feature set is queued behind keyword quality

**What changed.** `docs/ui-checklist.md` rewritten: a **§1 priority queue** at the front, then open
work by theme, verification debt, a compressed Shipped section, and the per-feature review gate.
Seven new user asks (2026-08-11) folded in. The pre-rewrite file is kept **verbatim** at
`.claude/ui-checklist-archive-001.md`.

**Why.** It had grown to **90 KB / 202 lines**, with single rows running 11,480 characters —
entire diagnostic essays restating what DEVLOG, the ADRs and KNOWN_ISSUES already own. A board you
cannot scan is not a board: nothing said what to build *next*.

**The archive is local-only, deliberately.** `docs/ui-checklist.md` is gitignored working state
(ADR-029) but `docs/archive/` is **tracked** — putting the verbatim copy there would have pushed
local working notes into a public repo. It went to `.claude/` instead, which is already the
gitignored home for that class of file. Several rows carried measurements that exist nowhere else,
so freezing rather than deleting was the only safe compression.

**What the compression preserved.** Each row now points at its source instead of repeating it, but
the *load-bearing* constraints stayed inline, because they are the ones that get re-derived
expensively when lost: `ANSWER_PROMPT`'s citing block is the integrity layer's wire format, not a
prompt · `EMBEDDING_MODEL` is the catastrophic knob and is governed by nothing · BM25's `avgdl` is
corpus-global, so per-document chunking is contained but not free · reference order is not
recoverable from today's schema · a keyword that looks like junk is usually specialist vocabulary.
A coverage diff against the archive caught **four open items** the first pass silently dropped
(two shipped-feature defects, the RAG-governance row, conversation rename) — they were restored.

**Priorities, as the user set them.** P1 **keyword quality** — explicitly ahead of the advanced
RAG/chat modes, and it is repair rather than research: all three causes were already traced
(extractor truncation on `.`/`/`, no suppression list, an undemoted 1-doc tail). P2 **LLM-assisted
ingestion** (opt-in, over the programmatic default). P3 **projects**. P4 chat modes. P5 search.
P6 KI-45. P7 settings.

**The one synthesis worth recording.** Conversation folders, a home/project picker, and per-folder
concepts arrived as three separate asks across two days — they are **one feature**. Building them
separately yields three grouping systems that disagree about what a "project" is. The checklist now
says so and gates all three behind a single ADR.

**Rejected.** Deleting the long rows outright (several held the only copy of a measurement).
Splitting the board into per-theme files (the point is one scannable queue). Archiving to
`docs/archive/` (tracked — would leak local state).

**What it opens.** The Project ADR is now the blocking artifact for three queued features. P2 needs
a spec + a cost-gate decision before any LLM ingestion pass runs.

**Follow-up the same day — P1's baseline overturned P1's own queued fix.** Measuring before planning
(`docs/PLAN_2026-08-11_ingestion-quality.md`) found the keyword layer at **1,376 keywords, 98% on a
single document, exactly one reaching five documents, 15 per document, and 15 of 97 documents with
none**. The checklist's framing — "60 keywords, 50% singleton, demote the tail" — was measured at 76
documents and had gone stale in kind, not degree: at 98% singletons the layer does not partition the
corpus at all. Two samples gave the mechanism: `transformer_vaswani_2017.pdf` spends **9 of its 15
slots** on overlapping shingles of one figure artifact (`eos` / `eos pad` / `pad` / `pad br` /
`eos pad br`), and `nihms-66884.pdf` spends **11 of 15** on shingles of the PMC running header. So
the ranked defects are page furniture → shingle overlap → stale enrichment → the `_TOKEN_RE` `.`/`/`
truncation → author leakage — and **the fix that was queued first is fourth**. Recorded as a memory
because the stale number had already misled the plan once.

---
## 2026-08-11 — chat history gets a cleanup: the whole thing exports to one file, then many chats delete at once

**What changed.** `POST /api/conversations/export` renders every conversation into one markdown
file; `POST /api/conversations/bulk` soft-deletes (or restores) a list of them in one transaction.
In the UI: a **✓ select mode** in the sidebar's chat rail (tick rows, All/Clear, Delete, Done) and
an **"Export all conversations"** action in a new Settings → *Chat history* section.

**Why (user request 2026-08-10).** *"An easy way to clean-up the historic — maybe in settings a
full export + delete history; or a select option to select chats so I can delete many at once."*
Both primitives already existed — single-conversation soft delete and per-conversation export —
so this is the bulk layer over them, not new capability.

**Export lands before delete, and that ordering is the design.** The sidebar's delete is a *soft*
delete: the `AnswerRecord` rows survive, which is right for provenance and useless as a restore
path — nobody cleans up their history confident that "the rows are still in the database". So the
file comes first, the confirmation dialog names it (*"To keep a readable copy, export your history
from Settings first"*), and the Settings section sits above the destructive control rather than
beside it.

**The export is uncapped, and that is the point.** `list_conversations` stops at ~100 by design
(Decision 10). An export that inherited that cap would omit conversations *invisibly*, exactly
when the file is being relied on — so `all_conversation_ids` is a separate, uncapped read.
Measured on the real history: the sidebar shows **100**, the export wrote **184 conversations /
188 turns / 347 KB**. Eighty-four would have gone missing silently.

**One document, not a zip:** greppable, opens anywhere, no archive handling on either side. Each
conversation keeps its own heading and session id so a single one stays findable inside it.

**Bulk delete is one transaction, not N requests** — "delete selected" is a single user action and
half of it landing is worse than none of it. Per-row semantics are identical to the existing
single delete, so it is undone by the same route with `deleted: false`, which is what makes the
confirmation honest. Unknown and duplicate ids are not errors: the sidecar row is created on first
action, so "delete a conversation with no meta row" is the ordinary case.

**Kept local:** the *selection* lives in `Sidebar.svelte`, not App. Nothing else in the app needs
to know which rows are ticked — unlike the Library's select mode, where the main pane's
add-to-folder menu participates, which is why that one lives in App. Two new props instead of
seven.

**Rejected.** A hard delete (the provenance layer is the product; a cleanup that destroys evidence
is a different feature needing its own ADR). Looping the existing single-delete endpoint N times
from the client. Putting export-all next to the sidebar's delete button — it is a whole-history
action, and the confirmation already points at it. **Conversation folders**, the third thing asked
for, are *not* built: `Folder` is document-scoped (ADR-025), so grouping chats is a new
relationship and needs its own ADR rather than a schema reuse.

**Verified live on the real history ($0).** Select mode ticks rows and counts them (100 tickboxes,
"2 selected"), the dialog reads *"Delete 2 conversations?"*, confirming exits select mode; bulk
delete + restore verified **by exact session id** (the list count proves nothing here — with 184
conversations behind a 100-row cap, deleting 2 just lets 2 more fill in). The history was left
exactly as found: 184 live, 1 soft-deleted, and that one was already deleted ~16 h before this
session — checked by timestamp rather than assumed.

**What it opens.** Conversation folders (ADR needed). An age-based retention policy, still parked.
And the export is whole-history only — a "export just these selected chats" would fall out of the
same two pieces if it is ever wanted.

---
## 2026-08-10 (2) — the index moved out of the scroll, the blocks became a list, and a figure can finally be read

**What changed.** Three refinements to the document view, all from looking at it in use.

**1 · The index is above the metadata and outside the scrolling area.** It was a sticky strip
*inside* the scroller, which meant it passed over the text as the reader scrolled — a bar that
covers what you are reading is worse than no bar. `.browser` is now a flex column of a fixed
`nav` + a `.scroller` that owns the overflow, so the strip is always visible and never overlaps
anything. `scroll-margin-top` on the blocks drops from 3.6rem to 0.5rem: there is no longer a
sticky element to clear. (`min-height: 0` on both is load-bearing — without it the flex child
refuses to shrink and the inner scroller never gets a scrollbar.)

**2 · Parent blocks are collapsed too, as a scannable list.** Each row is marker + `Block N` +
**a preview of its text** + child count, opening to the full block. The preview is the point:
`blockPreview` strips the leading markdown (`## `, `**`, `> `) that carries nothing at one line
wide and truncates at a word boundary, because "Block 0 / Block 1 / Block 2" is not a list anyone
can read. Expand-all / Collapse-all for the reader who does want the whole thing. Measured on the
82-block paper: opening Chunks is now **842 DOM nodes instead of 2,089**, and 2,432 with every
block expanded — so the default costs a third of what it did, and the ceiling is still reachable.

**3 · Figures open full size, with a real zoom.** The cards cap images at 180 px, which is enough
to recognise a figure and not enough to read one. `FigureViewer.svelte` (scrim + centered card +
Esc, the `AboutDialog` pattern) has **two levels**: fit-to-window, then one click to natural size
inside a scrolling frame — the browser's own scrolling is the pan, so no zoom library. ← / →
step through the document's figures, and a new figure always starts fitted (carrying the previous
one's zoom over would drop the reader into the middle of an image they have not seen whole).
Verified on a full-page plate: thumbnail 180 px → fitted **914 px** → actual size **1755 px**,
**9.8×**, stage scrolling. The viewer steps only over figures that *have* an image, so ← / →
never lands on a card with nothing to show.

**Rejected.** A click handler on the `<img>` for zoom — `svelte-check` flagged it, correctly:
zooming has to be reachable from the keyboard, so it is a `<button>` wrapping the image and
Space/Enter come free. A separate "expand" icon on each card: the thumbnail is already the thing
the reader is pointing at.

**What it opens.** Scroll position is still not restored with a document's remembered chunk
state. The parent-block open set resets on document change (the block *list* is cheap; the open
set is not worth persisting). And the chat-history cleanup the same review asked for — bulk
delete, export-all, conversation folders — is logged in `docs/ui-checklist.md`, not built: both
primitives exist (`App.svelte:392` soft delete, `POST /api/export` per conversation), so it is a
bulk layer over them, and the folders half needs its own ADR (`Folder` is document-scoped, ADR-025).

---
## 2026-08-10 — the Library document view is five ordered blocks, chunks cost nothing until asked for, and the References block's links had to be verified before they could ship

**What changed.** The document view (`LibraryBrowser.svelte`) is now **Metadata → Connections →
Chunks → Figures → References**, each with a heading, an anchor, and a sticky nav strip that jumps
between them. Chunks is **collapsed by default and not fetched until opened**. A new
`GET /api/library/documents/{id}/references` + `DocReferences.svelte` render the paper's own
bibliography at the foot of the view, with the entries already in the library as links. The
outgoing half of the connections bundle (`cites`, `external_refs`, `external_total`) moved into
that endpoint and is gone from `DocConnections`.

**Why (user feedback, 2026-08-09).** Two rows: the organisation was liked but *"the structure is
not legible"*, and a references block was wanted at the bottom where *"a reference already in the
library is a link"*. The chunk collapse was raised as a **performance** concern, and it is one —
measured here, the detail payload is a **median 170 KB and up to 1.85 MB** per document (663
parent blocks + 2,608 children for `hebb_1949.pdf`), all of it rendered eagerly on open. Live on a
142-block paper: opening the document is **924 DOM nodes**, expanding Chunks adds **2,984**, and
collapsing returns to 924. Opening a document now transfers no chunk text at all — the header
reads the summary the Library list already holds.

**The part that nearly shipped broken.** The block's headline feature is the link, so the links
were measured first. **13 of the 16 stored resolutions in this library are false** — one
document's reference *"A review of graph neural networks and pretrained language models"* pointed
at a paper on axonal projections in mouse whisker cortex. Cause: `match_to_library`'s second rule
matches **first-author surname + year with no title comparison**, and resolution is computed only
at insert time, so it also never re-runs (**KI-45**, filed with both halves and the fix sketch).
In a research-integrity app a false "you own this paper" is worse than no link at all.

**So the read side verifies what the write side asserted** (`resolution_is_credible`): exact DOI,
or title ratio ≥ `FUZZY_TITLE_THRESHOLD` (the matcher's own 0.80, now a shared constant), or one
normalised title contained whole in the other. Corpus-wide that takes the presented links from
**16 to 4**, and all four check out by eye. A rejected resolution keeps its place in the list and
loses only its link — the paper does cite it.

**Containment is why the threshold did not have to move.** A strict ratio also rejected a *true*
match: the regex prefixes titles with the tail of the author list (*"A., Lopes, G., … Real-time,
low-latency closed-loop feedback …"*), scoring 0.78. Containment recovered exactly that one and
admitted **none** of the 12 false links, which score 0.11-0.37 and contain nothing. The
populations are cleanly separated, which is the evidence that the defect is upstream and not a
threshold to tune.

**Two more things the real data showed, both fixed.** `target_year` carries 5 impossible values
(2034-2089, lifted out of identifiers) — harmless at 0.1% of 4,282, except that sorting
newest-first put **every one of them at the top of the block**, so `plausible_year` drops what
cannot be a publication year and they sink instead. And a `scrollIntoView({behavior:'smooth'})`
across the ~76,000 px of an expanded document is an animation, not a jump; it is instant now.

**Rejected.** Keeping `cites` in the connections bundle *and* adding a References block — the two
would have shown the same document's outgoing citations under two headings, split by whether the
match happened to resolve. Capping the reference list by simple truncation: the cap spends its
budget on linked rows first, so a paper with 346 references cannot lose the one entry the reader
can actually open. Sorting the owned references to the top: the block is a bibliography, and
re-ordering it to flatter one feature would misrepresent it.

**Chunks remembers per document, for the session** (`chunkmemory.ts`, added on review): the reader
asked that the top ← → arrows come back to a document as they left it. It remembers the **open
flag keyed by document id, never the payload** — restoring the state costs one re-fetch (measured
on this box: **27 ms / 258 KB** typical, **339 ms / 1.83 MB** for `hebb_1949.pdf`), while caching
the text would mean holding that per visited document, which is the cost the block exists to
avoid. Session-scoped, not `localStorage`: remembering "open" across launches would restore the
eager render on a fresh start, where nobody has asked for anything yet. The restore runs inside
`untrack` — `startChunkLoad` reads `detail`/`loading`, and a synchronous read in a Svelte 5 effect
makes them dependencies, so the completing fetch would re-run the effect, null the payload, and
fetch again forever.

**Rejected on the same question:** persisting the preference globally (a remembered "open" would
restore the 1.85 MB worst case on *every* navigation, which is the request inverted) and expanding
by default (the old behaviour, and the reason the request exists).

**What it opens.** KI-45's write-side fix (~19 more correct links, none of them reachable from the
UI until resolution re-runs). A `Citation` ordinal column, which is the only way the list could be
in the paper's own order — today it is year-descending, and the block says so. The other consumers
of `target_document_id` (`cited_by`, `graph_subgraph`, `concept_skeleton.py`'s provenance pairs,
the CLI display) are still reading the unverified rows. And scroll position is *not* restored with
the open state — coming back re-opens the block at its top.

---
## 2026-08-09 (2) — the page-scan discriminator was defeated by OCR: 109 "full-page plates" were 3 scanned PDFs

**What changed.** `is_page_scan` now requires **two** conditions to exempt a full-page region, not
one: a caption **and** a page carrying at most `FIGURE_PLATE_MAX_TEXT_LINES` (20) lines of text.
New `text_line_count` at the impure boundary feeds it, counted once per page.

**Why.** The 2026-08-08 rule exempted any captioned full-page region, reasoning that "a scanned
page has no text layer, so there is no caption to pair". **A scan with OCR does.** Three documents
here are exactly that: the text layer puts a `Fig. N` block on every page, pairing succeeds, and
every page is stored as a figure. Measured: **109 of 962 rows (11.3%) were whole pages** —
`Computational_neuroanatomy.pdf` **81 of 81**, `hodgkin_huxley_1952.pdf` 22/23,
`hubel_wiesel_1959.pdf` 6/9 — and **55 had already been paid for at the VLM**, describing a page of
prose rather than a figure. The same entry's claim that these were "the 109 real full-page figures
this corpus does contain" was wrong, and it was wrong because the number was never looked at.

**How the threshold was chosen — three signals tried, two rejected.**
- **Caption share of page text** — rejected: a body block that matches the caption regex inflates
  it (`hodgkin_huxley` p1 paired a 1659-char "caption", so a page of prose scored 0.709).
- **Tallest text-free band** (fraction of page height) — rejected, and this is the interesting one:
  it ranked `hodgkin_huxley` p2, which **does** carry a real circuit diagram, at **0.091 — below
  every pure-text page (0.095-0.110)**. The figure is small and inline, so it leaves no hole.
- **Text line count** — kept. Immune to caption fragmentation (which corrupts any character
  measure), and the two populations separate: verified-by-eye plates at **4-13 lines**, pages of
  prose at **25-117**. The cut sits in the empty band, and a guard test asserts `13 < N < 25`.

**Verified before applying, on the real PDFs:** the 3 scanned documents go **81→28, 23→1, 9→3**;
three control documents (`transformer_vaswani_2017`, `1707.01836v1`, `cajal-lecture`) move by
**exactly 0**. Then applied: **962 → 881 figures**, 43 survivors re-described (0 errors).

**The limit, stated rather than hidden.** A scanned page carrying prose *and* an inline figure is
now dropped with the rest — `hodgkin_huxley` p2's circuit diagram is a real loss. Every text-layer
signal ranked that page below pages with no figure at all, because the figure's location is in the
pixels, not the text layer. Recovering it needs the image analysis `regions.py` deliberately keeps
out of the v1 hot path. Losing one figure beats keeping 81 pages of prose.

**Rejected.** Tying the budget to `CHUNK_SIZE_CHILD` (a "paragraph of prose" is tempting, but it
would make figure extraction change silently whenever the chunking sweep moves a locked setting).
Self-calibrating against the document's median lines/page (more elegant, unvalidated today, and it
drops the verified 13-line plate).

**What it opens.** Pixel-level region detection inside a scanned page — the only way to recover an
inline figure from a scan, and the same lever that would crop it instead of storing the page.

---
## 2026-08-09 (1) — the figure claim is VERIFIED end to end; the 17.5% VLM loss had one cause, and it was never transient

**What changed.** Yesterday's headline — "the RAG can retrieve images" — was built but unproven,
because no `chunk_type='figure'` chunk existed. It exists now: **575 figure chunks across 83
documents**, and a figure retrieves and arrives inside the passage that cites it. Plus three code
fixes the run itself forced.

**The diagnosis, and why the fix was a one-line record.** The 2026-08-08 pass lost **96 of 549
calls (17.5%)** to `ValidationError`, recorded as `f"error: {type(e).__name__}"` — the type without
the message, so a 96-failure run was undiagnosable without paying for another one. That is KI-42's
truncated-note defect in a second place. `describe_error_reason` now records the message
(whitespace-collapsed, capped at 400 chars), and **the cause appeared within one minute of the
re-run**: on a figure carrying several values Haiku returns `key_quantities` as one comma-joined
**string**. A forced tool-use schema constrains the shape the model is *asked* for, not the shape
it returns.

**It was never transient, and the retry proved it.** A single retry was added on the belief
(2026-08-08) that the failures were random — a hand re-run of one image had succeeded. Measured:
of 153 calls, **31 failed both attempts**. If failures were independent at 17.5% only ~5 would
have; ~20% failing twice means the per-figure probability is near 0.45 and **the same figures fail
repeatedly** — the ones with many quantities. So the retry is worth keeping but is not the fix;
`FigureDescription` now accepts the string shape. **Kept whole, not split on commas:** we do not
know where the model intended item boundaries, and `1,000 ms` would become two quantities that
were never in the figure.

**Two corrections to yesterday's plan, both found by running it.**
- **`ingest` alone is a no-op** — `--dry-run` reported `skip_unchanged=97, would_add=0`. The dedup
  gate is `h in indexed` on the *extracted-text* hash, which a VLM pass never changes, so a
  described figure cannot enter retrieval without `--rebuild`. Filed as **KI-44**: at 97 documents
  that is ~4 min; at the contract's 10,000 it is hours, for a sidecar that changed nothing about
  the text.
- **Two dated DB backups were staged into this public repo** (`library.db.bak-…` 8.4 MB,
  `eval.duckdb.bak-…` 5.5 MB — the private corpus's metadata). `.gitignore` covered the bare names
  but not a suffixed copy. Unstaged before any commit; `data/*.bak*` now ignored.

**Verified, with the numbers.** Rebuild: 97 added / 0 errors, and KI-43's own interim check run by
hand around it — `figures` 962→962, `chunk_epistemics` 445→445, `concept_presence` 31→31, **0 ids
changed**. Then: all **575** chunks open with the figure's own text; **231 carry a citing
passage** (194 `cited`, 37 `placed`); **0** disagree with their `figure_context` flag; 4/4
description-derived probes retrieved the figure (3 at rank 1), each arriving as figure-then-prose.

**The honest part.** Those probes are *plumbing* — the query is built from the chunk's own text.
On five natural questions written blind, **2 surfaced a figure** ("a diagram of the transformer
architecture" → that paper's architecture figure at **rank 1**) and 3 did not. And the 60% of
figures with `figure_context: none` is **not** a matching failure: **336 of 344 have no caption at
all**, 2 have a caption the label parser cannot read (`"Source: by authors Figure 1. …"` — the
regex anchors at the start), and only **6 of 962** are labelled-but-unmatched. The `none` bucket
is a *4b captioning* observation, not a 4c one.

**Rejected.** Splitting `key_quantities` on commas (invents boundaries). Loosening `_LABEL_RE` to
`search` for those 2 captions (a mid-caption "Figure 1" would attach the wrong figure — 2 rows is
not worth a wrong-attribution class). Writing figure chunks into Chroma directly to dodge the
rebuild (it would verify a substitute for the real ingest path, which is the thing under test).

**What it opens.** **87 figures remain eligible and undescribed** (31 that failed twice + 56 the
per-document budget skipped) ≈ $0.17 — one of the three natural-query misses was a paper whose
figures are in that set. KI-44 (an incremental trigger for sidecar-only changes). The 336
caption-less figures: worth asking whether 4b's caption pairing is missing them or they genuinely
have none.

---
## 2026-08-08 (7) — figures become retrievable *in context*, and browsable per paper (L1b)

**What changed.** Two halves of the same goal — make the RAG able to answer with images.

**RAG side.** A figure chunk's `parent_text` is no longer the figure alone. New pure helpers in
`ingest/figures.py` — `figure_label`, `reference_pattern`, `find_figure_context`,
`figure_parent_text` — locate the prose that *uses* a figure, and `ingest` attaches it. Two rules:
**cited** (a passage says "as shown in Fig. 2.2") beats **placed** (the passage carrying the
caption). 29 tests.

**Library side.** New `library/figures.py` (`list_document_figures`) + payloads + route
`GET /api/library/documents/{id}/figures` + `DocFigures.svelte`, wired into `LibraryBrowser`
beside `DocConnections`. 15 tests.

**Why the parent change.** Figures were `parent == child` — "atomic retrieval units", self-contained
and citable with no surrounding argument. A figure means something *because of the passage that
argues from it*; retrieving the description alone hands the model a caption and a shape. The user
asked for the citing section as parent, with positional placement as fallback, and the two rules
above are exactly that.

**Three decisions inside it worth keeping.**
- **The caption is stripped from a parent before searching it.** The caption contains the label
  too, so without this every figure "cites" itself and the `cited` branch is unreachable — the
  feature would have silently degraded to `placed` for every figure and still looked like it worked.
- **The figure keeps its own `parent_index`.** Sharing the citing parent's would let `pipeline`'s
  dedup (keyed `doc_hash`+`parent_index`) drop the figure whenever the prose chunk was already
  retrieved — the figure would vanish from exactly the answers it is most relevant to.
- **The figure's own text comes first in `parent_text`.** It is what matched the query; burying it
  under a page of prose invites the model to answer from the surrounding argument and cite the
  figure for it.

**Why the library panel reports retrievability.** A figure enters retrieval only once it has a VLM
description (`figure_units` filters on exactly that). A panel that listed rows without that
distinction would show images the assistant cannot see, with nothing to tell them apart — so each
card carries `retrievable` and, when false, a **translated** reason (`caption_sufficient` →
"Caption already describes it"). Same posture as the rest of the UI: inform, don't hide.

**Rejected.**
- **Widening `figure_units`' tuple** to carry the caption. It is monkeypatched by ingest's
  write-ordering guard tests; a separate `figure_captions()` means a caller that patches only
  `figure_units` degrades to the old self-contained behaviour instead of raising.
- **A fourth `/api/library/*` prefix** for figures. `/connections` is already a document
  sub-resource in `documents.py`; a new prefix would have made the composition docstring false.
- **Hiding non-retrievable figures.** They are the ones the user most needs to see — that is the
  gap between what is in the paper and what the assistant can reach.
- **Mixing figures into the chunk browser.** The user asked for them separate, and they are a
  different kind of object; interleaving them into parent blocks would bury them in prose.

**Measured, and the number is small.** Median figure is 787×376 → **327 image tokens** (median;
mean 502), well under Haiku's 1600 cap — so ≈$0.002/figure, **~$1** for all 549 eligible. Sampled
one document first: the description of `nihms-66884` p30 matches the two-panel schematic on the
page, and it transcribed the state-space equations including `Σ` correctly.

**What it opens.** The full VLM pass was still running at session close (189/962 described), so
**no figure chunk exists yet** — `ingest` must run after it to materialise them, and the retrieval
claim is unverified end to end. That is the first task next session.

---
## 2026-08-08 (6) — ADR-042: a document's identity is its source, not its extraction (KI-43's decision)

**What changed.** `docs/decisions/ADR-042-document-identity-is-the-source-not-its-extraction.md`
(**proposed**) + its row in the living index + **RG-027** in `.claude/RIGOR_TODO.md`. No code.

**Why.** KI-43 is a design fault, not a bug with a patch: `document_id` is resolved from the hash of
the **extracted markdown**, so any extraction change mints a new id and orphans every id-keyed
sidecar. Fixing it in place would have edited the locked ingest path and every sidecar's key while
the session was busy with something else.

**The framing that decided it:** `doc_hash` is answering two questions with opposite requirements —
*"is my extraction current?"* (must change when extraction changes) and *"which document is this?"*
(must not). No single hash can do both, so the fix is to separate them: identity becomes the
**source file's bytes**; `doc_hash` keeps the cache/version job.

**A second observation sharpened it into something more useful than a re-key.** `extract_figures`
opens the **PDF** — markdown is not an input — so keying figures to the extraction output was a
category error, not a fragile choice. But that is not true of every sidecar: `chunk_epistemics` and
`concept_presence` are keyed to chunk indices and *should* die when chunking changes. So the ADR
splits sidecars into **source-derived** (must survive) and **extraction-derived** (invalidate
**loudly**). Deleting the latter is not the error; deleting them silently is.

**Rejected** (each recorded in full in the ADR):
- **Identity = source path.** Cheapest, and it regresses the move/rename property the current design
  deliberately bought — *and* is unsafe in the other direction: replacing a file at the same path
  inherits the previous document's sidecars, turning a loud loss into a quiet wrongness.
- **Keep content identity, migrate sidecar rows on change.** Correctness would depend on a
  hand-maintained list of id-keyed tables, so the next sidecar anyone adds is silently omitted — the
  same class of defect that produced this one.
- **Detect-and-warn only.** Adopted as an **interim, not an alternative**: it makes the loss visible
  without preventing it, and it stays useful after the migration because it verifies the invariant
  the ADR asserts rather than assuming it.

**Status is honest: `proposed`, not accepted.** The direction is settled and the defect is measured,
but the migration is unbuilt and its backfill unvalidated — nothing yet shows that assigning a
`source_hash` to 97 existing documents preserves every sidecar link, nor that the source file is
always available and stable to hash. Both are ⚠-marked in the ADR's Confidence section and owned by
**RG-027**.

**What it opens.** RG-027 is the gate. Until it closes, the standing rule stands: **re-run
`extract_figures --apply` after any extraction change, PDF-library upgrade, or table splice.**

---
## 2026-08-08 (5) — the figure pipeline was wrong in three independent ways; 45 rows → 962 real ones

**What changed.** `config.FIGURE_MAX_AREA_FRACTION = 0.85` + `figures.is_page_scan` (a page-sized
region with no caption is the page, not a figure); `extract_figures`'s `--force` can now **clear** a
document to zero instead of only replacing. 20 unit tests + 3 integration tests. **KI-43 filed** for
the third fault, which needs a decision rather than a patch.

**Why.** The user reported the image pipeline "not working well at all". The `figures` table read
**0 rows**. It had held 45. A re-derive produced 1522. So it had been running at a few percent of
its true size, and 46% of what it *did* produce was not figures.

**Fault 1 — silent data loss (KI-43, open).** `document_id = _existing_document_id(h) or str(uuid4())`
keys identity on the hash of the **extracted markdown**. Any extraction change mints a new id and
orphans every id-keyed sidecar. Measured: 11 of 97 documents changed id across the chunking sweep,
and **all 10 that owned figures were among them** — not coincidence, because the 2026-08-07
text-layer fallback fires on exactly the image-heavy documents. `chunk_epistemics` 743 → 445,
`concept_presence` 66 → 31. The comment above that line claims figures stay linked; the table-splice
runner's docstring concedes the opposite. **The project's own table splice triggers it.**

**Fault 2 — a scanned page is not a figure.** `select_region_bboxes` had an area floor and no
ceiling, so a scan (one full-page image per page) yielded one "figure" per page: `hebb_1949.pdf`
gave **365 for 365 pages**, exactly 1.00/page, zero captions.

**The cut is structural, and the distribution says so.** Area-fraction over 1452 rows is bimodal
with an effectively empty band: **783 below 0.7, one row in [0.7, 0.9), 669 at/above 0.9** — cuts at
0.80 / 0.85 / 0.90 partition it identically, so 0.85 is the middle of a gap rather than a tuned
number (the same shape as `_TEXT_LAYER_KEPT_MIN`). The two sides are different populations:
**51%** captioned below, **16%** above, and only **7%** in the top bucket. **The caption is the
discriminator, not the area** — a genuine full-page plate has a caption, a scan has no text layer so
it has none. That keeps the 109 real full-page figures instead of trading one systematic error for
another.

**Fault 3 — `--force` could replace but not remove.** Found only because fixing fault 2 exposed it:
`hebb_1949` still read 365 rows *after* the ceiling correctly rejected all of them, because the
persist call was guarded by `and regions`. A document that legitimately drops to zero kept its rows
forever, and the runner printed `no-figures` while the database said 365. **The runner's own output
was right and nothing reconciled it against stored state** — the same shape as KI-42.

**Result, and it is validated rather than counted.** 45 → **962 figures across 92 documents**;
captioned rate **37% → 59%**; per-doc max 365 → 105, median 7; 936 PNGs, **0 rows pointing at a
missing file**. `hebb_1949` and `middleton-2001` correctly hold zero. Spot-checked crops against
their captions — e.g. `nihms-66884.pdf` p30 is a cleanly-bounded two-panel schematic matching
"Fig. 3. A schematic model for generating goal directed movements".

**Rejected.**
- **Restoring the 45 rows from the pre-sweep backup.** They were stale *and* mostly page scans;
  figures are derived data, so re-deriving is both correct and cheaper than reconciling ids.
- **Rejecting page-sized regions on area alone.** Simpler, and it would have destroyed the 109
  genuine full-page plates — swapping a false-positive problem for a false-negative one.
- **Fixing KI-43 in place** by carrying the id forward on a filename match. It is the right fix and
  it changes the locked ingest path plus every id-keyed sidecar; that wants an ADR, not an edit made
  while fixing something else.
- **Tuning the ceiling per corpus.** The empty band is the justification; a value chosen to make
  this library's numbers look right would be exactly the corpus-tuned constant the robustness
  contract forbids.

**What it opens.** Figures are re-derived but **not yet described** — `describe_figures` (the paid
VLM pass) has never run against this set, so `vlm_description` is empty and Feature 4c's
`chunk_type='figure'` chunks do not exist yet. That is the next step and it costs money. KI-43
remains the standing hazard: **until it is fixed, re-run `extract_figures --apply` after any
extraction change, PDF-library upgrade, or table splice.**

---
## 2026-08-08 (4) — Marker is pinned and its failures are legible (KI-42 fixed); RG-025 closes on the comparison it unblocked

**What changed.** `config.MARKER_VERSION = "1.10.2"`, used as `uvx --from marker-pdf==<version>`;
a new `MarkerUnavailableError` that `extract_tables_marker` refuses to swallow; and a report that
sorts errors **first** with their cause **untruncated**. 10 tests in
`tests/unit/test_marker_availability.py`. RG-025 closed, its baseline updated with the head-to-head.

**Why.** KI-42: `uvx --from marker-pdf` was unpinned, so it resolved to **marker-pdf 2.0.0**, whose
surya routes inference through a spawned backend — `vllm` wants a running Docker daemon (auto-picked
on an NVIDIA GPU), `llamacpp` wants an uninstalled `llama-server`. Both die at the *layout* stage,
so the shipped table path did not run on this machine at all.

**The invisibility was the worse half, and it had three independent causes.** The availability guard
checked that `uvx` **exists** — it does; the failure is two layers deeper. Every failure was then
swallowed into a per-document status row, its note **clamped to 30 characters** (cutting
`RuntimeError: marker_single e|xited...` mid-word), and error rows sorted by table count, which put
them **below every success** in a 97-row report. A total outage was formatted to look like scattered
per-document noise.

**The fix rests on one asymmetry:** *a document with no tables still exits 0*. So a non-zero exit is
never a fact about the PDF — it is a fact about the machine, and the run stops and says so instead
of producing 96 more useless rows.

**Verified end to end, not just in tests:** the runner found **7 tables** in `rag_lewis_2020.pdf` in
40 s on the pinned version.

**RG-025 closed with the comparison this unblocked.** Tesseract+Ghostscript vs Marker 1.10.2 on the
same 3 pages: **85.8% vs 85.5%** word-like — a tie on accuracy, at **~1/30th** the runtime (≈1.4 s
vs ≈40 s per page) and without a model stack. **ADR-039 option 1 confirmed**, and its
dependency-surface argument is stronger than when it was made, since Marker's runnability turned out
to be version-fragile. Marker's one real edge is **reflow**: Tesseract leaves **88 hyphen-split
words** across the document (`cortico-\nspinal` will not match a query for "corticospinal"), so
**de-hyphenation is now the named quality task the OCR sidecar inherits** — a deterministic
post-process, not a second engine.

**Rejected.**
- **A cheap capability probe** (`marker_single --help`) as the guard. It would have passed on 2.0.0 —
  the break is at model-spawn time. A real probe must run Marker on a page, which costs ~2 min; the
  pin plus fail-fast gets the same protection for free.
- **Starting Docker / installing llama.cpp** to make 2.0.0 work. Multi-GB system installs on the
  user's machine to satisfy a comparison, when the last working release is one pin away.
- **Floating the pin** (`>=1.10,<2`). The whole defect is that an unpinned escape hatch changes
  without a commit; a range is a smaller version of the same bug.

**What it opens.** The table path runs again, so a `--apply` splice pass across the corpus is now
possible — **but note it rewrites the markdown cache, which changes `doc_hash`, which orphans
id-keyed sidecars (KI-43)**. Re-run the enrichment chain after any splice.

---
## 2026-08-08 (3) — OCR of the one true scan is good enough to retrieve (RG-025, 3 of 4); Marker turns out to be unrunnable (KI-42)

**What changed.** Measurement only, no source change. New baseline
`tests/eval/baselines/ocr_quality_middleton_2026-08-08.md`; RG-025 advanced 3 of 4 items and left
open on the fourth; **KI-42 filed**.

**Why.** ADR-039 chose its OCR engine on architecture, not accuracy, and RG-025 gates enabling
recovery: *a document with no text layer is honestly absent, while one full of OCR garbage is
retrievable, rerankable and citable.* The user installed Tesseract + Ghostscript, so the
measurement became possible.

**Result — it clears the bar.** All **15 pages** of `middleton-2001.pdf` (0 chars of text layer,
now the *only* non-healthy document in the library) read in **21 s**: **87.0% word-like**, zero
empty pages, against real-text-layer controls at 88.5–92.1%. Hand-read of p006 against the rendered
image: **one error in ~850 chars** of body text. Retrieval, in an isolated corpus against 8
topically adjacent real papers: **4/4 rank-1**, and clean on a negative control.

**Two things the numbers would have said wrong without looking.** The low-scoring pages (79–83%)
are the **references section** — initials, volumes and page ranges score as "not word-like" while
being read correctly, so the metric is a garbage detector, not an accuracy score. And the noise that
*is* real is **1.0% of characters**, confined to figure interiors (`MDpl`→`wippt`, arrows→`j j | j`)
— which is the genuine finding: prose is excellent, **diagram labels are not**, and those are
exactly the plausible-but-wrong tokens that become citable.

**A near-miss worth recording (non-negotiable #9).** A `§` misread showed in-terminal as `�` and
looked like an encoding bug. The file is **valid UTF-8, zero U+FFFD** — the bytes are `\xc2\xa7`
and the cp1252 console could not render them. An OCR error was one step from being filed as a
corruption bug. Byte-check before believing the console.

**Then the Marker comparison failed, and took a shipped runner with it.** `uvx --from marker-pdf
marker_single` now dies at the **layout** stage: current `surya` spawns a backend — `vllm` wants a
running Docker daemon (auto-selected because this box has an NVIDIA GPU), `llamacpp` wants an
uninstalled `llama-server`. **`scripts/extract_tables_marker.py` resolves Marker through the same
command**, so high-fidelity table extraction does not run here at all, and nothing surfaced it: the
runner's guard checks whether `uvx` *exists*, not whether Marker *works*. Root cause is that
`uvx --from marker-pdf` is **unpinned** — `eval_marker_tables.py:85` still promises "the pinned
marker-pdf version at build time" and no pin exists anywhere. **KI-42.**

**Rejected.**
- **Installing `ocrmypdf` first.** RG-025 asks whether the *text* is worth retrieving; Tesseract is
  the engine ocrmypdf wraps, so the wrapper adds a dependency ahead of the evidence that justifies
  it. ADR-039 puts the sidecar after this measurement, and that order was kept.
- **Starting Docker / installing llama.cpp to unblock Marker.** Both are system-level installs on
  the user's machine pulling multi-GB artifacts, to satisfy a comparison item — reported instead of
  performed.
- **Calling RG-025 closed.** Three of four items is not four; its contract names the Marker
  comparison as the precondition for "concluding the engine choice was right".
- **Reading the 87% as an accuracy score.** It is a word-shape heuristic, confounded by content
  type — the references pages prove it.

**What it opens.** Building ADR-039's sidecar behind its runner is now justified; enabling recovery
by default still is not (that needs the broken-rate work, RG-023/RG-024). Figure interiors argue
for letting the existing figures layer own figure regions rather than letting OCR emit them as
prose. And KI-42 should be fixed — pin `marker-pdf`, and make the guard probe the capability
instead of the launcher — before any future Marker comparison can be reproducible.

---
## 2026-08-08 (2) — the chunk sizes are measured for the first time (RG-026 closed, KI-41 resolved)

**What changed.** Two chunking sweeps, both self-audited, replacing the void 2026-06-06 run.
New baselines `tests/eval/baselines/chunking_sweep_public_2026-08-08.md` (verified-10, paid
`claude-haiku-4-5` + judge, 22 min) and `chunking_sweep_private_2026-08-08.md` (**97 docs / 35
cases**, local `llama3.1:8b`, `--with-embedding`, ~3.5 h). `.claude/CONTEXT.md`'s locked row goes
from ⚠ UNMEASURED to ✅ MEASURED; RG-026 closed; KI-41 resolved.

**The audit, which is the point.** Each run recorded **6 distinct geometries** across its 18 runs
(the void run: one geometry, six notes claiming otherwise), and on the paid run `token_input`
spans **2529 → 7044** — parent 3000 reads 7044 against the control's 4627, +52%, exactly as the
evidence-block size demands. That is the same instrument that convicted the old sweep, answering
the other way.

**The verdict: keep `2000/200 · 400/50`, now on evidence.** Nothing beats the control beyond its
variance on either corpus. On the 35-case run it is the most balanced point in the grid — 2nd on
retrieval, tied-1st on `contains_all`, 2nd on embedding — and no other config is top-two on more
than one metric. **The defaults survived, which is what the void run also claimed; the difference
is that this time the claim has an audit trail.**

**The real finding is a trade-off, not a winner.** `citation_overlap` **stops saturating** on the
private corpus (0.877–0.946, where the public 10 pinned it at 1.000 for every config) — a real
corpus supplies the distractors that make retrieval measurable, so retrieval experiments belong
there from now on. And it splits: **smaller child (256/32) retrieves best** (0.946 vs 0.936, at
*zero* trial variance since retrieval is deterministic) while scoring **worst** on answers (0.734);
**larger parent (3000/300) answers best** (0.785) at +52% tokens. Child = what gets retrieved,
parent = what the model reads. The control is the balance point.

**The confound I could not remove, stated rather than buried.** The private run's generator was
`llama3.1:8b` (36% citation coverage vs Haiku's 81%, KI-36) — the user's cost choice. So
`contains_all` there is measured through a weak model, and Haiku scored that same 256/32 child at
0.919 (level with its control) where llama put it 0.04 down. **The small-child answer penalty may
be a local-model artifact.** `citation_overlap` is immune (computed pre-generation) and is flagged
in the baseline as the one number to trust.

**Operational notes.** The public sweep ran in an **isolated data home** so the working library was
never rebuilt — verified first, because `CHROMA_PATH` does *not* derive from `DATA_PATH` (a
non-ASCII path relocates the store under `ProgramData`, the KI-11 fix). The private sweep did
rebuild the live library six times; it was restored to the locked geometry afterwards (3m50s,
97 docs, 0 errors, retrieval spot-checked). Its first config paid a full re-extraction — the first
ingest since KI-40, so every cache entry lacked a `.fp` sidecar and was stale by definition.

**Rejected.**
- **Trusting the "GPU-day" estimate.** It predates the CUDA wheel; a full re-embed is ~2.1 min and
  chunk sizes are not in the extraction fingerprint. Both sweeps together cost under 4 h.
- **Running the private sweep with the LLM judge on Ollama.** `llama3.1:8b`'s ratings are known
  flat (~0.8, non-discriminating), so it would have roughly doubled a 3.5 h run for a signal that
  would have to be discounted. `--with-embedding` is free and local, and was used instead.
- **Backfilling a cost verdict from the token win.** `1000/100 · 256/32` at −45% input tokens is
  the most tempting result here, and it is exactly the one the weak-generator confound sits on.
  Recorded as a lead with a named successor experiment, not taken.

**What it opens.** The successor experiment is specified in the private baseline: re-run configs
1/2/5/6 on the 97-doc corpus with the **shipped Haiku generator** + judge, which isolates the
confound and is what would let the 45% cost win be banked. Also open: `citation_overlap` saturating
on the public set means the public regime cannot rank retrieval changes at all — worth saying
wherever that corpus is offered as the eval.

---
## 2026-08-08 (1) — the chunking sweep now refuses to run a grid that does not reach the code (KI-41's first-run error)

**What changed.** A preflight in `scripts/sweep_chunking.py`, run before the first re-embed and in
`--dry-run` too. For each of the 6 arms it spawns the interpreter the sweep spawns, under that
arm's environment, and asks what settings the run would actually use — then fails the whole sweep
unless **(a)** every arm gets the four values it asked for and **(b)** no two arms resolve to
identical run-defining settings. Exit 1, nothing ingested. New pure helpers `probe_settings` /
`ineffective_settings` / `duplicate_arms` / `preflight`; 15 tests.

**Why.** KI-41 / RG-026, which names this as the thing worth adding before the re-run itself: the
2026-06-06 sweep drove its grid through `PARENT_CHUNK_SIZE` / `CHILD_CHUNK_SIZE`, `.env` overwrote
them (KI-38), and six arms re-embedded the same corpus. It took ~6 corpus re-embeds and two months
to notice, because **the failure direction was "no effect"** — which is indistinguishable from a
confirmed default. The previous change made a run *record* what it ran; this one makes a sweep
*check* it, before the GPU-day rather than after.

**The probe calls the contract instead of restating it.** It runs
`run_defining_settings()` — the same function that writes `config_json` on every eval run — so the
gate and the record cannot disagree. That is the rule the RG-012 false failure earned on 2026-08-07
(a gate that restated the citation contract more strictly than the app, and was believed). It also
has to be a **subprocess**: `config` resolves the environment once at import, and the sweep's
ingest and eval are subprocesses — reading the parent's own config would test nothing.

**Verified against the real config module, not only fakes.** With an arm's variables arriving blank
(`_load_env` treats empty as absent, so `.env` fills them — a live override path today), the
preflight reports `parent_chunk_size: asked 3000, effective 2000` **and** flags the arm as the
control's duplicate. Live on this box the full grid passes in ~4 s, all six distinct.

**Audited the other driver while here, since RG-026 says the failure generalises.**
`scripts/sweep_bm25_weight.py` is **clean** — it passes the weight in-process to
`resolve_ensemble_weights()` and rebuilds only the `EnsembleRetriever`; no environment, no
subprocess. It also already carries the property this preflight adds: `pre@5` moves across weights
while `post@5` is flat, which is positive evidence that the instrument discriminates. The
2026-07-03 negative result stands.

**Rejected.**
- **Checking distinctness only** (RG-026's literal wording). It catches the 6-identical-arms shape
  but not a grid that resolves to six *wrong* values distinctly, and it can only report after the
  arms have run. Comparing asked-vs-effective per arm is strictly stronger and available before
  the first re-embed.
- **A `--skip-preflight` escape hatch.** The failure this guards reads as a normal result, so the
  moment skipping is available it will be skipped on the run that needed it. There is no
  environment where the sweep can run but the probe cannot — the probe is the sweep's own
  interpreter.
- **Asserting post-hoc on `Store.run_config(run_id)`.** Correct and complementary, but it spends
  two re-embeds before it can speak, and the point of the row is to spend nothing.
- **Comparing only the chunk keys for duplicates.** Two arms recording the same *whole* snapshot
  are the same experiment whatever made them so; a difference the record cannot show is a
  difference the comparison cannot use.

**What it opens.** RG-026's remaining item is now only the sweep itself (~a GPU-day, 6 configs,
still the user's call) — and `--dry-run` is now the cheap proof it is wired first. The same
asked-vs-effective check would fit any future driver that varies a setting through a channel it
does not own; nothing enforces that on the next one but the note in `scripts/CLAUDE.md`.
