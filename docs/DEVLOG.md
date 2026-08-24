<!-- status: active · updated: 2026-08-15 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps 2026-08-02 onward.** Older entries live in the archives, moved verbatim:
> **2026-07-15 → 2026-08-01** in [`docs/archive/DEVLOG-archive-002.md`](archive/DEVLOG-archive-002.md)
> (rotated 2026-08-15) · **2026-05-21 → 2026-07-14** in
> [`docs/archive/DEVLOG-archive-001.md`](archive/DEVLOG-archive-001.md) (rotated 2026-07-21).
>
> **This file is capped at 4,000 lines** by `tests/unit/test_doc_sizes.py`. When it trips, rotate
> the oldest entries into a new `DEVLOG-archive-NNN.md` and update the list above — **do not raise
> the cap.** The cap exists because this log reached 8,244 lines before anyone noticed: every entry
> is individually small and correct, so unbounded growth is invisible per commit.

---
## 2026-08-21 (2) — the add-documents SPEC, and the accept surface turns out to need a spike first

**What changed.** New `docs/specs/feature-add-documents.md` — the code-level contract for Track A
(AD1-AD4) and Track C (CS1-CS2) of the ingestion plan, under ADR-046. Contracts per file, 15 test
cases written before the code, a DoD, the task-level decision ledger from the grill, and an open-
questions table.

**The finding that reshaped the spec: there is no way to get a filesystem path yet.** Measured
2026-08-21 — `src-tauri/Cargo.toml` carries **only** `tauri-plugin-shell`; `capabilities/
default.json` grants `core:default` plus one sidecar-execute permission; and `package.json` has
exactly **one** runtime dependency (`marked`), which `forceLayout.ts` calls *"a deliberate 1-dep
artifact"*. So the accept surface is not a UI task sitting on ready infrastructure:
- a **dialog plugin** means a Rust dependency, a registration, a capability *and* a second npm
  dependency — which breaks a stated project property, so it is a decision, not a detail;
- the **Tauri drag-drop event** is believed to carry real paths and may need neither, but that is
  belief, not evidence;
- an **HTML `<input type="file">`** is believed insufficient (a webview yields `File` objects with
  no path, and every contract in the spec is path-based).

Rather than design around a guess, the spec opens with **W0**, a gating spike whose definition of
done is a working round-trip that prints a real path, plus a note recording the chosen route and
whether it adds a dependency. **AD1's contracts are marked provisional until W0 lands.** Writing the
UI first and discovering the path problem during the build is the failure this avoids.

**Contracts recorded** (all additive, all reusing what ships): `SourceFile` gains `root_id` /
`origin` / `source_sha256` with a new `SourceRoot` table and a `uq(root_id, rel_path)` replacing
today's `rel_path` key · `registry` goes per-root · a new `library/add.py` splits **`inspect`
(no mutation)** from **`add` (no extraction)** · `delete_document` gains `delete_file: bool = False`
per ADR-046 · two API routes that mutate nothing until confirmed · a pure
`dropzone.svelte.ts` for verdict sorting and pagination, testable under the `node --test` runner the
repo already has, because the alternative is another untestable component.

**Definition of done includes a corpus fact, deliberately:** not done until an **EPUB and an HTML
file** have been added and indexed through the UI. The corpus is 97/97 PDF, so those paths are
otherwise untested in the wild — the same gap that let three extraction defects sit undetected until
2026-08-20.

**Also recorded, unresolved:** `docs_check --strict` flipped from **0/0 to 15 errors** during this
session, all of them rule-11a ordering violations in `docs/archive/SESSION-archive-001.md` and
`-002.md`. Both are gitignored, local-only, and were last written **before** any of today's work;
neither was touched. The flip is **unexplained** — the gate ran 0/0 six times earlier today with
those files in their current state. What is *not* in doubt is that the content genuinely is
oldest-first: `.claude/SESSION.md` records that archive 001 holds *"the pre-convention
bottom-appended tail"*, so those entries predate the newest-on-top rule. **They should not be
reordered** — rewriting an append-only historical record to satisfy a linter is the wrong repair.
Archive 002's header claims "Newest first" while its body is oldest-first, which is a real (small)
inconsistency worth fixing in the header rather than the body.

---
## 2026-08-21 — ADR-046: an added document is copied in or referenced in place, and three contracts move

**What changed.** New `docs/decisions/ADR-046-added-documents-copy-or-reference.md`, indexed in
`docs/decisions.md`, with **ADR-014's row annotated as partially amended**. It records the scoped
grill's three ADR-class resolutions (`docs/PLAN_2026-08-20_user-friendly-ingestion.md` §8,
branches 1, 2 and 5) — not one decision, three.

**The decision, and that it overruled the recommendation.** Both modes — copy-in **and**
reference-in-place — ship in v1. The recommendation was copy-in-first with the reference model
merely *designed*; the user chose the larger option, and the deciding reason is the part worth
keeping: **it is the only option that does not make the Zotero adapter a second migration.**
Copy-in-first would build the copy path, then rebuild `SourceFile`'s key and re-decide delete when
the adapter arrives. The ADR records the recommendation, the overrule and the reason, because an
ADR is a record of a decision rather than of who was right.

**Why it is three contracts and not a UI task.** Each was verified in code before being written down:
- **`SourceFile` becomes `(root, rel_path)`.** It is keyed by `rel_path`, documented *"relative to
  the source dir"*, with **no root column** — so referencing a file that lives elsewhere is a
  **schema** change. This is the fact that makes the whole feature bigger than it looks.
- **ADR-014 is amended.** `delete_document` sends the source to the Recycle Bin **first**. Safe only
  while the source folder is Provenote's own; with reference-in-place it would take files out of a
  user's Zotero folder. Delete now asks, defaults to **library-only**, and **must name the real
  path** when the file is referenced — the accepted risk of a per-delete choice is a mis-click, and
  showing the destination is what makes the click informed, so it is part of the decision rather
  than copy-writing. **This absorbs the queued "library-only delete" checklist row**, which was
  never a separate feature.
- **Add-time identity becomes `sha256(source bytes)`.** There is no cheap pre-extraction identity
  today: the duplicate gate is `doc_hash(text)` on the *extracted* text and fires only after
  `load_or_extract`. Source-hash identity is the direction ADR-042 already chose and RG-027 has yet
  to build, so the add path **deliberately leads** it. The ADR states the consequence plainly so it
  is not later read as drift: **two identities coexist until RG-027** — source-hash answers
  *"already added?"*, `doc_hash(extracted)` answers *"already indexed?"*, and they can disagree.

**Rejected alternatives** (all four options are in the ADR): copy-in only — cheapest, ignores the
organisation the user already has; reference only — no copy path for users who want the app to own
its files; copy-in with reference merely designed — the recommendation, kept in the ADR as the
documented fallback with a note that nothing built for the chosen path is wasted if it is taken.

**What it opens.** The SPEC (`docs/specs/feature-add-documents.md`) is now unblocked and is the next
artifact; the task-level resolutions (indexing ticked by default, pagination, partial-copy failure)
are deliberately **not** in the ADR because they change no existing contract, and belong in the
SPEC's own ledger. RG-030 gates the review sheet's *no text layer* row, not this decision.

---
## 2026-08-20 (9) — a plan for the one thing the app cannot do: accept a document

**What changed.** New `docs/PLAN_2026-08-20_user-friendly-ingestion.md` — three tracks, wireframes,
a grill list, and the process route from here to code.

**The finding it is built on.** Provenote indexes a folder well and **cannot accept a document**.
Grep across `apps/desktop/src` returns no file picker, no drag-and-drop, no upload endpoint
anywhere; the entire front door is a text box reading *"Paste the full path to the folder holding
your documents."* Everything downstream of that step already exists and is tested — registry scan,
selection resolution with up-front validation, background job, status polling, per-file exclude.
The plan therefore adds one step and reuses `POST /api/ingest` unchanged.

**Two structural facts turned up while measuring, and both change the shape of the feature:**
1. **`SourceFile` is single-root by construction** — keyed by `rel_path`, *"relative to the source
   dir"*, with no root column. "Add documents from anywhere" is a **schema** change, not a UI one.
2. **Delete bins the user's file.** `delete_document` moves the source to the Recycle Bin *first*
   (ADR-014). Safe today because the folder is Provenote's own; the moment a document can be
   registered in place, deleting it takes a file out of the user's Zotero folder.

So the innocuous-looking question *"where does an added document live?"* is an ADR with a schema
consequence and a delete-semantics consequence — **queued as ADR-046**, and it also absorbs the
separately-queued *"library-only delete"* checklist row, which turns out not to be a separate item
at all. Recommendation recorded: copy-in for v1, with the reference model *designed* in the ADR so
the planned Zotero adapter is not later found to be blocked on a decision nobody wrote down.

**A second track that needs no decision.** Making ingestion easy makes KI-47 urgent: the shipped
installer has no OCR, so a scanned PDF becomes a 0-chunk `broken` document with no explanation.
Today that is hidden behind a CLI most users never run. The plan detects it **at add time** and
says so in the review sheet, and notes that Track B beats most of Track A on value if the ADR
stalls.

**Wireframes are in the plan** (empty state · review sheet · per-file indexing progress · the two
Settings changes), ASCII so they diff and survive in git. The review sheet is the load-bearing one:
nothing is copied or indexed before the user sees per-file verdicts, and the unsupported rows carry
`get_format_status`'s advisory **verbatim** — that string already exists and already names the fix.

**Rejected alternative.** *Writing the spec now.* `docs/ui-checklist.md` says this item "needs a
grill to scope" and it is right — nine open branches are listed in the plan (duplicate rule, folder
recursion depth, mid-batch copy failure, batch caps…). Writing a contract over unresolved branches
is how a spec becomes fiction. Route recorded: **grill-me → ADR-046 → `docs/specs/
feature-add-documents.md` → ROADMAP/ui-checklist rows → build**.

**Explicitly out of scope**, with owners: P2's LLM-assisted ingestion (quality, not friendliness),
P7's settings reorganisation (only the folder picker and its relabel are taken here), the Zotero
adapter, and ADR-039's OCR recovery.

---
## 2026-08-20 (7) — a ledger of when each aspect of the project was last actually reviewed

**What changed.** New `.claude/REVIEWS.md`, and a fifth row in `AGENTS.md`'s coordination list
pointing at it. Six aspects — documentation/presentation, backend, frontend, unit testing, the cpc
system (lint, gates, CI/CD, vault), UX/UI + MCP — each with a cadence, a last-reviewed date, and a
log entry recording what that pass did **not** cover.

**Why a new file rather than a section of the baton.** `SESSION.md` is append-only with a 10-entry
cap; a cadence table needs to be *updated in place* and to survive rotation, and it would have been
archived out of existence inside two weeks. The DEVLOG is the wrong home for the opposite reason —
it records changes, and the useful signal here is the **absence** of one.

**The design decision that makes it worth having: `never` is a legal value, rendered as loudly as a
date.** Seeded honestly from the record rather than from optimism, and two rows came back `never`:
- **Backend code** — no dedicated pass is recorded, ever. Feature work and incidental fixing, yes;
  a read end to end, no. `KNOWN_ISSUES.md` shows where that lands — nine tracked defects in
  ingest/extraction alone.
- **Frontend code** — never *as a code pass*. 2026-08-19 was a live behavioural review of one
  feature, which is a different thing, and recording it as "frontend reviewed" is exactly the
  overstatement this file exists to prevent.

Two more rows are marked **partial** for the same reason: the 2026-08-19 cpc pass audited gate
wiring only (CI, the lint rule set and the vault were untouched), and **MCP has never been reviewed
because it does not exist** — the Global CLI + MCP server is parked (`docs/ROADMAP.md`, user call
2026-07-13). An empty cell there would read as an oversight rather than as a fact.

**Rejected alternatives.** *A section inside `SESSION.md`* — see above; the cap kills it. *Tracking
it in the repo* — it is working state (ADR-029), so it lives in `.claude/` with the rest; the file
says how to move it under `docs/` if it should ever be public, and nothing depends on the location.
*Recording only completed reviews* — that reproduces the blind spot the file exists to remove.

**Flagged for the cpc default template**, as the user asked, with a section at the foot naming the
two properties that must survive generalisation: `never` as a first-class value, and every log
entry stating its own gaps. A partial review recorded as a bare date reads as full coverage six
months later, which is worse than no entry at all.

**One thing it caught immediately.** `AGENTS.md` said "All four are local-only working state" about
a list whose third item is `docs/DEVLOG.md` — which is tracked. Corrected while adding the fifth
row.

---
## 2026-08-20 (6) — a fake document id read as a secret; fixed at the source rather than in the baseline

**What changed.** One id in `tests/unit/test_commands_formatters.py`:
`"abcdef0123456789abcdef0123456789"` became `"document-under-test-000000000000"`.

**Why.** The `detect-secrets` pre-commit hook failed on the change set. The finding was a **Hex
High Entropy String** — a 32-character hex document id invented as a test fixture. Not a secret,
but indistinguishable from one by entropy.

**The fix was to stop generating the false positive, not to record it.** `git add .secrets.baseline`
was the offered path and would have worked, but a baseline entry carries a `line_number`: it goes
stale the moment anything above it moves, and the next contributor inherits a suppression whose
subject they cannot locate. The other builders in that file already use patterned ids
(`f"{name}aaaa…"`, `"f"*32`, `f"{i:032d}"`) and none of them trips the detector; this one was the
odd case, so it was made to match. `.secrets.baseline` is unchanged — no new suppression exists.

Commented at the site, because the reason a test id avoids hex is not self-evident and the obvious
"tidy-up" is to put it back.

**Verified:** the full hook set passes over the change set (ruff · ruff format · bandit ·
detect-secrets · hygiene), and `.secrets.baseline` shows no diff.

---
## 2026-08-20 (5) — extraction checked against committed documents, and the third defect that found

**What changed.** New `tests/fixtures/documents/` holding a real `treatise.epub` and a
hand-authored `article.html`, a `make_fixtures.py` regenerator, a README, and
`tests/unit/test_extraction_fixtures.py` (28 passing, 4 xfailed). `.gitattributes` gains
`*.epub binary`.

**Why, and it is the gap the morning's work admitted to.** `test_extractors_formats.py` builds each
fixture with the same library that reads it back, and says so in its own docstring: a round-trip
cannot prove anything about files from *other* producers. These files are **frozen artifacts** —
once committed they stop tracking what `ebooklib` emits today, so the assertions keep describing
what a real file on disk does.

**They immediately found a third defect, and it is the worst of the three.** `get_text(separator=
"
")` breaks a line at **every inline tag boundary**, so `Emphasis <em>inside</em> a sentence`
arrives as three lines. It affects EPUB and HTML alike. Scientific prose italicises constantly —
gene names, species, emphasis — so this shatters sentences across the corpus, degrading both the
embedding and the BM25 token stream, and **no health check would ever flag it**: every character is
still present. The two already recorded in entry (1) are confirmed here on real files rather than
inferred: the EPUB nav document lands as trailing prose under a heading repeating the book title,
and the HTML `<title>` lands as the first line, above the real `<h1>`.

**All three are pinned as `xfail(strict=True)` — the first use of xfail in this suite.** Asserting
the broken output would cement the wart as the contract; a plain comment rots. `strict=True` makes
the test **fail when the bug is fixed**, so whoever fixes it is told to come back and update the
expectation. None is fixed here because each changes extraction *content* and therefore `doc_hash`
for every affected document (ADR-042) — decisions, not patches.

**`strict=True` earned its keep within the hour: it caught a bad test of mine.** The first draft of
the sentence-fragmentation test normalised newlines away before asserting, which rejoins the
fragments — so it passed against the broken output and XPASSed loudly instead of sitting there
green and worthless. It now asserts per line.

**Two smaller things the fixtures forced.** `*.epub binary` in `.gitattributes`: `* text=auto`
decides by heuristic, and a zip that gets line-ending normalisation is an unreadable file — the
suite also asserts the committed container still starts with `PK`. And the `&minus;` sign in the
HTML fixture is referred to in the test as `chr(0x2212)`, because a literal U+2212 is exactly what
ruff's RUF001 confusables rule rejects — the same rule that stopped `test_docs_encoding.py` quoting
its own evidence.

**Rejected alternatives.** *Generating these at test time too* — that is the existing file, and it
is the thing this one exists to complement. *Asserting current (broken) behaviour* — it reads as
approval and gives a future fixer nothing. *Fixing the three defects here* — they change every
affected document's identity; that belongs with a re-ingest plan, not a test commit.

**What it opens.** A decision on all three defects, which is now a single conversation with three
tripwires attached. Also noted while reading real output: bs4 raises `XMLParsedAsHTMLWarning` on
EPUB chapters (`extractors.py:152` parses XHTML with the `lxml` HTML parser), and both extractors
emit long runs of blank lines that nothing collapses.

---
## 2026-08-20 (4) — the CLI's formatters and the eval harness's one adapter

**What changed.** New `tests/unit/test_commands_formatters.py` (42 tests) and
`tests/unit/test_eval_adapters.py` (21 tests). `commands.py` was **16%** — the largest single
uncovered block in `src/` at 215 lines — and `eval/adapters.py` **26%**.

**`commands.py` is the CLI's entire user-facing surface and is almost all pure.** `test_commands.py`
covered `parse_command` and a few `execute_command` branches; every formatter had nothing. The
interesting behaviour in them turns out to be **truncation** — six independent caps (authors 50,
titles 80, keywords 10, citation snippets 120, external references 30, graph nodes 25), each
deciding how much of the truth the user sees. Where the code reports what it hid, that report is
now asserted; where a cap changes the output shape entirely, both sides are. Also pinned: health
grouping is worst-first (a broken document must not sort below a healthy one), a null health lands
in `unknown` rather than raising, absent optional fields are omitted rather than rendered as empty
labels, and the mermaid graph escapes double quotes in a filename — an unescaped one ends the label
early and corrupts the whole diagram.

**`eval/adapters.py` is small but structurally special.** Its own docstring calls it *the only
module in `doc_assistant.eval` that depends on the rest of `doc_assistant`*, so it is the seam
Feature 5 cuts along. The two things it owns are both silent-corruption risks rather than crashes:
a **fresh `TokenCounter` per query** (a leaked one would make every row after the first overstate
cost while still looking plausible) and the **citation list** that `citation_overlap` — the
harness's *zero-variance* scorer — reads. Deduplication, first-appearance order, skipping chunks
with no filename, and `0 tokens -> None` (unknown cost, not free) are now all asserted.

The pipeline is a hand-rolled ~20-line stub, not a `Mock`: it documents the shape an adapter must
present, and unlike a `Mock` it fails when a refactor changes that shape instead of accepting any
call at all.

**Rejected alternative.** *Testing `execute_command`'s remaining branches* — they reach SQLite,
DuckDB and config, so they are integration tests wearing a unit test's clothes. The formatters they
delegate to are where the logic and the line count both are.

**A tooling trap worth writing down.** `pytest --cov=doc_assistant.eval.adapters` (dotted target)
**fails at collection** with `ModuleNotFoundError: No module named '_duckdb._sqltypes'`. Coverage
imports the named module very early, which pulls the `doc_assistant.eval` package and `store.py`'s
top-level `import duckdb` into a partially-initialised C extension. duckdb 1.5.3 imports fine
normally and the full suite passes; only that invocation trips. Use `--cov=src/doc_assistant`
(path form) — which is what the full run uses — and do not go hunting for a duckdb bug.

---
## 2026-08-20 (3) — the frontend's largest untested module, tested with the runner already in the repo

**What changed.** New `apps/desktop/src/lib/graph/forceLayout.test.ts` (19 tests). The frontend
suite goes **108 to 127**, still `node --test`, still zero new dependencies.

**Why this module.** At 189 lines `forceLayout.ts` was the largest untested file in the frontend,
and its own header states the case for testing it: *"the layout is the risk; determinism is the
safety net"*. The safety net had nothing checking it. It is also pure — node ids in, a
`Map<string, Point>` out, no DOM — so it needs none of the harness the repo lacks.

**Asserted as properties, not as a golden snapshot.** Every claim the module makes about itself in
prose is now a test: identical input gives identical positions; a different `seed` gives a
different layout (without that second half the first would also pass if the seed were ignored and
the layout were merely constant); no coordinate is ever non-finite, including the dense all-pairs
case the EPS floor exists for and a 40-node graph past the spec's 21-node hub; every node lands
inside the padded box, and a larger padding demonstrably shrinks the occupied area; unknown edge
endpoints are ignored, so an unfiltered edge list lays out identically to a pre-filtered one, which
is what the docstring invites callers to rely on. A snapshot of 300 cooled iterations would break
on any tuning change while proving none of that.

**The degenerate cases are the robustness contract in the graph**: zero nodes returns an empty map
instead of throwing, one node is centred, one node with a self-edge stays finite, and two nodes
never end up coincident (they would render as a single dot). Exactly one behavioural claim is made
— a bonded pair settles closer than an unbonded one — and it is kept deliberately weak, because
anything tighter is a snapshot of the cooling schedule wearing a property's clothes.

**Rejected alternatives.** *Adding vitest + jsdom + @testing-library/svelte first* — it is the
right next step and the standing gap is real (**39 `.svelte` components cannot be tested at all**
under `node --test`, which is why yesterday's CSS source-order bug and the four-read-path merge bug
were both caught by eye rather than by a test), but it is a tooling decision with a lockfile
change, and it should not ride along inside a test-writing commit. `forceLayout` needed none of it.
*Testing `conversations.svelte.ts` in the same pass* — it is the other module worth covering (it
owns the new `archiveConversations`), but it is `$state`-backed and fetch-driven, so it wants the
harness decision made first.

**What it opens.** The component harness question, unchanged and now the largest single gap in the
project's testing. Also worth noting for whoever takes it: the frontend runner is
`node --test src/**/*.test.ts`, so a test file placed outside `src/` runs nowhere and reports
nothing.

---
## 2026-08-20 (2) — the two other untested user-facing read paths: the chat router and the citation graph

**What changed.** New `tests/unit/test_query_router.py` (45 tests) and a `graph_subgraph` section
appended to `tests/unit/test_library.py` (9 tests). `query_router.py` goes **29% to 100%**;
`library/citations.py` **73% to 99%**. Same ranked coverage pass that produced the extractor work
above; these were the next two entries on it where the uncovered code is read by a user.

**`query_router` was the lowest-covered module that runs on a user request.** `ChatController`
consults `is_library_query` on *every* message (`chat_controller/controller.py:305`); a match
short-circuits the entire RAG pipeline and `answer_library_query`'s string is shown verbatim.
Neither half had a test.

The load-bearing part is the negative lookahead `_NOT_TOPICAL`, which is what keeps *"show my
papers about RAG"* out of the metadata branch. If it regresses, a content question silently
receives a document count — no exception, no log line, no way to notice except reading the answer.
**The test asserts both halves of that difference**: the bare phrase must match *and* the
qualified one must not. Asserting only the rejection would keep passing if the pattern stopped
matching altogether, leaving a green test guarding nothing — which is the failure mode that made
KI-41's chunking sweep compare one configuration with itself six times.

Also now pinned: the empty-library branches (`answer_library_query` on 0 documents names
`data/sources/` rather than returning a bare zero), the "documents exist but have no add dates"
case that refuses to nominate a latest, the `and N more` truncation on long broken lists — a
silent cap would read as *these are all of them* — and, over every phrasing the router claims,
that an answer actually comes back. That last one is the contract *between* the two functions,
which is where a blank reply to the user would come from.

**`graph_subgraph` had 0% coverage** and is one of the four surfaces KI-45 names as still trusting
`Citation.target_document_id` unguarded — the References block re-checks resolutions at read time,
the citation graph does not. It is also the only traversal in the module that walks both
directions and dedupes. Pinned: exactly one centre, both directions at depth 1, unresolved
citations excluded (a bibliography is mostly external — 36 references and 0 library matches on the
recovered scan), `depth` actually bounding the walk, repeated citations collapsing to one edge
(nine of one document's eleven resolutions pointed at two papers), termination on a cycle, and an
unknown id returning an empty graph instead of raising into the panel.

**One line is deliberately left uncovered.** `citations.py:343` — the `if nid in visited: continue`
guard — is unreachable as the traversal is written: a node only enters `next_frontier` when it is
not already in `nodes`, and every visited node is in `nodes`, so no node can be queued twice. It is
defensive, and deleting a defensive guard to buy a coverage point is the wrong trade. Recorded here
so the next coverage pass does not re-litigate it.

**Rejected alternatives.** *A new `test_citations_graph.py`* — `temp_database` is defined locally
in `test_library.py`, and copying a 40-line engine-patching fixture to a second file is how the two
copies start drifting (the same argument that made `effective_metadata` one function yesterday).
*Patching `doc_assistant.library.list_documents` for the router tests* — `query_router` binds both
names at import, so patching the source module leaves this module's references untouched; the
patch targets `doc_assistant.query_router`, the trap `chat_controller/__init__.py` already
documents.

**What it opens.** The router's regexes are still English-only and pattern-based; the tests pin the
behaviour, not its adequacy. And `graph_subgraph` is now tested but still *unguarded* — KI-45's
false resolutions reach it exactly as before. A test that pins current behaviour is not a fix, and
the write-side precision fix is still the thing that matters there.

---
## 2026-08-20 — five of the seven supported formats had no extraction test at all; now they do

**What changed.** New `tests/unit/test_extractors_formats.py` (37 tests). `extractors.py` goes
from **44% to 91%** line coverage. The only lines still uncovered are `extract_pdf_pymupdf`'s body
(92-103, 262), which needs a committed PDF.

**Why, measured before writing anything.** A coverage pass over the whole suite (1,839 tests, 89%
overall) put `extractors.py` at the bottom of the ranked list with **lines 141-220 never executed
once** — that range is the entirety of `extract_epub`, `extract_html`, `extract_docx`,
`extract_rtf` and `extract_odt`. `test_extractors.py` covered format *detection*, `.txt`/`.md`, and
the PDF placeholder strip; five of the seven extensions in `SUPPORTED_EXTENSIONS` had no test
touching their extractor. That is the app's front door, in the subsystem with the most tracked
defects (KI-14/26/34/40/42/43/44/46/47 — nine issues, three of the five currently open).

**What the tests assert**, beyond "output is non-empty": the structure heuristic per format (DOCX
style names to `#`/`##`/`###`; the EPUB Dublin Core title leading as an H1; HTML heading
conversion), the one content decision `extract_html` makes (script/style/nav/footer are
decomposed — stated per tag, so a future edit to that tuple fails on the tag it dropped), RTF
control-word stripping and ANSI escape decoding, dispatch through `extract_to_markdown` for all
five, `.htm` and `.html` reaching the same extractor, and `get_format_status`'s advisories.

**Every format asserts a non-ASCII round-trip**, because that is the failure this project has
already shipped: nothing on Windows defaults to UTF-8 (CONTEXT.md section 9), four tracked docs
were committed double-encoded, and an extractor that mangles an accent produces garbage that still
*reads* as prose — so it passes ingest, reaches the chunk store, and nothing downstream notices.
`test_html_reads_as_utf8_regardless_of_the_ansi_codepage` pins the explicit `encoding="utf-8"` on
that read specifically.

**Two defects surfaced while writing the fixtures. Neither is fixed here** — they are extraction
*content* decisions, not test bugs, and fixing them changes `doc_hash` for every affected document
(ADR-042), which is not a thing to do inside a test commit:
- **EPUB pulls the navigation document into the body.** `get_items_of_type(ITEM_DOCUMENT)` includes
  the generated nav/TOC item, so a book's markdown ends with its own table of contents as prose.
  The tests assert title, headings and body are present; they deliberately do **not** assert the
  nav text is absent, so the wart is not cemented as expected behaviour.
- **`extract_html` leaks `<head><title>` into the text.** Only script/style/nav/footer are
  decomposed, so `soup.get_text()` emits the title as a bare leading line, indistinguishable from
  body prose. Same shape as the page-furniture problem behind the keyword layer's singleton rate.

**Rejected alternatives.** *Committed binary fixtures* — better at catching producer variance, but
an opaque blob in a public repo, and every writer library here (`python-docx`, `odfpy`,
`ebooklib`) is a **base** runtime dependency, so an in-test fixture can never skip for a missing
library and shows the reader the input beside the required output. HTML and RTF are hand-authored
markup, so those two are producer-independent regardless. *Conditional skips per format* — the
suite has 8 skip markers and all 8 are environment-conditional; a skip here would hide exactly the
breakage the file exists to catch. *A literally empty EPUB body for the zero-content case* —
`ebooklib` cannot serialise one (lxml raises "Document is empty" building the nav), so it would
have tested the fixture builder; a whitespace-only paragraph tests the extractor.

**What it opens.** These are round-trips through the same library that wrote the file, so they
cannot prove the extractors survive files from *other* producers — the variance that broke tier-1
citation parsing (KI-45). Said explicitly in the module docstring rather than left implied. Closing
the PDF gap (92-103) needs a committed fixture and is the same binary-in-a-public-repo decision,
deferred. The two leaks above want an issue each before anyone re-ingests.

---
## 2026-08-19 (3) — a metadata override applied in the Library grid and not on the document's own page

**What changed.** ADR-013's `override ?? auto` merge now lives in one function,
`effective_metadata`, and **all four read paths that display a document use it**:
`list_documents` (which already merged, inline), `get_document_details`, `get_document_chunks`
(the document page's own header) and `list_document_figures`. Three of the four did not.

**How it surfaced.** Correcting the OCR-derived title on the recovered scan
(*"A Revised Neuroanatom of Frontal—Subcortical Cireuits"* → *"…Neuroanatomy of
Frontal-Subcortical Circuits"*) wrote the override correctly and changed nothing on the page it
was edited on. The list endpoint returned the corrected title; the detail endpoint returned the
extracted one. `list_documents` merged inline, `get_document_details` read `doc.title` straight —
the same rule written twice, and the second copy was simply missing.

**Worth noting about the write path: it was already right.** The PATCH cleared `year_override`
because the submitted 2001 equalled the extracted 2001 — ADR-013's "an effective value equal to the
default clears that field" — so only title and authors persisted. The bug was entirely on the read
side, which is why editing appeared to half-work rather than fail.

**One path deliberately does NOT merge.** `document_years` feeds the year-aware epistemics rule
(G3), which is an analysis of what the corpus *says*, not a display of what the user prefers to
see. Letting a metadata edit move a `superseded_trend` verdict would change a knowledge-layer
result for a reason no baseline records — that is a decision with an eval behind it, not a
consistency fix. Said in a comment at the call site and pinned by a test.

**Guarded by eight tests**, the load-bearing one being *every display surface agrees on the title*
— stated once over all four paths, so a fifth surface added without the merge fails in the suite
rather than in a screenshot. Also pinned: no override means both report what
extraction found (the override stays additive), and a title-only override leaves authors and year
extracted.

**Rejected alternative.** *Merging in the detail function too* — that is the fix that created the
bug: two copies of one rule, drifting the moment someone touches one. The helper is the point, not
the call site.

**What it opens.** Nothing else reads `Document.title` directly for display — checked — but the
same shape exists wherever an additive sidecar has a "merge for display" step. Also unresolved: the
extracted values under the override are still the OCR ones, which is correct (ADR-043 keeps
received content verbatim; the override is a separate, inspectable layer) but means a re-extraction
never silently corrects them.

---
## 2026-08-19 (2) — chat select mode reviewed: the tick was stacked ON the row by a CSS ordering accident, and the reported "connection issues" were another project's dev server

**What changed** (`Sidebar.svelte`, `App.svelte`, `conversations.svelte.ts`), from a live review:

- **The tick now sits left of the title, email-client style.** It had been rendering *above* the
  row and centred, clipping names. The markup was always right — the tick precedes the body in the
  DOM — and the cause was CSS: the element carries both `.pickrow` (`display:flex`) and `.rowmain`
  (`flex-direction: column`, for the normal row). Equal specificity, so **source order decided it**,
  and `.rowmain` is defined 60 lines later. Fixed as `.rowmain.pickrow` — a two-class selector, so
  the row layout no longer depends on where either rule sits in the file. Verified in the running
  app: `flex-direction: row`, tick at x=15 and body at x=39, vertically centred, and **all 100 rows
  share one left edge**, with titles truncating on `text-overflow: ellipsis` instead of being cut.
- **The action bar is sized like a toolbar** (0.5rem padding, 0.8rem type, 41px tall against ~24px)
  — it is the only way out of a destructive mode and it read as a caption.
- **"Done" is gone.** The header toggle already reads *Leave select mode* and does the same thing;
  two controls for one action, and the redundant one sat where a confirm button would.
- **Bulk archive added**, mirroring bulk delete. It is a *toggle*: when every ticked chat is already
  archived the button reads **Unarchive**, which is what the "Show archived" view needs. Applied
  immediately rather than through a confirm — archiving is reversible and the toggle right below it
  brings them back (delete keeps its confirm). `archiveConversations` PATCHes concurrently and
  refreshes the list **once**: N single-archive calls would each refresh, rerendering the sidebar
  under the user mid-action. A failure is logged per conversation and does not abandon the rest.

**Verified end to end in the running app, and the data left exactly as found** — ticked one chat,
Archive, confirmed archived server-side; re-selected it under "Show archived", confirmed the button
now read *Unarchive*, clicked it, confirmed 100 conversations / 0 archived again.

**The "connection issue inside library" was not a defect in this app.** Reported as
`TypeError: Failed to fetch` on connections, chunks, figures and references. The API was healthy
throughout (200s in ~14 ms directly on 8001). **Port 1420 was being served by a different
project's dev server** — the page at `localhost:1420` is titled "Scribe", and
`localhost:1420/api/health` returns **404 with that project's HTML**. Provenote's Vite owns `/api`
as a proxy to 8001; with the port taken (and `strictPort: true`), an already-open Provenote window
keeps rendering while every later `/api` call goes to a server that has never heard of it. Same
cause for the graph panel. Re-verified on the alt port from `.claude/launch.json`: connections,
figures, references and the document payload (66 parent blocks) all 200.

**Two diagnostic notes worth keeping.** `curl http://127.0.0.1:1420` reported nothing while
`localhost:1420` answered — Vite binds `::1` only, so an IPv4 probe says "dead" about a server that
is running; check both before concluding a process is down (I concluded it once and was wrong).
And the *shape* of the error was the tell: `Failed to fetch` is connection-level, so it points at
who is answering the port, not at the handler.

**Rejected alternatives.** *Setting `flex-direction: row` on `.pickrow` alone* — it would work today
and break again the next time a rule is appended below it. *A checkbox `<input>`* — the whole row is
the target on purpose (a 15px hit area in a list is poor), and the tick is presentational. *A
confirm for bulk archive* — reversible actions that ask twice train people to click through
confirms. *Leaving "Done"* — the user's point stands: the escape hatch already exists above it.

---
## 2026-08-19 — the corpus's one broken document is recovered (96 → 97 retrievable), and the reason it was broken is not what it looked like

**What changed.** No code. `middleton-2001.pdf` — 15 pages, `extraction_health='broken'`,
`chunk_count=0`, the only unhealthy document in the library — is now **healthy, 52 chunks, and
retrieved rank 1** on three of the RG-025 baseline's queries against the full 97-document library.
`data/library.db.bak-20260819-preocr` is the pre-change backup.

**The finding is the trigger, not the text.** RG-025 measured this document's OCR a fortnight ago
and concluded the text was worth retrieving; that was never in doubt. What is new is that **the
shipped extractor produced it on its own**: same code, same pinned dependencies, same PDF, 0
characters on 2026-08-08 and **34,600 today**. PyMuPDF4LLM finds a `tesseract` binary on `PATH` and
OCRs pages with no text layer. Nothing in the repo changed — the box's `PATH` did. Filed as
**KI-47**, because extraction output is supposed to be a function of things the cache fingerprint
can hash, and an external binary is not one of them: the stale 349-byte extraction stayed "fresh"
forever with a fingerprint that still matched the live one to the digit. **The installer ships no
OCR** (`tesseract` is a system binary; the spec, README and setup docs never mention it), so the
app's real behaviour on a scan is still the 0-character one and this box is the outlier.

**A second defect fell out of the fix.** `doc_hash` hashes the extracted *text*, so recovery mints a
**new document identity** — and the old row survived a full cleanup pass, leaving the file in the
library twice. `cleanup_orphans_sqlite` builds its candidate set from **Chroma metadata**, so a
document with **zero chunks is invisible to the orphan sweep** and its row can never be classified
stale. That is the shape of every recovery, not a one-off: a broken document has no chunks by
definition. Filed as **KI-46**; the stale row was removed with the same `session.delete(row)` call
cleanup uses (FK cascades drop the outbound enrichment), guarded on `chunk_count == 0` plus a
healthy sibling — **not** through the app's delete path, which would have sent the source PDF to the
Recycle Bin (ADR-014).

**One measured improvement over the 2026-08-08 run, unexpectedly.** That baseline's single
highest-value carried item was *de-hyphenate line-broken words* — raw Tesseract left **88** of them
(`cortico-` + `spinal` never matching a query for `corticospinal`). This path leaves **0**: mean
line length 250 characters, 3 lines ending in a hyphen, and `corticospinal` present intact. The
reflow defect that motivated the work does not occur here.

**Rejected alternatives.** *`--rebuild`* — it wipes the vector store and re-embeds all 97 documents
to fix one. *Hand-running Ghostscript + Tesseract and pasting the text in* (what the baseline did)
— unreproducible, and it would put unmarked OCR prose in the library by a path no runner owns.
*`delete_document`* — bins the source file. *Fixing the cleanup blind spot inline* — a real change
to a core ingest path, which deserves its own tests rather than riding on a data repair.

**What it opens.** The corpus is now **97 retrievable documents, digest changed** — so by the
comparability layer's own rules every earlier run (96 documents) is *not comparable* to anything run
from here, and `compare_runs` will now say so instead of leaving it to memory. ADR-039 is still
**proposed**: what happened today is recovery that is neither opt-in nor marked as OCR-derived,
which is exactly what that ADR wanted to avoid — recovered text now enters as ordinary prose that
retrieval and citation cannot distinguish from a real text layer.

---
## 2026-08-18 (2) — a committed baseline now carries its own evidence, so it can be checked without the run store

**What changed.** New `eval/baseline_doc.py` + `scripts/emit_baseline.py` (+ `just emit-baseline`)
write a baseline document **from the run record**, and `compare_runs --against <file>` checks a
later run against that document.

**The gap.** `tests/eval/baselines/` is the committed reference record; `data/eval.duckdb` is
gitignored. So the numbers travel and the evidence does not — a fresh clone holds ~30 documents
whose setup sections were typed by hand, and nothing can contradict them. That is exactly where the
last error hid: the Haiku-vs-llama split across the 2026-08-08 arms lived only in prose, and prose
is recoverable by a human reading the right file and by nothing else.

**What the emitter writes, and what it refuses to write.** Settings, corpus composition, generator,
judge, and the aggregate table — copied from `config_json`, never re-derived — plus a visible fenced
JSON provenance block (visible, not an HTML comment: the reader should see exactly what the checker
reads). It **refuses to emit from runs that are not one experiment**, because a baseline averages
its trials and mixing two would present them as one number; the refusal prints the comparability
report that explains why, and exits 4. It writes `TODO` for the judgement, because the caveats are
what make a baseline worth keeping and no emitter can derive them. A key the runs disagreed on is
**dropped**, not taken from the first trial, and then renders as "not recorded" — the document
cannot vouch for it.

**Older baselines parse to nothing, on purpose.** `parse_provenance` returns `{}` for a document
with no block, a malformed block, or a JSON block that is not provenance — so `--against` an
existing hand-written baseline reports *unknown across the board* and says the document never
recorded those facts. Verified against the real folder: every committed baseline parses without
raising, and the check against `chunking_sweep_private_2026-08-08.md` names 20 unrecorded settings
rather than inventing agreement.

**Verified end to end, $0.** Emitted a baseline from the two identical-settings validation runs,
then checked runs against the committed file: the matching run exits **0**, the qwen2.5:3b run exits
**4** with `contains_all` not comparable and `citation_overlap` still fine, and the hand-written
baseline path exits **3** with the "carries no provenance block" note. 27 new unit tests.

**Rejected alternatives.** *An HTML-comment block* — hidden from the reader, which invites the
document and its record to drift; this layer exists because a claim and its evidence drifted apart.
*Emitting the caveats too* — an emitter can produce a table, not a judgement, and a baseline that is
only numbers is what the project already has too many of. *Taking the first trial's value when
trials disagree* — it would state something no reader could act on. *Raising on a baseline with no
provenance* — a checker that rejects the entire existing corpus of baselines is a checker nobody
runs; `{}` and an honest "unknown" is the useful behaviour. *Back-filling provenance into the ~30
existing documents* — the same rule as everywhere else here: an inference must not become
indistinguishable from a recording.

**A document that argues with itself is caught too.** `table_drift` compares the visible results
table against the document's own provenance block and reports any row where they disagree — the
small nasty failure where someone tidies a number in the pretty table, or copies a document and
edits it, while the machine block still carries the original. Verified by tampering with an emitted
baseline: the check names the scorer and both values before the verdict, because the reader needs
to know which half of the document they have been quoting.

**What it opens.** The existing baselines stay unpinned unless someone re-runs and re-emits them,
which is a deliberate non-goal — their numbers are still valid readings, they simply cannot be
machine-checked. No gate *requires* a new baseline to carry a block, so the discipline is still a
habit rather than a rule; and drift is only checked on the mean, not on the std or the counts.

---
## 2026-08-18 — the harness now answers "may these two runs be compared?", per scorer, and says UNKNOWN when nobody wrote it down

**What changed.** Yesterday's entry made a run record what it measured. This makes something *read*
that record. New `eval/comparability.py` (generic, no app import — it travels with the harness),
plus `report.compare_runs` / `format_comparability`, a `scripts/compare_runs.py` CLI, a
`--baseline RUN_ID` flag on `run_eval`, and `just compare`.

**The idea it encodes, which this project had already written by hand twice.** A score depends on a
*prefix* of the pipeline: cases → index → retrieval → generation. `citation_overlap` and
`figure_retrieval` read the **retrieved documents** (`output.citations`, `output.raw["retrieved"]`,
both filled from `pipeline.retrieve` before a token is generated), so they survive a generator swap.
Everything else reads `output.answer` and does not. That asymmetry is exactly what localised RG-029
— `citation_overlap` reproduced to the digit while `contains_all` moved — and it is the reasoning in
the generator caveat at the top of `chunking_sweep_private_2026-08-08.md`. A hand-written caveat
protects only the reader who opens the right file, so now one differing setting yields **per-scorer**
verdicts rather than a wholesale pass/fail.

**Three states, and UNKNOWN is the common case, not the corner one.** Of the 75 runs in the live
store, **not one** records its generator or its corpus. So the honest verdict for almost every
historical pair is *unknown*, and the layer says so: an unrecorded setting is never assumed to have
matched, and it is never inferred — not from the run's `note` prose, not from a sibling run. The
five annotated Haiku trials carry their generator in `note` precisely so that an inference cannot be
mistaken for a recording, and `--list` shows them as "not recorded" for the same reason.

**Verified against real runs, $0.** Three one-case runs on the live 96-document index over local
Ollama, two of them differing *only* in generator:

| | `contains_all` | `citation_overlap` | verdict |
|---|---:|---:|---|
| llama3.1:8b vs llama3.1:8b | 0.500 → 0.500 | 1.000 → 1.000 | **comparable**, exit 0 |
| llama3.1:8b vs qwen2.5:3b | **0.500 → 1.000** | 1.000 → 1.000 | **not comparable** on `contains_all`, comparable on `citation_overlap`, exit 4 |

The second row is RG-029 reproduced deliberately: a 3B model "beats" an 8B one by +0.500 on the
answer score while the retrieval score does not move a digit. Also verified on the real 75-run
store, where every pair comes back UNKNOWN with the missing keys named.

**`--varying` is the half that makes it useful to a sweep, and it catches KI-41 from the record.**
A sweep *intends* to differ in one setting, so a bare comparison would object to the experiment
itself. Declaring the independent variable (`--varying child_chunk_size`) stops that difference
blocking while **everything else still blocks** — which is the useful direction, because a sweep's
real risk is that something besides the grid moved. The opposite failure gets its own field and its
own exit code: a declared variable that came back **identical** means the arms are one configuration
compared with itself, which is KI-41 exactly (the 2026-06-06 sweep drove its grid through
environment variables `.env` silently overwrote). Verified on two real runs: verdict `ok`, and a
banner above the table saying the declared variable did not change, exit `5`. Note the deliberate
split — those runs *are* comparable; it is the experiment between them that is void, so it must not
be folded into the comparability status. `sweep_chunking`'s preflight catches this before a run;
this catches it afterwards, from the record, where a preflight-less runner cannot hide it.

**Two smaller things fell out of building it.** `judge_provider` / `judge_model` are now recorded —
`llm_judge` grades with a model that is *not* the generator, so two runs can share a generator and
still have been graded differently; it is the same membership rule as yesterday's keys, and its
absence would have been an invisible hole in exactly the scorer that costs money. And
`Store.resolve_run_id` now owns prefix resolution, so `run_eval --baseline` and `compare_runs` cannot
resolve an id two different ways.

**Rejected alternatives.** *A plain config diff* — it would flag `trial_index` (which differs by
design between trials of one experiment) and would report a generator swap as invalidating
everything, including the one number that survives it; the per-scorer prefix is the whole value.
*Treating an unrecorded setting as unchanged* — that is precisely the assumption RG-029 was, and it
would return the store to printing two means side by side. *Parsing the `note` prose* to recover the
five attributed Haiku trials — the annotation exists because a back-filled inference must stay
distinguishable from a recording. *Suppressing the score tables on a NOT COMPARABLE verdict* —
informing beats blocking; a suppressed table sends the reader to the raw store, where there is no
verdict at all. *Comparing every trial against the baseline under `--repeat N`* — the trials share
their settings by construction, so it would print the same verdict N times. *Folding an
ineffective variation into the comparability status* — "these runs cannot be compared" and "these
runs are identical when they should not be" are opposite diagnoses with opposite fixes.

**What it opens.** RG-021 is now closed end to end (record + warn). Not built: nothing *emits* a
baseline markdown from a run record, so `tests/eval/baselines/` stays hand-written; nothing compares
a run against a *committed baseline file* (only against another stored run); and the 75 historical
runs remain permanently unknown, which is a fact about them, not a gap to fill. `sweep_bm25_weight`
still persists nothing, so it participates in none of this.

---
## 2026-08-17 — an eval run now records the corpus it measured and the generator it used, and a paid one says so before it spends

**What changed.** Three keys and a flag, closing the "make the run record and the run agree" class
that RG-029 opened and RG-021 kept open.

- `sparse_index.doc_set_digest(doc_hashes)` — SHA-256 over the sorted, de-duplicated `doc_hash`
  set. A *corpus* identity, deliberately not `fingerprint`, which identifies a *build*: chunk ids
  move on every re-ingest, so it cannot answer "same documents?" — the question a chunking sweep
  needs answered while the geometry is changing under it.
- `RAGPipeline.indexed_doc_hashes` — every document retrieval can currently reach, read from the
  keyword arm because that arm **is** the BM25 corpus (same `keep_for_retrieval` filter as the
  vector arm). Three states: the hashes, `set()` for an empty corpus, `None` only when a build
  failed over a non-empty corpus and this process genuinely never learned what the store held.
- `scripts/run_eval.py` records `index_doc_count` + `index_doc_digest` (RG-021) and
  `llm_provider` / `llm_model` **as the run actually ran them**, and grows `--provider` /
  `--model`. A paid generator now trips `llm.assert_provider_intent` — the same banner + 3 s abort
  window every enrichment runner has had since Feature 7.

**Why the composition comes from the pipeline and not from `run_defining_settings`.** Config does
not know the corpus; it is state on disk. Asking the live pipeline is also the only source that
cannot drift from `pipeline.py`'s own "which collection is active" rule — a second copy of that
resolution in the harness edge is exactly the restated-gate failure `sweep_chunking`'s preflight
docstring warns about. The cost is the usual injection cost: a new runner must record it, so
`scripts/CLAUDE.md` now carries that rule beside the `settings_provider` one.

**Why `--provider` refuses to change provider without `--model`.** `.env` pairs `anthropic` with
`claude-haiku-4-5-…`; inheriting that model under `--provider ollama` hands Ollama a name it has
never heard of, and it fails once per case, minutes in. The error names the model it refused to
inherit and the flag that fixes it.

**`synthesis_mode` was asked for and is deliberately NOT recorded — the finding is bigger than the
key.** `ai` vs `human` does change the answer, but not this harness's answer: `eval/adapters.py`
calls `pipeline.retrieve` + `pipeline.stream_answer` directly, and `stream_answer` is a bare
`ANSWER_PROMPT | llm` chain. `SYNTHESIS_MODE` is read only in `chat_controller.helpers`, which the
eval path never enters. Recording it would pin a setting the run did not honour — RG-029 with the
sign flipped. The consequence worth carrying: **the eval harness measures the raw answer path, not
the shipped one** (no synthesis split, no provenance, no reviewer). Written into
`eval/run_settings.py`'s docstring, where the membership rule lives.

**Verified, $0.** One case over the live corpus on `llama3.1:8b` into a scratch DuckDB:
`config_json` came back with `index_doc_count 96`, a 64-char digest, and `llm_provider ollama` —
which is proof the override reached the record, since `config.LLM_PROVIDER` still reads
`anthropic`. The paid banner was confirmed on the default path by killing the process inside its
own abort window, before the pipeline loaded. Unit tests: corpus digest (order/duplicate
independence, one-extra-document sensitivity, and that it survives a re-chunk where the build
fingerprint does not), the pipeline's three states, and both CLI helpers.

**The first thing the new key found.** The index holds **96** documents where `library.db` holds
**97**: `middleton-2001.pdf` has `chunk_count = 0` and `extraction_health = 'broken'`, so it has
never been retrievable. Every eval run on this box has been over 96 documents, and now says so.
Not a regression and not caused by this change — but it is the shape RG-021 predicted, found on
the first read, and it is very likely the same document as the carried "one document still
extracting zero keywords" item (no text, no keywords).

**Rejected alternatives.** *Composition from `library.db`* — it is the registry, not the index:
its non-archived filter does not match what retrieval can reach (archiving hides a document from
the UI, not from BM25), and it would have reported 97. *Composition inside
`run_defining_settings`* — it would reach the sweeps for free, but that function is a pure
config snapshot called on every `persist_run`, including in unit tests and in the sweep's probe
subprocess; giving it disk I/O makes the suite depend on machine state and CI has no index at
all. *Recording `synthesis_mode` anyway* — see above. *A bare `nosec`-style default flip to
`ollama`* — moving the default is a policy change with an eval consequence; the flag plus the
banner makes the current default impossible to trip over accidentally, which was the actual
defect.

**What it opens.** RG-021's **build** half is done; its **warn** half is not — nothing yet
compares a run's composition against the baseline it is being read beside, and the 70 historical
runs carry no composition to compare against (same unattributable-history problem RG-029 has).
`sweep_bm25_weight.py` still writes its own baseline markdown rather than persisting runs, so it
records none of this. And `middleton-2001.pdf` needs a decision: re-extract, or accept 96.

---
## 2026-08-16 — `just clean` exists because `cargo clean` would delete the one installer with no copy anywhere

**What changed.** New `scripts/clean_build.ps1` + a `clean *ARGS` recipe in the `justfile`. It drops
`target/debug` in full and everything under `target/release` **except `bundle/`**. Flags: `-DryRun`
(report only), `-Registry` (also drop `~/.cargo/registry/src`), `-Force` (override the guard). Invoked
through `powershell -File` like the existing `app:` recipe, because `set windows-shell := cmd.exe` means
recipe bodies are single commands. ASCII-only per the `.ps1` rule (§9).

**Why `cargo clean` is the wrong tool here specifically.** It wipes all of `target/`, and this repo's
`target/` holds two things that are not intermediates: the current installers in `target/release/bundle`
(3.06 GB at 0.5.1) and a 1.53 GB build-time copy of the frozen sidecar. Of those installers **only the
NSIS `.exe` is attached to the GitHub release — the MSI exists nowhere else on earth**, and `cargo clean`
offers no way to keep it while dropping the rest. Measured on this box today: `target/` was 5.97 GB, of
which 2.89 GB was reclaimable with zero loss. The frozen sidecar in `src-tauri/binaries/` is outside
`target/` and so was never at risk — worth stating, since it is the artifact that costs a PyInstaller
re-freeze.

**Why the active-build guard is scoped rather than machine-wide.** First cut refused to run whenever any
`cargo`/`rustc` lived. On this machine that is close to always — a `cargo build --workspace` and a
`cargo test -p harper-core` from an unrelated project blocked it twice while it was being written, and
during the final verification **24 foreign `rustc` processes** were alive. A guard that always cries wolf
is one you learn to `-Force` past, so it now blocks on what each scope can actually break: a build **in
this repo** blocks the target clean (detected via `rustc`/`link` absolute `--out-dir`, with a fresh
`target/` mtime as the cwd-independent backstop), while **any** build blocks `-Registry`, since that cache
is shared with every Rust project on the box. Both branches were verified live: the target clean proceeded
with 24 foreign processes running, and `-Registry` aborted against the same 24.

**Rejected alternatives.** *Plain `cargo clean` in the recipe* — the whole point is that it takes the
installers and gives nothing back cheaply. *Dry-run by default*, matching `scripts/`' enrichment-runner
convention — a `clean` that cleans nothing is a worse surprise than one that does; `-DryRun` is opt-in and
exempt from the guard instead, so looking is always free. *Registry cleanup in the default path* — it is a
cache shared across projects, so it stays opt-in. *Teaching the recipe to prune installers* — that is
`docs/RELEASE.md` §8's job and it is deliberately manual and deliberately about the **previous** release.

**What it opens.** Nothing deletes an installer without a human: §8 remains the only path. The wider
finding this came from is that **this repo was never the disk problem** — the reclaimable GB on this box
sit in *other* projects' `target/` dirs (~18 GB across two of them), which no recipe here should touch.

---
## 2026-08-15 (3) — the DEVLOG had grown to 8,244 lines because nothing made rotation happen; now a test does

**What changed.** 112 entries (**2026-07-15 → 2026-08-01**) moved verbatim to new
`docs/archive/DEVLOG-archive-002.md`; the live log goes **8,244 → 2,550 lines** and keeps
2026-08-02 onward. New `tests/unit/test_doc_sizes.py` caps the append-only coordination docs —
`docs/DEVLOG.md` 4,000 · `.claude/KNOWN_ISSUES.md` 1,800 · `.claude/RIGOR_TODO.md` 1,500 — skipping
the local-only ones when absent. The DEVLOG header now lists both archives and states the cap.

**Why — the belief that a cap already existed is the interesting part.** `scripts/conventions.toml`
caps the entry file (`entry_max_lines = 600`), each module `CLAUDE.md` (40), and the baton
(`session_max_entries = 10`, cpc ADR-018 rule 11b — the rule that rotated `2026-08-10` out of the
baton earlier today). **None of them covers the DEVLOG**, and `docs_check` implements no DEVLOG rule
at all: it reads the file only as a *route source*. So the log had exactly one rotation ever
(2026-07-21, by hand) and nothing to make the second one happen. **623 KB and 6.4x the next largest
doc in the repo** is the result — reached without any single bad commit, because every entry is
individually small and correct and the growth is only visible in aggregate.

**Why a pytest guard rather than a cpc gate.** The cpc tooling under `tools/conventions/` is
vendored, gitignored and owned by another project (ADR-001/ADR-007) — a cap added there is lost on
the next refresh, and it cannot run in CI because cpc is private while this repo is public. This
repo's own suite is the durable home, which is already why `test_docs_encoding.py` lives there.

**Rejected alternatives.** *Cap by bytes* — lines match `conventions.toml`'s existing
`*_max_lines` vocabulary and are what a reader actually feels. *Cap `.claude/SESSION.md` too* — it
is already capped at 10 *entries*; a second cap in a different unit could fail an entry-compliant
baton, so the two rules would contradict each other (the test says so where someone would think to
add it). *Rotate KNOWN_ISSUES and RIGOR_TODO now as well* — at 1,299 and 1,034 lines they are
under cap with real headroom; rotating them today would be churn, and the guard will say when.
*Keep a whole month live* — that is what produced the problem; two weeks is the working window.

**What it opens.** The caps are calibrated to ~40-60% above post-rotation size, which at this
project's busiest rate (~250 DEVLOG lines/day) is roughly a fortnight per rotation — if that proves
too frequent the answer is a smaller live window, **not a bigger cap**, and the test says so in its
failure message. `KNOWN_ISSUES.md` already has its rotation target
(`docs/archive/KNOWN_ISSUES-resolved-001.md`); **`RIGOR_TODO.md` has none yet**, so the first time
its cap trips someone must create `RIGOR_TODO-archive-001.md` and decide what "closed enough to
archive" means for an RG entry — closed ones like RG-026/RG-028 are the obvious candidates.

---
## 2026-08-15 (2) — an eval run now records which LLM wrote its answers (RG-029)

**What changed.** `eval/run_settings.py` records `llm_provider` + `llm_model` alongside
`embedding_model`; its docstring carries this incident next to the KI-41 one it was written for.
Two guards in `tests/unit/test_eval_run_settings.py`: the generator is recorded, and two runs on
different models are distinguishable. New `RG-029` in `.claude/RIGOR_TODO.md`. Committed `83b730b`.

**Why — a model swap was reading as a pipeline win.** Auditing a 5-trial control run on the private
35 against the committed baselines: `contains_all` **0.822** where the 2026-08-08 control recorded
**0.777**. That is not an improvement — the 08-08 run generated with `llama3.1:8b` and this one
inherited `.env`'s Anthropic Haiku. Nothing in the run record could say so: across all 75 runs in
`data/eval.duckdb`, the union of `config_json` keys contained exactly one model name,
`embedding_model`.

**The tell, and why the diagnosis is trustworthy rather than plausible:** `citation_overlap`
reproduced *to the digit* (0.9363 vs 0.936). Retrieval is deterministic and generator-independent,
so an unchanged retrieval score sitting beside a moved answer score localises the change to
generation. Any reading in which the pipeline improved has to explain why the deterministic half did
not move.

**This is KI-41 one layer up.** KI-41 was a sweep whose arms were not what its note claimed,
invisible because `config_json` recorded no chunk sizes. Here the Haiku-vs-`llama3.1:8b` split
across the 2026-08-08 public and private arms is real and documented — but only in
`evals/README.md` **prose**, recoverable by a human reading a document and by nothing auditing the
data. The module's own membership rule already decided this case: *a value belongs here if changing
it changes what the run measures.*

**Rejected alternatives.** *Retro-fill the 70 historical runs by inference* — their generator was
never written down; "unknown" is the honest state and a guessed provenance is worse than none.
*Also record `synthesis_mode` in the same commit* — it has the identical defect (`ai` vs `human`
changes the answer, so it changes `contains_all`), but it was outside what was asked; flagged for
the next pass instead of quietly widening the change. *Add a `--provider` flag to `run_eval`* —
a real gap, but a bigger interface decision than this fix needed.

**What it opens.** **RG-021 should be fixed in the same pass** — it asks `run_eval` to record
**index composition** so a run over a polluted index is not silently incomparable. Same defect
class (the run record does not pin what the run measured), same file, same one-line-per-key remedy;
with `synthesis_mode` that is three keys and the class is closed. **The five 2026-08-15 trials were
annotated the same day** — their `note` (not `config_json`, which would disguise an inference as a
recording) carries the attributed generator and an explicit "not comparable to the 2026-08-08
private control" warning. The 70 pre-2026-08-15 runs stay unattributable and should remain so; they
were annotatable only via a `.env` read on the same day, and that chain does not exist for them.
**The policy call was taken the same day (user): the private arm stays local-only on
`llama3.1:8b` and the two arms are never compared** — so nothing is re-baselined, the 2026-08-08
local numbers remain the private control, and the Haiku trials are off-policy evidence rather than
a baseline. Written into `evals/README.md`, where the comparison would actually be attempted.
⚠ It is a convention, not a control: `run_eval` has no `--provider` flag and `.env` defaults to
`anthropic`, so the next bare invocation on the private set inherits Haiku again and bills. The
override (`LLM_PROVIDER=ollama LLM_MODEL=llama3.1:8b`) is verified to take; a `--provider` flag is
the durable fix and belongs with the RG-021 pass.
**⚠ Cost:** `--with-llm-judge` was not passed, but `run_eval` *always* generates an answer per case
and has no `--provider` flag, so generation inherits `.env`; those 5 trials billed ~839 K input +
~46 K output tokens for 175 generations.

---
## 2026-08-15 — the shipped v0.5.1 re-tested on a clean box: RG-012 PASS, and RG-028 closes as contention

**What changed.** No code. `.claude/RIGOR_TODO.md`: **RG-028 closed**, RG-012's gate-defect table
gains run 4 and its status line the 2026-08-15 re-pass. `docs/desktop-packaging.md` §5 gains a
log-reading note. Evidence archived to `C:\rg012-host\out-2026-08-15-v0.5.1-run4-PASS-devstack-down\`
(local-only, like the whole harness).

**What was run.** RG-012 Tier-2 against the **published** 0.5.1 installer — the staged copy hashes
`985331FF…`, matching both the bundle and the SHA-256 that was verified against the GitHub asset, so
this tested the bytes users download, not a look-alike rebuild. Windows Sandbox, `python on PATH?
False`: silent install **235 s** → `/api/health` **~40 s** → 3 PDFs → **322 chunks** → one turn,
**16 s**, 10 sources, **5 resolved citations (5 canonical `[n]`, 0 unresolvable)** → **PASS**.
`release_preflight` was green on all nine checks beforehand.

**Why it mattered — RG-028 asked for exactly one number.** The 0.5.1 release runs measured 178/194/78 s
against 14 s (0.4.1) and 17 s (0.4.2) on the same question, corpus, model and box. The entry named
the deciding experiment: re-run **with the dev stack down and Ollama freshly restarted**; near
14–17 s means contention, over 100 s means a real regression in the shipped artifact. It came back
**16 s**. The cause was host contention — my own dev stack sharing the GPU with a 97-document
corpus's embedder and reranker — and the answer path is exonerated without bisecting `v0.4.2..v0.5.1`.
**322 chunks also re-confirms KI-34**: extraction genuinely works in the frozen binary.

**Rejected alternatives.** *Bisect the answer path first* — the entry itself ranked contention as the
likelier cause and the experiment was far cheaper; bisecting would have burned a session to reach the
same place. *Download the GitHub asset to test "what users get"* — the hash already proves the local
staged copy is byte-identical, so a 1.5 GB download would have added a network round-trip, not
evidence. *Close RG-012's flakiness finding on the strength of a green run* — see below.

**What it opens.** **RG-012's citation verdict stays open and that is deliberate.** Run 4 passed, which
moves the failure rate from 1-in-3 to 1-in-4 on a byte-identical healthy artifact — it is one more
sample, not a fix, and the cause (KI-36's bimodal citing on `llama3.1:8b`) is untouched. The green run
is precisely when a flaky blocks-ship gate gets quietly retired, so the entry now says so in terms.
Its two cheap fixes (drive the gate's one turn with a paid model, or assert across N turns) remain the
work. Also opened: **dev-stack-up is now a known confounder for any RG-012 timing** — check 8001/1420
are clear before reading a turn time as signal. The 16 s is **not** promoted to `docs/performance.md`
or RG-010/011: one sandbox run over a 3-document corpus is a gate observation, not a benchmark.

---
## 2026-08-13 — ADR-045 (taxonomy display rule), and the auto-propose run that says the scope is the bug

**What changed.** New `docs/decisions/ADR-045-taxonomy-display-rule-and-document-identity.md` + its
index row; RG-015 in `.claude/RIGOR_TODO.md` gains a third evidence section. No code.

**ADR-045 fills a gap ADR-028 names in its own "Must revisit": *"the display rule is unspecified."***
A document shows its attached nodes **minus any that is an ancestor of another** (so a concept placed
under both `optics` and `neuroscience` shows both — neither subsumes the other); search and filter
match the **full ancestor closure**; attachment happens only at the most specific node and ancestors
are derived, never stored. Both rules degrade to identity on an unplaced vocabulary, which is not a
future edge case — it is the path 344 of 357 concepts take today.

**Why it matters beyond tidiness:** this is the actual fix for the partitioning failure measured
twice on 2026-08-12. `rag` is `df=1` and always will be; `machine learning` is not. Under closure a
97%-singleton vocabulary still filters cleanly, with **no change to extraction and no re-ingest**.
Two further decisions, both taken to keep something out of the vocabulary: a document identity key
(`rag_lewis_2020`) is a **computed field, not a `Concept` row** — it is an identifier, `df=1`
forever, and admitting it would manufacture 97 permanent singletons in the facet being repaired; and
bibliographic type/origin are **metadata columns**, because "is a journal article" is not "belongs to
a research field" and modelling it as `in_field` gives that edge two meanings.

**Then the measurement, because ADR-045 makes placement load-bearing.** Under closure a wrong
placement silently *widens a filter*, so `propose_taxonomy --apply --all-concepts --limit 25` ran on
`qwen3.5:9b` (49 calls, ~2 min, $0) before anything bulk. Judged against each term's own source
document: **~4 right · ~8 coarse · ~12 wrong**, and the wrong ones are not near-misses —
`acdc`→**Music** (the ACDC cardiac-MRI benchmark, from a Mamba-UNet segmentation paper),
`alpha`→**Analytical chemistry** (an EEG band), `actor`→**Performing arts** (actor–critic RL),
`accessory`→**built environment** (from a Cajal neuroanatomy paper).

**The finding is not "the model is bad".** On the **13 curated `graph_include` concepts** a *weaker*
model (llama3.1:8b, 2026-07-25) was **13/13 plausible**. On raw keyword rows a *stronger* model is
half wrong. **The variable is scope, not capability** — `--all-concepts` crosses the ADR-018
boundary on purpose and hands the classifier 344 rows nobody curated, ~40% of which are not concepts
at all (venue artifacts `aclweb`/`aclanthology`, orphans with no document links, and bare fragments
like `alpha`/`actor` that only mean something inside a phrase). So `graph_include` is a
**precondition for auto-propose working**, not bookkeeping.

**Confidence is still not a signal, third independent confirmation:** 0.80–0.95 across the sample,
`acdc`→Music at 0.80 and a correct placement at 0.80.

**A correction worth more than the run.** The taxonomy has **never had a human-accepted concept
placement**: all 37 concept→field links were `origin='proposed'`; the 213 `curated` edges are the
ANZSRC trunk itself (domain→domain). Every earlier statement of the form "13 concepts are placed"
should have said *proposed*. State which, always — they have never meant the same thing here.

**Disposition.** The 24 rows were deleted via `taxonomy.remove_hierarchy_edge` (back to 213 curated
+ 13 older proposals); backup `data/library.db.bak-20260813-pretaxonomy`. The remaining 333 were not
run. ⚠ **`--limit N` takes the alphabetically-first N**, so this was an `a*` sample — enough to
answer "usable in bulk?" (no), not enough to quote as a precision figure.

**Rejected.** Running the remaining 333 (would have written ~330 rows of this quality into a layer
ADR-045 just made load-bearing). Keeping the 24 as a labelled sample (the taxonomy view would show
`acdc → Music`, and a wrong placement that is *visible* is how a feature teaches something false —
the exact failure the 2026-08-12 relabels were about).

**What it opens.** The sequence is now clear and is upstream of the classifier: curate the
vocabulary, clean the D4/D5 residue out of it, then place a small set — by hand or with a stronger
model — and only then trust coverage. RG-015's original debt (per-kind detector precision) is
untouched and still needs the detectors to exist.

---
## 2026-08-12 (4) — keyword quality D4 + D5: the bibliography is where surnames come from, and the tokeniser was renaming genes

**What changed.** `knowledge/keywords.py` gains `strip_reference_section`, `is_citation_artifact`,
a rewritten `_TOKEN_RE`, and a head-check in `candidate_terms`; 15 more unit tests (44 in that file
now). Re-applied to the live 97-document library.

**D5 — the tokeniser was silently renaming things.** `_TOKEN_RE` split on `.` and `/`, so
`16p11.2` became `16p11` (a different locus), `C57BL/6` became `c57bl`, and `gpt-3.5`/`gpt-4.5`
became `gpt-3`/`gpt-4`. The fix keeps a separator **only when a digit follows it** — which is the
whole of what distinguishes a designator from prose. `e.g`, `i.e` and `arxiv.org` still split
exactly as before, because a *letter* follows their separator. The corpus now carries `16p11.2`,
`dlight1.1`, `gpt-3.5` and `gpt-4.5` whole, where before it carried truncations of them.

**D4 — surnames come from the bibliography, so remove the bibliography, not the surnames.**
⚠ The obvious fix here is a name filter and it is the wrong one: `cajal`, `cre`, `dbs`, `16p11`
and `c57bl` are real vocabulary in this corpus and have been mistaken for noise twice. So D4 is
structural — cut from a whole-line `References` heading to the end of the document — plus three
pure *shape* rules (`2014a` year-suffixes, `e04250` article ids, `10.xxxx` DOI prefixes). A
surname that survives that appears in the document's own prose, which is exactly when it is a real
term for that document. `shadmehr` and `wolpert` are still keywords of Shadmehr's own review, and
should be.

**The measurement that saved the fix from looking finished.** The first regex matched only a plain
`References` line and fired on **25 of 97** documents. Sampling the actual cache showed why:
PyMuPDF4LLM's dominant rendering is `## **References**` (32 of the sampled headings), with
`_REFERENCES_` behind it. Allowing markdown emphasis took it to **79 of 97** — a fix that would
otherwise have shipped working on a quarter of the corpus while reporting success.

**Effect on the two probes** (dry run, live corpus): year-suffix citations **2 → 0**;
`rag_lewis_2020.pdf` lost `10 18653 v1` and `aclweb org anthology` (DOI + URL fragments) and gained
`dpr` and `generator`; `transformer_vaswani_2017.pdf` lost `eos` / `my opinion` / `pad br` and
gained `attention function` and `convs2s`.

**A nuance worth recording, because it looks like a regression and is not.** Keywords on ≥2
documents went **48 → 42** and on ≥5 went **2 → 0**. Removing bibliographies removes terms that
were "shared" across documents *because they were citation artifacts* — spurious cross-document
links, not corpus structure. Partitioning is not what D4/D5 were for, and it remains the open
question from DEVLOG 2026-08-12 (3): per-document TF-IDF cannot produce a facet.

**Rejected.** A surname/author stopword list (the trap above). Rejecting any n-gram containing a
bare decimal — it would kill `gpt 3.5`, a real term, to remove `0.0 true`. Cutting at the *last*
References heading rather than the first past the halfway mark (an appendix after the bibliography
is smaller than the risk of matching a heading in the body).

**What it opens.** Residual noise this did not touch and did not create: `br` is an HTML
line-break artifact leaking out of the markdown (`1020 br`, `br 32`, `0.01 br`), and bare decimals
still ride inside bigrams (`0.0 true`). Both are small, both are their own fix, neither is D4/D5.

---
## 2026-08-12 (3) — the four release-readiness relabels: a rank instead of a score, keywords that are about the paper, an experimental label, and a hidden tab

**What changed.** The user's decision on `docs/REVIEW_2026-08-12_release-readiness.md` §2b was
**"fix all four, hide the graph tab"**. Four independent changes, one theme: *a built feature that
teaches a first-time user something false is worse than one that is missing.*

**R1 · Connections shows a rank, never a score.** New `lib/library/connections.ts` (+5 tests);
`DocConnections.svelte` prints `1st / 2nd / 3rd` and a one-line caveat instead of `score.toFixed(2)`.
Measured on the live corpus while verifying: one document's top four neighbours score **0.982,
0.965, 0.962, 0.956** — the old UI rendered that as "0.98 / 0.97 / 0.96 / 0.96", which reads as a
precise claim about how alike two papers are. It is not one: `doc_vectors` mean-pools every chunk,
so same-field papers collapse (750 edges, **median 0.918**, against a 0.5 threshold that therefore
excludes almost nothing). The ordering survives that; the distance does not. The caveat is a
sentence, not a tooltip, because the user who never hovers is exactly the one who would over-read it.

**R2 · Keywords — D1 (page furniture) + D2 (shingle suppression), and the honest half of the result.**
New pure functions `split_pages` / `strip_page_furniture` / `suppress_nested` in
`knowledge/keywords.py` (+14 tests), wired into the shipped `per_doc` path, then **applied to the
live 97-document library** (`extract_keywords --apply --force`, $0, deterministic; `library.db`
backed up first).

| | before | after |
|---|---|---|
| documents with ≥1 keyword | 82 / 97 | **96 / 97** |
| nested/shingle slots | **334 / 1230 (27%)** | **0 / 1440 (0%)** |
| distinct keywords | 1192 | 1375 |
| on ≥2 documents | 30 | 48 |
| on ≥5 documents | 1 | 2 |
| on ≤1 document | 1162 (**97%**) | 1327 (**97%**) |

The eyeball test is decisive. `nihms-66884.pdf` went from **11 of 15 slots** being shingles of the
PMC running header (`author manuscript` · `exp brain res` · `2008 september 26` …) to
`motor commands` · `optimal control` · `proprioceptive` · `state estimation` · `saccades` — the PMC
stamp is *gone*. `transformer_vaswani_2017.pdf` went from **9 of 15** slots on one figure's
`<eos> <pad>` artifact to `multi-head attention` · `self-attention` · `scaled dot-product` ·
`sequence transduction`. `rag_lewis_2020.pdf` went from **zero** keywords to `rag` · `rag-sequence` ·
`rag-token` · `retriever` · `non-parametric memory`.

**But the number the plan said to report did not move, and that is the finding.** Singletons are
**97% before and after**. D1/D2 fixed *what the keywords say*; they did not make the layer
**partition** the corpus — and they cannot, because per-document TF-IDF selects `df≈1` distinctive
terms *by construction* (this module's own docstring says so). **So keywords are a good per-document
descriptor and are still not a facet.** Partitioning would need the `corpus_band` or `contrastive`
mode, which is a different decision, not a bug fix. The Library overlay already tells the truth here
(PR-2.7 F4 collapses the 1-document tail behind *"on 1 document — search still finds them"*), so no
UI change was needed — but the open question survives this session intact.
*(The coverage jump 82 → 96 is **D3**, not D1/D2: it is simply what re-running a stale enrichment
does. One document still yields nothing.)*

**R3 · The epistemics chips say `experimental`, in both places they are met.** The chip carries an
inline `experimental` tag (inside the pill, so a line wrap cannot separate the qualifier from the
claim) and a tooltip that says it is not a measurement and can be wrong; the Settings section gains
a `(experimental)` heading and a plain-words **Known limitation** paragraph. The default was already
`false` (flipped 2026-08-03, KI-33) — what was missing was *saying so*: a silent default tells a user
who turns it on nothing, and a user who leaves it off nothing either. Rebuild is ADR-041.

**R4 · The Graph tab is hidden.** New `lib/core/features.ts` with a single documented
`GRAPH_TAB_ENABLED = false`; the tab is `{#if}`-gated in `Topbar.svelte` and `selectMode` coerces
`graph → chat` defensively, because that function is also the nav-history restore path and landing
in a mode with no exit is the one failure a hidden tab can still cause. **Nothing was deleted** —
`/api/concepts/*` stays mounted, the gap list keeps its triage writes, and `GraphIndex` /
`ConceptGraph` / `GapList` are untouched and still tested. Checked before hiding: the Taxonomy view
is a global overlay opened from the **Library** rail, so it is not orphaned by this.

**Rejected.** Deleting the graph code (throws away working, tested work to solve a *placement*
problem). Keeping the similarity number in a tooltip (if the number is not meaningful, hiding it one
hover away is not honesty). Growing `VENUE_STOPWORDS` for D1 — the signal is position and
repetition, not vocabulary; every publisher has a different stamp and the words in them
(`brain`, `september`) are not junk anywhere else. Applying D2 to `contrastive` (it already
discounts nesting via C-value) or `corpus_band` (its exposure is via furniture, fixed at source).

**What it opens.** The facet question above. Whether corpus growth should re-trigger keyword
enrichment automatically (**D3**, which is KI-44's question in a different costume — answer them
together). And the one document that still extracts zero keywords.

---
## 2026-08-12 (2) — the graph rail's three small fixes, and two checklist rows that were already done

**What changed.** `gaps.ts` gained `filterGapRows` (+5 tests); `GapList.svelte` gained a filter box
and turned its two checkboxes into pressed-state `.lens` buttons; `GraphIndex.svelte` did the same
to `Include under-connected` and lost the now-unused `.toggle` rule. Both `Gaps only` and
`Include under-connected` got tooltips that say what the control *does* rather than restating its
own label. `docs/ui-checklist.md` updated.

**Why.** Three rows the checklist marked "no dependencies", and they were: `Gaps only` was a button
while `Include under-connected` next to it was a checkbox — two different kinds of thing doing one
job — and the Gaps tab had no filter while the concept rail beside it did. The filter matches the
**gap kind** as well as the concept label, because "single" is the list's own vocabulary for a class
of problem; matching labels only would return nothing and read as *"no such gaps"*.

**The other half is a finding, not a change.** The two "review findings" in §3 —
`reviewer_kind="llm_haiku"` hardcoded, and `set_llm_selection` accepting an empty model — are
**both already fixed in code**, with a test pinning the first
(`test_chat_controller.py:1052`). The rows were stale. §3 now says so, because the cost here was
only ten minutes of reading, and the cost of *believing* it would have been a redundant "fix" to
working code.

**Verified live** (dev server, $0): 15 gap rows → 3 on `single`, all Single-source; a no-match query
shows the honest empty line; both rails show two matching lens buttons and zero leftover checkboxes.

---
## 2026-08-12 (1) — the app can now tell you a new version exists, and that is deliberately all it can do (ADR-044)

**What changed.** New `docs/decisions/ADR-044-update-notification-not-delivery.md` + its index row.
New `src/doc_assistant/update_check.py`, `apps/api/{models,routers}/updates.py` (three routes),
`apps/desktop/src/lib/core/{types,api}/updates.ts`, `apps/desktop/src/lib/settings/updates.ts`
(pure display logic) and an **Updates** section in `Settings.svelte`. `app_settings.py` gained
four accessors. **`src/doc_assistant/__init__.py` gained `__version__`** — it was empty, and the
app had no runtime knowledge of its own version at all. Tests: 26 unit + 12 integration + 9
frontend, all passing; `release_preflight` now checks **six** version strings, not five.

**Why.** The app ships as an NSIS installer with no store and no package manager behind it, so
every install is frozen at the version it shipped with — including one carrying a bug we have
already fixed. The user's framing was explicit: do it like calibre — signal, link, and let the
user install — and **an integrated updater is too ambitious for now** because several features
are not stable enough to push at people automatically. That rejection is the load-bearing half:
an in-app updater must replace a running binary, verify a signature and roll back a bad write,
and none of that is earned while the release process has not yet produced a verified artifact for
the tag it cut.

**The design decision that is not implementation detail: three states, never two.** A failed check
reports `unknown`, never `current`. Saying "you are up to date" because the network was down is
the one failure mode that would make this feature worse than not having it, so every network,
parse and decode error becomes `unknown` with a plain-words reason the UI shows verbatim. Verified
against the real endpoint: the repo has **no published releases**, and the app says *"no published
release to compare against"* rather than going quiet or claiming currency.

**Two defects found while building it, both worth recording.**
1. **My first cut cached only the check timestamp, not the observed version** — reasoning that a
   stored verdict would go stale. It does, but the cure was worse: a GET inside the 24 h window
   then *forgot* the update the last check found, so the banner vanished on the next page load.
   Fixed by storing the observed **version** and recomputing the verdict against the running one
   on every read — which self-corrects in both directions (after the user updates, the same cached
   observation reads `current` with no further request). Two regression tests pin it.
2. **mypy caught a real portability bug**: `datetime.UTC` is 3.11+, and this package declares
   `requires-python = ">=3.10"`. Now `timezone.utc`. CI runs the same interpreter as the dev box,
   so nothing else would have caught it.

**Rejected.** In-app download/install (user's call, and it needs a code-signing decision first).
Default-on checking (this app makes no outbound calls the user did not ask for; the toggle is off
by default, and the **manual** "Check now" runs regardless, because gating an explicit press would
leave a user who declined background traffic no way to answer the question at all). A configurable
repository URL — pointing an update banner at an arbitrary host is a way to get someone to install
something they did not choose. Notifying on `unknown` outside Settings — an offline machine would
nag forever about a check it cannot run.

**What it opens.** This project now has a **network code path in `src/`, permanently**, and it
inherits the frozen-build OS-trust obligation (KI-10) for a second caller. It also creates a
release-process coupling that is now real and currently unmet: **the banner is only truthful if
GitHub *release objects* are cut for tags**, and none exist — `v0.5.0` is not even pushed. Cutting
a release is now a step in `docs/RELEASE.md`, not a courtesy. Not built, and deliberately: any
surface outside Settings (a topbar dot, a startup banner) — `shouldNotify()` exists and is tested,
but nothing calls it yet, so today the only place an update appears is the Settings panel.

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

---
## 2026-08-07 (6) — an eval run now records the settings that produced it (RG-026's precondition)

**What changed.** New `eval/run_settings.py` — `run_defining_settings()`, a snapshot of the 13
config values that determine what a run measures (the six chunk sizes, the embedder, and
`use_parent_child` / `use_multi_query` / `top_k` / `candidate_k` / `bm25_weight` /
`rerank_candidate_cap`). `Store` takes a `settings_provider` and merges the snapshot **under** the
caller's `config`, so an explicit per-run override (`run_eval --bm25-weight`) still wins — the
recorded value is always the one that ran. New `Store.run_config(run_id)` so a past run can be
audited at all. `scripts/run_eval.py` wires it; 9 tests.

**Why.** KI-41: the 2026-06-06 chunking sweep swept one configuration six times and **nothing in
the run record could contradict its notes** — `config_json` held `embedding_model` / `n_cases` /
`scorers` and no chunk sizes. That is why it took two months and an unrelated investigation to
find. RG-026 makes recording the varied settings the precondition for re-running the sweep, and
this is that precondition.

**The design was wrong first, and the suite said so.** The obvious shape — merge the snapshot
inside `persist_run`, importing `run_defining_settings` at the top of `store.py` — makes it
impossible for a runner to forget, and I built that first. It fails
`test_eval_harness_isolation.py`: `doc_assistant.eval` must import **no** app wiring, because the
harness is designed to be lifted into a standalone repo (ADR-003 Decision 8), and `run_settings`
reaches into `config` by nature. Injection is the shape that satisfies both — the coupling sits at
the project's edge of the harness, and a lifted copy drops one file.

**The cost of that is real and is written down where it bites:** a new runner must remember the
argument, so the rule is in `scripts/CLAUDE.md` and in both docstrings. A guarantee enforced by
construction would have been better; the extractability contract is worth more, and it is
test-enforced while "remember the argument" is not.

**Rejected.**
- **Backfilling old rows** with today's config values. Their geometry is genuinely unknown —
  substituting a plausible value would convert "unrecorded" into a false record, which is worse
  than the gap. `run_config` returns exactly what is there, and its docstring says to read with
  `.get()` and report "not recorded".
- **Recording every config knob.** A record nobody trusts to be minimal is a record nobody reads.
  The membership rule is "changing it changes what the run measures", so worker counts, caches and
  the lazy reranker are deliberately absent.
- **Surfacing the settings in `format_run_summary`.** Worth doing when someone is comparing runs;
  it is a different change and would not have caught KI-41 (the sweep printed per-config notes
  that *looked* right).

**What it opens.** RG-026's precondition is met, so a chunking re-run would now be self-auditing —
still a GPU-day, still the user's call. Nothing reads the new fields yet: a `sweep`-side assertion
that the arms actually differ (fail loudly when two grid points record identical settings) is the
natural next guard, and it is the one that would have turned KI-41 into a first-run error.

---
## 2026-08-07 (5) — `.env` stops beating the environment (KI-38), and the chunking sweep turns out to have measured nothing (KI-41)

**What changed.** `config.load_dotenv(override=True)` → `config._load_env()`, whose rule is: **a
non-empty process environment variable wins; `.env` fills in the absent and the empty.** 7 tests in
`tests/unit/test_config_env_precedence.py`. Corrections to the record where it now reads false:
the locked-settings chunk-size row (`.claude/CONTEXT.md`), `evals/README.md` § Chunk sizes, and a
dated correction appended to the 2026-06-06 baseline.

**Why.** KI-38 recorded the override as a credit leak — `LLM_PROVIDER=ollama <cmd>` runs on
Anthropic and bills. That was true and it is the smaller half. The override applied to **all 19 keys
`.env` defines here**, and `.env.example` ships the chunk sizes uncommented, so
`scripts/sweep_chunking.py` — whose entire mechanism is passing `PARENT_CHUNK_SIZE` /
`CHILD_CHUNK_SIZE` to an ingest subprocess — had no effect on the thing it sweeps.

**The finding, and it is measured rather than argued.** `case_results.token_input` scales with the
evidence block, so a parent size of 1000 vs 3000 must move it. Across all **18** runs of the
2026-06-06 sweep it does not move at all: mean **4326.7**, min **3582**, max **5106**, in every
config — and **identical per case** between the control and parent 3000, all ten. The field is live
elsewhere in the same DB (4372.7, 4615.8). **The sweep that closed the "defaults never measured"
caveat compared one configuration with itself six times**, at a cost of ~6 full corpus re-embeds.
The chunk-size lock is back to *unmeasured* — not wrong, unsupported.

**One free result out of it.** Those six configs report `contains_all` 0.906–0.933 and `llm_judge`
3.793–3.951 **on identical inputs**, which makes that spread a direct reading of the harness's noise
floor at n=3 on the public 10. The baseline's own text called it "within the trial-to-trial noise
bands" — right, for a reason it could not have known.

**Rejected.**
- **The fix the KI proposed** — threading an explicit provider argument through the answer path. It
  addresses the provider symptom and would never have reached `PARENT_CHUNK_SIZE`; the defect is in
  config loading, so that is where it is fixed. One place, one rule, whole class.
- **Dropping `override=True` outright**, which the KI correctly ruled out: it re-opens the
  empty-`ANTHROPIC_API_KEY` shadowing the original comment describes. Narrowing the override to
  exactly the empty case keeps that protection — and it is what the comment already justified, so
  the code now does what its own reason says.
- **Changing `REVIEWER_PROVIDER_PINNED`.** The pin is ADR-011 U1c and deliberate. It stays; what
  changes is that `REVIEWER_PROVIDER=ollama` in the environment now *works*, so the residual partial
  leak has a cure instead of being unreachable.
- **Re-running the sweep now.** It is a GPU-day of re-embedding and a deliberate call, not a
  cleanup — and it should not run until it records what it ran.

**What it opens.** `runs.config_json` stores `embedding_model` / `n_cases` / `scorers` and **no
chunk sizes** — an experiment that does not record the setting it varies cannot be audited, which is
exactly why this survived two months. Recording the varied settings is the precondition for any
re-run. Unknown whether other env-driven experiments were affected: the two checked are clean —
`sweep_bm25_weight.py` passes weights in-process, and the 2026-06-04 embedder A/B demonstrably took
its `EMBEDDING_MODEL` override (its arms differ), which is consistent with it having run on the
retired CPU box under a different `.env`.

---
## 2026-08-07 (4) — the Windows encoding rules are written down as rules, not as one runbook's war story

**What changed.** One rule, four homes: `.claude/CONTEXT.md` non-negotiable **#9** (canonical text),
an `AGENTS.md` digest bullet (the only version a fresh clone gets — `.claude/` is local-only), a
`docs/setup.md` § *Windows: text encoding* table (contributor-facing), and a line in
`scripts/CLAUDE.md`, which is where the console rule actually bites. `docs/desktop-packaging.md`
§5 trap 1 now points at the general rule instead of standing alone.

**Why.** All three defaults were already known and each had cost a run, but each was recorded only
where it was discovered: the PowerShell-5.1-reads-BOM-less-UTF-8-as-ANSI trap lived inside the RG-012
sandbox runbook (so it read as a sandbox quirk rather than a Windows one), the `sys.stdout`
reconfigure existed as a copied header in **36** files (all 36 correctly `hasattr`-guarded, checked
while writing this) with the reason recorded nowhere, and the
`encoding="utf-8"` convention on file I/O was written down nowhere at all. A convention that lives
only in existing code is one refactor from being dropped as noise.

**The part worth carrying: none of this is gate-visible.** CI is Linux, pytest captures stdout
through its own UTF-8 buffer, and ruff's `PLW1514` is not in `select` — so all three failures are
green-suite failures. That sentence is now in the rule itself, because it is the reason the rule has
to exist at all rather than being replaced by a check.

**Rejected.**
- **A lint instead of a rule** (enabling `PLW1514`) — it would cover the file-I/O third only, says
  nothing about the console or PowerShell halves, and this repo's convention is that a locked
  behaviour gets its rule text first. Worth doing on its own merits; recorded as an open, not folded
  in here.
- **One home instead of four.** `.claude/` is local-only, so a canonical-only rule is invisible in a
  clone; a docs-only rule is invisible to an agent reading the entry file. The duplication is the
  digest pattern the repo already uses for the other eight non-negotiables.
- **Restating the rule in `src/doc_assistant/CLAUDE.md`** — the module files cap at 40 lines and are
  explicitly not for project-wide rules; the file-I/O rule is already visible there as `fsutil`.

**What it opens.** `PLW1514` is unselected, so nothing stops a new `open()` without `encoding=`.
Enabling it is cheap and would need a sweep of the existing call sites first.

---
## 2026-08-07 (3) — the extraction cache now knows which extractor wrote it (KI-40), so yesterday's fix can actually reach a user

**What changed.** `ingest/cache.py`: an `extraction_fingerprint()`, recorded beside every cached
`.md` as `<name>.md.fp` and compared in `is_cache_fresh`; a `write_cache()` that writes the pair;
`reason=` on the extraction log line. 9 new tests, 5 integration fixtures re-pointed.

**Why.** The text-layer fallback recovers three documents from ~0 to 46k / 89k / 778k characters —
**on a fresh ingest**. `is_cache_fresh` compared mtimes only, so on an existing library it changed
nothing, silently. **Shipping it would have been inert for every current user**, which is the worst
kind of release: the improvement is real, measured, and invisible to the people who have the
problem. KI-14 and KI-29 both changed extraction output and both had the same hole.

**Bump-free, copying the precedent already in this repo.** `sparse_index.fingerprint` hashes the
tokeniser's source specifically so a change invalidates *"without anyone remembering to bump a
constant"*. The same standard applies here, so the fingerprint hashes **every extractor function's
`co_code`** — plus three things bytecode cannot see:
- **module-level tunables**, referenced by *name* from a function so their values never reach
  `co_code`. `_TEXT_LAYER_KEPT_MIN` is exactly that knob, and it changes output;
- **`config.PDF_EXTRACTOR`**, which selects the extractor at all;
- **the PyMuPDF / PyMuPDF4LLM versions** — a dependency upgrade changes extraction output with no
  code change of ours.

Plus `_EXTRACTION_VERSION`, a manual escape hatch for the residue (a changed string literal alters
output without altering bytecode). **Bytecode, not source**, for two concrete reasons: PyInstaller
ships `.pyc` so `inspect.getsource` raises in the frozen build, and hashing source would re-extract
every library on a **comment** edit.

**The cost is surfaced before it is paid, and that came for free.** `plan_files` is stat-only, so it
reports the work without extracting: `--dry-run` now reads `would_reembed=97` **in 8 s**. Each
re-extraction logs `reason="extractor_changed"`, so a one-off slow ingest is explained rather than
mysterious.

**Rejected.**
- **Treating a fingerprint-less cache as fresh** (grandfathering existing libraries). It would have
  made the upgrade path work on paper and deliver nothing — the exact defect being fixed.
- **A hand-bumped version constant alone**, which is what the KI first proposed. The repo's own
  precedent rejects it, and a forgotten bump reproduces this bug in full silence. It survives only
  as the escape hatch.
- **A header inside the `.md`.** Those bytes *are* the document text and are hashed into
  `doc_hash`; anything added would change every document's identity.
- **Writing the fingerprint first.** It could then vouch for a truncated `.md`. Written last, a
  failed write costs one needless re-extraction and cannot lie.

**A drift the tests caught, and the fix that removes the class.** 13 integration tests failed
because five separate fixtures hand-wrote a cache entry as "a `.md`, newer than the source" — which
is no longer what a cache entry *is*. Rather than patch five call sites, "a cache entry is the pair"
now lives in one function (`write_cache`) that both the pipeline and the fixtures call. Five
hand-rolled definitions are how the meaning drifted in the first place.

**What it opens.** The first ingest after any extractor change re-extracts the library — inherent,
per-change rather than per-launch, and now visible in the plan. On a 10k corpus that is the ~41 h
figure from `performance.md`, so a future extractor change is a **release-note-worthy event**, not a
silent one. The API/UI ingest path shows the plan less prominently than the CLI's `--dry-run`;
worth a look when the Library UI is next touched.

---
## 2026-08-07 (2) — EX1: three of the four "scanned" documents never needed OCR. Retrieval recall 28/35 → **34/35**

**What changed.** A text-layer fallback inside `extract_pdf_pymupdf` (+ `_recover_lost_page`, a
`Protocol` pair to keep it testable, 9 guard tests), and **KI-40**. No OCR, no new dependency, no
system binary, no second extraction path.

**Why — the premise of ADR-039 was right as a category and wrong about which documents are in it.**
EX1 was scoped as "scanned PDFs have no text layer, so OCR them". Before installing anything, I
looked at the four degraded documents. **Three of them already carry a good text layer:**

| document | health | `page.get_text()` | what the extractor cached | word-like |
|---|---|---|---|---|
| `hebb_1949` | marginal | **776,162 chars** | 5,117 | 94% |
| `hodgkin_huxley_1952` | marginal | **88,754** | 3,219 | 72% |
| `hubel_wiesel_1959` | broken | **45,995** | 86 | 88% |
| `middleton-2001` | broken | **0** | 0 | — (a true scan) |

**Mechanism, reproduced on the shipped extractor** (so it was live, not a stale cache): PyMuPDF4LLM
sees a full-page image, emits `**==> picture [331 x 154] intentionally omitted <==**` and never
reaches the invisible text behind it; `strip_image_placeholders` (KI-14) then removes even that,
leaving `## ##`. Measured retention on those pages: **0.0%–3.2%**.

**The fallback is licensed by a measurement, not a guess.** The two populations do not overlap or
come close — 14 healthy PDFs over 28 pages kept **97.3%–108.9%** (over 100% because markdown *adds*
structure), the degraded ones **0.0%–3.2%**. A ~94-point gap with nothing in it, so a threshold in
the middle is **structural, not corpus-tuned** — which is what the robustness contract actually
demands. `_TEXT_LAYER_KEPT_MIN = 0.5`, and a test asserts it stays inside the measured gap.

**Result, end to end:**

| document | chunks before | chunks after | health |
|---|---|---|---|
| `hubel_wiesel_1959` | 1 | **61** | broken → **healthy** |
| `hodgkin_huxley_1952` | 7 | **125** | marginal → **healthy** |
| `hebb_1949` | 16 | **1,019** | marginal → **healthy** |
| `middleton-2001` | 0 | 0 | broken (unchanged — correctly) |

Corpus health **93/2/2 → 96 healthy / 0 marginal / 1 broken**. **Retrieval recall on the private
35-case set: 28/35 → 34/35.** The single remaining miss is `middleton_frontal_subcortical`, whose
document is the one genuine no-text-layer scan.

**So ADR-039's actual scope is now one document in 97**, not four — and that document is the case
the ADR describes exactly. The OCR work is still worth doing and still gated on RG-025; it is simply
much smaller, and it no longer blocks the recall gap it was thought to own.

**Rejected.**
- **Installing Tesseract + Ghostscript first.** Neither is present, both are system binaries, and
  three of the four documents turned out not to need them. Looking at the documents cost minutes and
  removed the dependency question from 75% of the problem.
- **Fixing this in the health scorer.** `health.py` was right: these documents *were* broken as
  ingested. The defect was upstream of the label.
- **Suppressing the placeholder differently / tuning KI-14's stripper.** The placeholder is a
  symptom; the text never reached the markdown in the first place.
- **A ratio derived from these four documents.** That would be corpus-tuned. The threshold is
  justified by the *gap between two populations*, and the test pins it to that gap rather than to a
  value.

**What it opens — and it is bigger than this fix (KI-40).** `is_cache_fresh` compares **mtimes
only**, so the cached `.md` is never invalidated when the *extractor* changes. **Shipping this fix
does nothing for any library that has already ingested** — including every existing user. KI-29 and
KI-14 both changed extraction output and both had the same hole. The pattern to copy is already in
this repo: `sparse_index` fingerprints its inputs and logs `sparse_index_stale`. Until then the cure
is manual, and it has a trap: re-extraction changes `doc_hash`, `_existing_document_id` matches on
`doc_hash`, so a `--files` re-ingest **mints a second row and skips orphan cleanup** — measured here
as 97 → 100 documents, each file listed twice with different health. A plain `ingest` reconciles it
(`cleanup_orphans_sqlite` handles "the pre-change hash of a document whose content changed"), which
is what was run: back to 97.

Also: the corpus grew by ~1,200 chunks, so the 2026-08-07 citation-coverage baseline was measured
against a *slightly different* corpus than the one now on disk. It compares before/after within
itself, so its conclusion stands, but a future re-run will not be exactly comparable.

---
## 2026-08-07 (1) — the prompt fix cured the citation FORMAT and moved coverage not at all (KI-36 re-measured on shipped code)

**What changed.** No code. One baseline —
`tests/eval/baselines/citation_coverage_2026-08-07.md` — plus KI-36.

**Why.** v0.4.1 ships a provider card publishing **36% / 14% / 81%** citation coverage, and those
numbers were measured on the **previous** prompt. The 2026-08-06 header change altered how sources
are presented to the model, so the app's own published claim about itself was unverified against
the code that shipped. That is exactly the kind of quiet staleness that turned a CHANGELOG limit
into a lie earlier this week.

**Measured** — same 27 healthy-document cases, same exclusions, same settings, one variable:

| provider / model | before | after (shipped) | Δ | citing nothing |
|---|---|---|---|---|
| `anthropic/claude-haiku-4.5` | 81.2% | **83.5%** | +2.4 pp | 0/27 → 0/27 |
| `ollama/llama3.1:8b` | 36.4% | **37.6%** | +1.2 pp | 11/27 → **12/27** |
| `ollama/qwen2.5:7b` | 13.5% | **18.0%** | +4.5 pp | 18/27 → 18/27 |

**The finding is the negative, and it is worth as much as a positive would have been.** Every delta
is same-signed but inside what a single repeat over 27 cases can resolve against ~3% case-level
retrieval noise, so **none of it is reportable as an improvement**. Paired by case, `llama3.1:8b`
had **no** answer that cited nothing start citing, and one that had cited stop.

**So: prompt engineering fixed what prompt engineering could fix, and did not touch the rest.** The
header change took header-copies 6 → 0 and RG-012 FAIL → PASS — a format defect, cured outright.
Coverage sat still. **That is a second, independent line of evidence for KI-36's capability-floor
conclusion:** the first was cross-provider (same prompt, 81% vs 36%), this is same-provider (changed
prompt, same coverage). Two different cuts, same answer.

**Rejected.**
- **Updating the shipped provider card to 38 / 18 / 84.** Every delta is within noise, and changing
  a shipped string means rebuilding and re-running RG-012 on a freshly tagged release — trading a
  real risk (an unverified rebuild) for a cosmetic gain. Refresh it at the next release that
  rebuilds anyway; the baseline records which column to use.
- **Calling +4.5 pp on qwen an improvement.** It is the largest delta and the noisiest arm (13.5 →
  18.0 with 18/27 answers still citing nothing). Reporting it would be exactly the "claim without a
  control" this project's rigor gate exists to stop.
- **Adding repeats now to settle the drift.** Worth doing before anyone acts on the drift; not
  worth blocking a release that does not depend on it.

**Correction to 2026-08-05 (2)** (append-only, so recorded here rather than edited there): that
entry's table says `qwen2.5:7b` had **19/27** answers citing nothing. Recounting the stored rows
gives **18/27**. The pooled figure (13.5%), the median (0.000) and every user-facing number derived
from them are unaffected; only the count was wrong. Corrected in KI-36.

---
## 2026-08-06 (4) — release tooling: a preflight that encodes every trap that actually bit, and CI finally runs the frontend

**What changed.** New `scripts/release_preflight.py` (+ `just preflight`), `docs/RELEASE.md`,
`tests/unit/test_release_preflight.py`, a **frontend job in CI**, and `npm audit fix` (postcss
→ 0 vulnerabilities).

**Why.** Today's release was cut by hand and nearly went wrong four separate ways. None of those
were interesting problems — they were all "did you actually do the thing" problems, which is what a
script is for. **Every check in the preflight is an incident, not a best practice:**

| Check | The incident |
|---|---|
| `versions` | v0.4.0 bumped four version strings and missed `uv.lock`. CI installs `--locked`, so the job died *before* the gates and `main` was ungated for days. |
| `artifact_fresh` | The whole of 2026-08-06. Source-green says nothing about a frozen binary (KI-34). If the installer predates the code, the thing tested is not the thing shipped. |
| `sidecar_size` | KI-34 is a size cliff — 1545.5 MiB broken vs 1562.1 MiB fixed. The cheapest possible check on a packaging bug that is invisible from source. |
| `rg012` | Matches the installer build timestamp the harness logged against the installer on disk. **A PASS from a previous build is worse than no PASS** — it reads as evidence. |
| `dev_commands` | KI-39: the app told users to run `just api`. |

**Two of the five checks were wrong on their first run, both in the "looks green" direction** — and
that is the part worth keeping:
1. `sidecar_size` was written in **decimal MB** while every recorded reference number is **MiB**,
   putting the floor at ~1478 MiB — *below both* the broken and fixed sizes. It would have passed
   the exact build it exists to reject, while printing `[ok]`. **A units bug in a safety check is
   worse than no check.**
2. `dev_commands` matched `just \w+` and flagged **"just now", "just a", "just the"**. `just` is an
   ordinary English word; only a real recipe name makes `just X` a command, so the names are now
   parsed out of the justfile — which also keeps the check honest as recipes come and go.

Both are pinned by tests, so neither can regress quietly.

**CI had no frontend job at all.** The desktop app is half the product, and `npm test` (78 tests)
plus `svelte-check` (189 files) ran on a developer's machine or not at all — every frontend fix
today, including KI-39 and the citation-contract pin, was guarded only by me remembering. Added as
its own job: `npm ci` (lockfile-exact, same discipline as `uv sync --locked`), `svelte-check`, then
tests. Seconds long, no Python.

**`npm audit fix`** (carried since 2026-08-05): postcss had a **high**-severity path-traversal
advisory and the lockfile pinned **8.5.15 while the local tree had 8.5.25** — so CI would have
installed the *vulnerable* one. Now 8.5.26, **0 vulnerabilities**, 78/78 + svelte-check still green.
Build-time only, so the v0.4.1 artifact is unaffected and needs no rebuild.

**Rejected.**
- **Automating the CHANGELOG check beyond "the section exists".** A script cannot tell whether a
  known limit is still true — and that is exactly the failure mode we hit (0.4.1 claimed the
  clean-machine install was unverified for three days after it was verified). `docs/RELEASE.md` §2
  makes it a judgment step instead of pretending.
- **Making the preflight a pre-commit hook.** It reads build artifacts and the RG-012 archive;
  most of it is meaningless mid-development and would train people to ignore it.
- **A `release` recipe that runs the whole thing end to end.** The sequence has two ~10-minute
  builds, a machine-level Ollama rebind and a VM in the middle. A checklist that a human drives
  beats a script that pretends those are atomic.

**What it opens.** `preflight` hard-codes the Windows harness path and the msvc triple — fine on the
one build box, wrong the day there is another. And it cannot check the thing that matters most:
whether the answers got worse.

---
## 2026-08-06 (3) — RG-012 FAILED on a citation form, and the fix was to stop showing the model a bracket to copy

**What changed.** `pipeline.format_docs_for_prompt` — the retrieved-passage header is no longer
bracketed: `[Source 3: paper.pdf, page 4]` → `Source 3 — paper.pdf, page 4`. `prompts.py` updated to
match, plus one new line: *"Square brackets appear nowhere in the sources — every `[n]` you write is
a citation."* Two guard tests.

**Why — a real RG-012 FAIL, and the genuine version of what KI-35 falsely claimed.** The re-run
against the rebuilt installer produced a meticulously attributed answer that cited
**`[Source 1: reranking_bert_nogueira_2019.pdf]`** six times — the *source header format*, copied
verbatim. `synthesis._CITATION_TOKEN_RE` accepts a label plus digits, not a filename, so it resolves
to nothing: **`valid=[]`, 6 malformed, 12/12 sentences uncited, all 12 claims badged `uncited`** on
an answer that names and quotes its source in every paragraph. `prompts.py:47` has warned against
exactly this since 2026-07-14 and the model did it anyway.

**Three runs, same model, same question, three different citation forms** — the instability
measured rather than asserted:

| run | form | outcome |
|---|---|---|
| 2026-08-05 17:31 | `[Source 1]` ×3 | resolved (the 2026-07-14 tolerance covers it) |
| 2026-08-06 14:13 | `[1]` ×5 | PASS |
| 2026-08-06 20:10 | `[Source 1: paper.pdf]` ×6 | **unresolvable → FAIL** |

**Chosen fix: remove the imitation target, not widen the reader.** Square brackets now appear
**nowhere** in the context, leaving `[3]` in the instructions as the only bracketed thing in the
entire prompt. The alternative — teaching the parser to swallow `[Source N: anything]` — was
explicitly rejected: KI-35 itself warned that *"a silently-widened parser hides model drift"*, and
that is precisely what it would have done here. The model would have gone on emitting a
non-canonical form and we would have stopped being able to see it.

**Measured after the change**, same question, 6 runs, shipped local default (`llama3.1:8b`, $0):
**header-copies 0/6**; 5/6 clean canonical `[n]` (3–6 citations each, all resolving); the sixth
flagged `[CLS]` — a BERT token quoted out of the source text, not a citation attempt, and the same
benign "audit cries wolf" class already noted for bracketed phrases. **Caveat: 6 runs on one model.**
The failure appeared in 1 of 3 runs before, so 6/6 is suggestive, not conclusive; RG-012 is the gate
that decides.

**Rejected.**
- **Widening the parser to accept `[Source N: …]`.** Unambiguous, and tempting — but it converts a
  visible defect into an invisible one. The header was the cause; fix the cause.
- **Arguing harder in the prompt.** Five citation rules and a parenthetical naming this exact
  confusion were already there and were ignored. A sixth rule is not a mechanism.
- **Keeping brackets but changing the inner text** (e.g. `[3]` alone as the header). Then the
  context contains bracketed numbers, and a model echoing one is *indistinguishable* from a real
  citation — worse than either alternative.

**RG-012 re-run after the full rebuild (sidecar re-frozen 646 s + installer re-bundled): PASS.**
Install 202 s → health ~30 s → 3 PDFs / 322 chunks → cited turn 17 s → **4 resolved citations, 4
canonical `[n]`, 0 labelled, 0 unresolvable.** Verified in that artifact: badges `uncited` ×6 /
`weakly grounded` ×2 (KI-37), `best source relevance` present and the old `Reranker scores` heading
gone, `citation_note_md` **empty** (audit clean — no malformed anything), and the answer's bracket
tokens are exactly `[3] [1] [8] [7]`.

**What it opens.** The audit still counts any bracketed token containing letters as a failed
citation attempt, so a passage quoting `[CLS]`, or a model wrapping a phrase in brackets, produces a
false "malformed citation" warning. Same disease as KI-35, one layer down; recorded in KI-36.

**Harness note — Windows Sandbox runs ONE instance, and a stale VM silently eats the run.** Two
launches produced a booted-but-idle sandbox whose `LogonCommand` never fired: `vmmemWindowsSandbox`
from the previous run was still alive, and killing `WindowsSandboxServer` /
`WindowsSandboxRemoteSession` does **not** take the VM down with it. Flat VM working-set (982 →
984 MB over 30 s) is the tell — an installing sandbox climbs. **Wait for every
`vmmemWindowsSandbox` to disappear before relaunching**, or the gate reports nothing and looks hung.

---
## 2026-08-06 (2) — the readiness gate no longer gives up, and stops telling users to run `just api` (KI-39)

**What changed.** New `lib/shell/startup.ts` (pure, tested) + the readiness `$effect` in
`App.svelte` + the status-bar copy. Frontend only — the sidecar is untouched, so the rebuild reused
today's freeze.

**Why.** Characterising the "first-launch dead-backend window" with the RG-012 numbers in hand
turned up three facts that are individually defensible and jointly a bad first five minutes:
1. **A 60 s budget** (60 polls × 1 s) against a PyInstaller **onefile** sidecar that extracts
   ~1.5 GB to `%TEMP%` before uvicorn binds.
2. **No retry.** The `$effect` reads no reactive state before its `await`, so it runs **once per
   mount**; after the loop fell through there was no timer and no control to trigger one. Only a
   relaunch recovered.
3. **The message was a developer instruction** — `backend unreachable. Run just api`. A task runner
   and a repo recipe that someone who installed an `.exe` does not have. **The app's only failure
   message asked for something that cannot exist on the machine showing it.**

The margin was thinner than it looked: RG-012 measured health at **~30 s — half the budget** — on an
idle VM, NVMe, file cache warm from the install that had just written those bytes. Defender scans
`%TEMP%`. Exceeding 60 s on a tester's machine is plausible, and that case was not "slow" but
terminal *and* misdirecting.

**The fix is the removal of a terminal state, not a bigger number.** Polling now backs off
(1 s → 2 s → 5 s, bounded) and **never stops**; `startupPhase` only changes what is *said*:
`connecting` → `slow` at 20 s ("a first launch can take a minute…") → `stalled` at 90 s, which shows
the fault colour **while still polling**, so a backend that turns up at minute three lands the app in
`ready` by itself.

**Verified live, on the exact scenario that used to be unrecoverable** — UI up with no backend at
all:

| elapsed | status bar | dot |
|---|---|---|
| 0 s | starting the engine… | wait |
| ~20 s | starting the engine — a first launch can take a minute… | wait |
| ~90 s | still starting — retrying. Restart the app if it never arrives. | **red** |
| backend started at ~170 s | 33,105 chunks · ollama/llama3.1:8b · bge-base | **ok** |

**and the page was never reloaded** (`performance.getEntriesByType('navigation')[0].type ===
'navigate'`). The old gate could not have done that from any elapsed time past 60 s.

**Rejected.**
- **Raising 60 → 120.** Trades one arbitrary cliff for another; the defect is that a terminal state
  exists at all, not its size.
- **Unbounded exponential backoff.** A late backend must still be noticed promptly; the delay caps
  at 5 s deliberately.
- **Dropping the red `down` state** so nothing ever looks broken. After 90 s something probably *is*
  wrong and saying so is honest — the fix is that saying so no longer means giving up.
- **A longer, fuller message.** The bar is `nowrap` + `text-overflow: ellipsis`, so a long sentence
  is simply cut. Both new strings were measured against the 375 px width (297 px and 307 px against
  335 px available); the fuller explanation is in a `title`.

**What it opens.** **RG-010 (cold start) has still never been properly recorded** — one measurement
on one fast VM is what we have, and that distribution is what decides whether the PyInstaller spec
should move onefile→onedir, which is the deeper fix. Filed in KI-39.

---
## 2026-08-06 (1) — **rebuilt the installer and RG-012 Tier-2 PASSED on it.** The release gate is closed

**What changed.** No source. A full artifact rebuild — sidecar re-frozen, installer re-bundled — and
the clean-machine gate re-run against it, because every fix since 2026-08-05 existed only in source
and the installer that last passed was built at 14:58 the previous day. **KI-34's whole lesson is
that a green source tree says nothing about a frozen binary**, so a release cannot rest on one.

**Build.** `uv sync --extra cpu --extra dev --extra packaging` (KI-3: `build_sidecar` refuses to
freeze a `+cu*` torch — the `cu130` wheel segfaults on a GPU-less box) → `just sidecar` (**847 s**)
→ `npx tauri build` (**601 s**). Venv restored to `--extra cu130` afterwards; `cuda_available True`
re-confirmed. **The sidecar came out at 1,562.1 MB — exactly the post-KI-34 reference** (1,545.5 MB
was the broken build), so the `pymupdf` data files are bundled; the size is the cheapest possible
regression check on that fix and it is worth keeping as one.

**RG-012 Tier-2 — PASS**, on a Windows Sandbox reporting `python on PATH? False`:

| step | result |
|---|---|
| silent install | **177 s** (330 s the previous day) |
| files laid down | `doc-assistant-desktop.exe` 10.7 MB · `doc-assistant-api.exe` **1,562.1 MB** |
| `/api/health` | **~30 s**, `chunk_count: 0`, `ollama/llama3.1:8b` |
| `/api/setup` | ollama reachable (9 models), `active_ready: true`, documents step correctly not-done |
| ingest 3 PDFs | **added=3, errors=0, 322 chunks**, ~36 s — KI-34 stays fixed |
| turn | **14 s**, 10 sources |
| citations | **5 resolved (5 canonical `[n]`, 0 labelled); 0 unresolvable** |

**Every 2026-08-05 fix is verified present in the frozen binary**, which is the point of the
exercise: `flagged_claims` badge as **`uncited` ×7 / `weakly grounded` ×3** (KI-37 — no
"unsupported" anywhere), and the card reading **`best source relevance` / `top-3 relevance span`**
with the old `**Reranker scores**` heading gone.

**The de-dup's fallback branch got verified for free, and by accident.** The card *did* print the
per-source list here — correctly: a fresh install has no concept graph, so
`_attach_source_evaluation` returns `None`, `source_eval` is null, no strip renders, and the card
is the only per-source surface. Between this run and the 2026-08-05 dev-app check, **both branches
are now exercised end to end** — strip present → card omits, strip absent → card keeps.

**The corrected gate also earned its keep immediately.** It reported
`5 canonical, 0 labelled, 0 unresolvable` instead of a bare count — and the model used canonical
`[n]` this time, on the same model and prompt that produced `[Source n]` the day before. **That is
the third independent confirmation that KI-35's premise was a one-off** (0/54 in the coverage runs,
plus this).

**Harness change.** `rg012-tier2.wsb` gained a `LogonCommand` so the gate runs unattended. The two
historical "LogonCommand never fires" failures were never Sandbox's fault — the script was
UTF-8-no-BOM and PowerShell 5.1 read it as ANSI. It is ASCII-only now and I byte-checked it (0
non-ASCII bytes) *before* trusting the command; the `.wsb` is byte-checked and XML-parsed too.

**Rejected.** Shipping the artifact that passed on 2026-08-05 — it predates every fix, so the thing
tested would not have been the thing shipped. Tier-1-only (skipping the answer engine) — it would
not exercise the cited-turn path, which is exactly what the fixes touched.

**⚠ Data loss, mine, recorded because the record must be honest.** Clearing `out\` for this run,
I issued the delete after an archive command had been **rejected as a whole** by a path-protection
guard — so its `New-Item`/`Copy-Item` never ran, and I did not check before deleting. The four
2026-08-05 artifacts are gone (`Remove-Item` does not use the Recycle Bin; no shadow copies). A
partial reconstruction recovered verbatim from the session that read them is at
`C:\rg012-host\out-2026-08-05-RECONSTRUCTED\README.md` — complete for the answer,
`flagged_claims`, `provenance_card_md` and `settings.json`; lines 3-26 of the log; fragments only
of the SSE stream. **No conclusion is at risk** (every decisive number was transcribed into KI-35/
36/37 and DEVLOG (2) beforehand, and the gate was re-scored against the real file while it
existed), but the raw provenance for a headline finding is not recoverable. **Copy, confirm the
copy, then delete — and never treat a failed command as a partially-applied one.**

**What it opens.** The gate now overwrites `out\` on every run; it should stamp its output into a
per-run folder instead. And `docs/desktop-packaging.md` still describes RG-012 as "paused" and
Tier 2 as blocked on the data-home decision — both stale, now twice over.

---
## 2026-08-05 (4) — pre-release UI pass: themed scrollbars, the answer column stops rendering under its own scrollbar, and "what is this 0.94?" answered

**What changed.** Three reported defects, all in the answer surface (user report with screenshot).

**1 · Scrollbars were unstyled — anywhere.** `grep scrollbar apps/desktop/src` returned **nothing**,
so every scrolling pane drew the raw OS bar: a bright grey slab against a dark reading surface,
and the loudest element on a page whose prose is deliberately quiet. Added `--scrollbar-thumb` /
`--scrollbar-thumb-hover` to `app.css` using the same `color-mix(in srgb, var(--fg) N%, transparent)`
trick `--graph-edge` already uses — **one declaration covers both themes**, because `var()` is
late-bound. Declared twice on purpose and they do not fight: `scrollbar-color` is the standard
property and wins in Chromium (WebView2, the Windows Tauri runtime) and Firefox — where it is set
to anything but `auto`, Chromium ignores `::-webkit-scrollbar` entirely — and the `::-webkit-` block
is the fallback for WKWebView, which has no `scrollbar-color`. Graceful degradation, not duplication.

**2 · The answer column rendered underneath its own scrollbar.** Measured, not guessed:
`section.conversation` had `padding: var(--space-2) 0` — **zero** horizontal — while scrolling, so
its 15 px vertical bar ate the content box (`width 790` → `clientWidth 775`). Every right-anchored
child sat exactly ON that boundary: `.usage` and the source-evaluation `.score` column both ended at
x=1206 = the content edge, gap **0 px**, which is why the screenshot reads "0 tokens · **loca**".
Fixed with a real gutter (`padding: var(--space-2) var(--space-3)`) plus `scrollbar-gutter: stable`,
so the space is reserved even when no bar shows and a growing turn no longer shifts the whole column
15 px left as it crosses the fold. Verified live: gap **0 → 11 px** for both elements, no body
overflow. *This was never a "cut window" — nothing was clipped horizontally; content was drawn under
an overlaying bar.*

**3 · "Is 0.94 a relevance score or a quality score?" — a fair question the UI invited.** It is
**retrieval relevance**: the cross-encoder reranker's score for that *chunk* against *this question*,
the same number the ranking used. **Nothing in this app scores a source's quality.** But the number
sat unlabelled, in a block headed *"Source evaluation"*, beside an epistemic assessment — and since
KI-33 withheld that assessment, the strip's only real content **is** a year and this score. So the
heading now promises an evaluation the row no longer delivers, and the bare decimal is the only
thing left to read as one. Fixed by naming it: a legend under the heading (*"relevance = how well
the retrieved passage matches your question (reranker score, 0–1). Not a judgement of the source."*)
and a small-caps `relevance` unit beside every number, so a lone decimal is unambiguous even when
the legend has scrolled away.

**And the repetition, which was real.** On a low-confidence answer the provenance card printed
`**Reranker scores** — [1] reranker 0.907 …`: the *same* measure, keyed the same way, one decimal
deeper, on the same answer as the strip. The card now omits it when the strip is on screen
(`_ProvenanceInputs.source_strip_rendered`, set from `source_eval is not None and bool(sources)` —
`Turn.svelte`'s own render condition). It **keeps** the aggregate signals (`best source relevance`,
`top-3 relevance span`), which appear nowhere else and are what the confidence verdict is derived
from. Renamed throughout: "reranker" is our word, not the reader's.

**Rejected.**
- **Dropping the per-source list unconditionally.** The strip needs a concept graph
  (`_attach_source_evaluation` returns `None` without one), so on a graph-less corpus the card is
  the *only* per-source surface. Pinned both ways by test, so the de-dup cannot become a data loss.
- **`overflow-x: hidden` on the conversation** to stop the clipping. It would hide the symptom and
  silently swallow any genuinely too-wide child (a table, a long code block) instead of letting it
  scroll. The defect was missing padding; fix the padding.
- **A tooltip alone for the score.** It already had one (`title="Reranker relevance score…"`) and
  the question still got asked — hover text is not documentation, and it does not exist on touch.
- **Renaming the strip's "Source evaluation" heading.** Tempting while the assessment is withheld,
  but the heading is right for what the strip is *for*; the honest fix is restoring the assessment
  (ADR-041), not renaming the feature around a temporary containment.

**What it opens.** The strip's heading over-promises while KI-33 holds — worth revisiting if the
containment outlives the release. And the **source explorer** the same report asked for (chunk →
parent → document) is filed in `docs/ui-checklist.md` §3, deliberately **not** built here: it is new
capability, not a defect, and the pre-release pass is for defects.

**Verified live**, real local turn (`ollama/llama3.1:8b`, $0, 12 s): score reads `relevance 0.87`,
legend renders, card shows `best source relevance 0.872` with **no** per-source list, 11 px gap to
the scrollbar, themed thumb applied. **Method note:** the first check appeared to *fail* — the card
still said "top reranker". The API sidecar had been started before the edits and uvicorn was not run
with `--reload`, so it was serving stale Python. **A frontend hot-reload proves nothing about the
backend**; restart the sidecar before believing a backend-rendered string.

---
## 2026-08-05 (3) — tell the user the free path cites less, where they choose it (KI-36 follow-through)

**What changed.** Four lines of copy in `apps/desktop/src/lib/settings/ProviderSetup.svelte`'s
Ollama card, beside the existing "needs ~5 GB of disk, happiest with a GPU" trade-off sentence:
the measured citation-coverage gap, with the run size and corpus named so it can be reproduced.

**Why.** Entry (2) measured it: on the same prompt and the same retrieval, `llama3.1:8b` cites 36%
of its sentences and `qwen2.5:7b` 14%, against 81% for Haiku. The integrity layer is right to flag
the difference — but a first-run tester on Ollama meets a wall of `uncited` badges with nothing
anywhere telling them the provider is the reason. **The measurement is useless to the user if it
only lives in the DEVLOG.** `ProviderSetup` is both the first-run surface and the ongoing switcher
(it is embedded in `Settings.svelte`), so one placement covers both moments of choice.

**Tone, deliberately.** Indicative and reproducible, not a verdict: it states *n*, the corpus size
and that prompt and retrieval were held constant, and it closes with what stays true —
*"Answers stay grounded in your documents; more claims will simply show as uncited."* Inform, don't
block: nothing is gated, the local option is not discouraged, and the number is the user's to weigh.

**Rejected.**
- **A computed field on `ProviderReadiness`** (backend → wire → `types.ts` → component). It is not
  a verdict the backend can recompute — it is a benchmark result — and three layers of plumbing for
  a static string is drift surface, not thinness. The card's existing trade-off copy sets the
  precedent, and a comment at the line points at the DEVLOG method so the numbers cannot rot
  silently.
- **Warning at answer time instead** ("many claims are uncited because you are on a local model").
  Better targeted, but it fires when the user can no longer act cheaply; the choice point is where
  the information changes a decision. Worth revisiting *in addition*, not instead.
- **Naming no model.** Vague hedging ("local models may cite less") is exactly the tone the project
  rejects — the numbers are measured, so print them.

**Verified live** (dev server, mocked nothing — this is static copy): renders in the Ollama card
between the trade-off line and the host line; identical colour treatment to its sibling paragraphs
in **dark and light**; at **375 px** its box is byte-identical to the two pre-existing `p.detail`
siblings (`left 481, width 284`) — the overlay's own pre-existing offset at that width, not this
change — and the body never scrolls horizontally. `svelte-check` 188 **0/0**, `npm test` **73/73**.

**What it opens.** The Settings overlay does not reflow at 375 px (pre-existing, seen while
checking this). And the frontend's health poll retries **21 times against a 500** while uvicorn
binds — the cold-start window the baton lists as item 2, now with a number.

---
## 2026-08-05 (2) — KI-35 was a **bug in the gate, not the app**: RG-012 Tier-2 had passed. The real defect is citation *coverage* (KI-36/37/38)

**What changed.** `.claude/KNOWN_ISSUES.md`: KI-35 rewritten as a corrected diagnosis; **KI-36**
(citation coverage), **KI-37** (the `unsupported` collision, fixed), **KI-38** (the `load_dotenv`
override hole) filed. The RG-012 gate at `C:\rg012-host\script\rg012-run.ps1` fixed (local-only
harness). Two code changes, both consequences of the investigation rather than of the original
report: the claim badge split (`helpers._claim_badge` + `ClaimReview.svelte`) and the citation
contract pinned across the wire (`apps/desktop/src/lib/chat/citations.ts` extracted from
`Markdown.svelte`, `tests/fixtures/citation_vectors.json` read by **both** suites).
**Nothing in the citation *parser* changed — it was already right.**

**Why.** KI-35 was the baton's #1 next action and it described a defect that does not exist. It
claimed `llama3.1:8b` cites `[Source 1]`, that neither parser resolves that, and that the integrity
layer therefore "inverts" on the shipped local path. Checked against the run's own archived
`result.json`, with the shipped code: `cited_source_numbers` → `[1, 5, 2, 1]`; `audit_citations` →
`valid=[1,2,5]`, `malformed=[]`, `clean=True`; the `[Source 5]` claim scored **`weakly grounded`**,
not `unsupported`. **Both parsers have tolerated `[Source n]` since 2026-07-14** —
`synthesis.py:35` and `Markdown.svelte:52`. The 13 flagged claims were the sentences that genuinely
carried no citation: the model cited **4 of 16**. The integrity layer reported the truth.

**The actual defect was one line of the gate.** `rg012-run.ps1` counted `'\[\d+\]'` — a *stricter*
contract than the app implements — logged `FAIL: answer produced but not cited`, and that verdict
was filed as an app bug. Re-scored with the app's own token: `resolved=4 canonical=0 labelled=4
unresolvable=0` → **PASS**. **RG-012 Tier-2 passed on 2026-08-05**; the release's packaging gate has
been green since KI-34 was fixed. The gate now separates three outcomes — resolved / unresolvable /
no-bracket-at-all — because they need completely different fixes.

**The rule, which is the reusable part: a verification gate must call the contract, never restate
it.** A restated contract drifts, and because a gate is trusted, its false verdict gets filed
against the code it was meant to protect. Same class as KI-34 one level up (there the gate tested
too little; here it tested something never promised).

**Then the measurement KI-35 should have been.** 35 private cases → the real `ChatController` on
the live 97-doc corpus, provider forced local, $0.
- **Retrieval first, to avoid confounding it:** recall 28/35. **All 7 misses are 4 documents the app
  already labels degraded** (`middleton-2001.pdf` 0 chunks, `hubel_wiesel_1959.pdf` 1, `hodgkin_
  huxley_1952.pdf` 7, `hebb_1949.pdf` 16). **Recall on healthy-document cases is 28/28.** That is
  ROADMAP **EX1** / ADR-039's OCR case, now quantified: degraded documents are the *entire*
  retrieval-recall gap, and they are all pre-1970 scans.
- **Citation coverage on the remaining 27** (`llama3.1:8b`): pooled **79/217 = 36.4%**, and
  **bimodal** — 11 answers cite nothing at all, 9 cite ≥85%, almost nothing between. 8 of the 11
  zero-citation answers are substantive uncited assertions; 3 are correct refusals.
- **The comparison that decides it** — same prompt, corpus, retrieval and cases:

  | provider / model | pooled coverage | median | answers citing **nothing** |
  |---|---|---|---|
  | `anthropic/claude-haiku-4.5` | **155/191 = 81.2%** | 0.875 | **0 / 27** |
  | `ollama/llama3.1:8b` (shipped local default) | 79/217 = 36.4% | 0.545 | 11 / 27 |
  | `ollama/qwen2.5:7b` | 40/296 = 13.5% | 0.000 | 19 / 27 |

  **So the prompt is not the defect** — a sixth citation rule is not what separates 81% from 36% —
  and `llama3.1:8b` is already the better of the two locals. **KI-36 is a local-model capability
  floor: an honest limitation of the free path to be *documented at the provider choice*, not a bug
  to fix before release.** The integrity layer is working; it correctly reports that a local model's
  answers are largely uncited. What a first-run Ollama tester lacks is any explanation of why.
- **`[Source n]` appeared 0/81 times** across all three models — KI-35's premise does not reproduce.
- Three *real* unresolvable forms showed, all rare and none of them the one filed: `Source 6
  [file.pdf]` (number outside the bracket, filename inside), `[2][11][17]` against 10 sources, and —
  on **Haiku** — claim text wrapped in brackets, `[a Bayesian non-parametric model …][7]`, the exact
  anti-pattern `prompts.py:53` forbids. That last one makes the audit **cry wolf**: the citations
  resolve (coverage 1.000) but the seven phrase-brackets are counted as failed citation attempts, so
  a fully-cited answer renders "⚠ 7 malformed citation(s)". A bracketed phrase immediately followed
  by a resolvable token is a style violation, not a citation attempt. Recorded in KI-36, not fixed.

**KI-37, found in passing and fixed.** The RG-012 card renders the reviewer's **"unsupported
claims: `0`"** directly above **"⚠ 13 claim(s) to review … *(unsupported)*"** — one word, two
meanings, same view. Worse, `claim_marker` labelled a *correct refusal* `unsupported` (3 answers,
16 badges), accusing the model exactly when it did the right thing. The badge now says what the
structural marker actually found — **`uncited`** (no citation token) or **`unresolved citation`**
(cites only numbers mapping to nothing) — the same three-way split the gate now makes. Presentation
only: `MARKER_UNSUPPORTED` and the persisted `AnswerClaim.marker` are untouched, so there is no
migration and `test_adjudication_persistence`'s marker triple still holds. `ClaimReview.svelte` now
tests for the one *benign* label and defaults the rest to `bad`, so a future severe badge cannot
silently render as mild.

**The contract is now pinned across the wire.** `[n]` had three implementations (Python, Svelte,
the gate) and a test on only one. The Svelte regex is extracted to a plain, dependency-free
`lib/chat/citations.ts` — the module kind `node:test` can actually run — and both suites now assert
the **same** `tests/fixtures/citation_vectors.json`. Verified the pin bites: deleting the
`[Source n]` tolerance from the TS side fails 3 frontend tests, naming the vector. (The component
imports it extensionless and the test imports it with `.ts`: `svelte-check` rejects the extension,
`node:test` requires it.)

**Rejected.**
- **Widening the parser** (KI-35's proposed mitigation): already implemented, and it changes nothing
  — the dominant failure is answers with no citation of any form, which no parser can reach.
- **Editing the prompt now.** It already carries five explicit citation rules and one of them names
  this exact confusion; a sixth on intuition is not a fix. Moving coverage is an eval-harness
  experiment with a control (rigor-gate), not a prompt tweak.
- **A refusal detector** for KI-37 — a heuristic wrong in both directions; renaming the badge to
  what it measures makes the refusal case merely true, with no detector to get wrong.
- **Archiving KI-35 as a resolved row.** The correction *is* the issue; compressing it to a row
  would drop the lesson and leave the original wrong story as the memorable one.
- **Filing the degraded-document finding as a new KI** — EX1/ADR-039 already own it; this run adds
  evidence, not a new issue.

**What it opens.** (a) **Tell the user why the free path cites less** — the measured 81/36/14 split
belongs where the provider is chosen (first-run setup / provider picker), in the "inform, don't
block" register. That is now the main open item from this work, and it is a UX change, not a
retrieval or prompt one. (b) **The audit should stop crying wolf** on a bracketed phrase that is
immediately followed by a resolvable citation (Haiku's one bad answer). (c) **EX1/OCR now has a
measured payoff** — 7 of 7 retrieval misses, all pre-1970 scans. (d) The gate still restates the
contract in PowerShell; the honest end-state is the API exposing the audit structurally so the gate
asserts on the app's own verdict instead of a third regex. (e) Unmeasured: run-to-run variance
(single repeat per model), and whether Sonnet differs from Haiku.

**Method note.** Two harness bugs of mine preceded the results and both would have corrupted them:
the first smoke run picked the two *worst* cases (both `hodgkin_huxley`, a `marginal` document) and
read 0% coverage as a citation defect — it was a retrieval miss; and a `Set-Location` for `npm test`
leaked into the next run's working directory. **Also: `config.py`'s `load_dotenv(override=True)`
means `LLM_PROVIDER=ollama <cmd>` silently runs on Anthropic and bills** — filed as KI-38, with the
seam that does work (`app_settings.get_llm_selection`, plus a separate line for the pinned reviewer).

---
## 2026-08-05 (1) — RG-012 Tier-2 finally ran, and the shipped installer **could not ingest a single PDF** (KI-34)

**What changed.** One line of `scripts/doc_assistant_api.spec` — `"pymupdf"` added beside `"fitz"` —
plus **KI-34**. The gate itself ran for the first time in the project's history.

**The finding.** Provenote 0.4.1 installed on a clean, Python-free Windows Sandbox. Install ✅,
launch ✅, backend ✅, `/api/setup` ✅ (Ollama detected reachable, 9 models, `active_ready:true`,
documents step correctly not-done). Then ingest of 3 PDFs: **`added=0, errors=3`, in 0.65 s** — far
too fast to have attempted extraction. Every file, identically:

```
[Errno 2] No such file or directory:
  ...\Temp\_MEI82922\pymupdf\layout/resources/onnx/layout_rf2.4.1+imf1.yaml
```

**Cause.** The spec collected **`"fitz"`** only. `fitz` is the *legacy import shim*; PyMuPDF's real
distribution directory is **`pymupdf/`**, carrying data read at extraction time. `collect_all("fitz")`
bundles the shim and none of it, so the frozen build **imports cleanly and then fails every PDF**.
Verified after the fix: `collect_all("pymupdf")` yields **129 data files**, including the exact
`layout_rf2.4.1+imf1.yaml` the error named.

**Why five layers of green missed it, which is the part to carry.**
1. **Invisible from source** — site-packages has the file, so the 1447-test suite, the eval harness
   and the desktop dev loop all pass.
2. **The v0.4.0 WSL clean-room run passed** for the same reason: it was a *source* install.
3. **The standalone sidecar smoke this session ran passed too** — `/api/health` → 33,105 chunks —
   because the missing file is read on the **extraction** path, not at import. **Booting a frozen
   binary proves nothing about whether its data files were bundled.** A packaging gate must push a
   real document end to end.

**Two more findings from the same run, neither fixed.** (a) **First launch has a long
dead-backend window** while 1.5 GB of onefile extracts before uvicorn binds — health lands in 10–20 s
warm, but minutes cold, and the UI does not signal it convincingly. A tester's first impression.
(b) The install folder is `Provenote/` while the executable is `doc-assistant-desktop.exe` — ADR-012's
product/code identity split working exactly as designed, but undocumented and it cost two diagnostic
rounds here.

**Method note.** Three of my own harness bugs preceded the real one and each was mine, not the
product's: a UTF-8-no-BOM script that Windows PowerShell 5.1 read as ANSI (em-dashes broke parsing —
and *that* was the true cause of the two "LogonCommand never fires" failures I had blamed on Sandbox);
`Select-Object -First 1` picking the **June 0.1.0** installer out of a folder holding both; and
filtering for a guessed `Provenote.exe`. **Scripts that drive a gate are ASCII-only and never select
an artifact incidentally** — both now enforced in the harness at `C:\rg012-host\`.

**Rejected.** Calling the earlier "backend unreachable" a product bug — measured twice at ~10 s and
~20 s to health, so it was extraction latency, not a failure. Declaring RG-012 Tier-2 passed on the
strength of install + launch + health: the gate's actual claim is **a cited turn**, and ingest fails,
so the answer is still unproven.

**What it opens.** Rebuild (sidecar + installer) and rerun in a **fresh** sandbox — the gate should
now reach the cited turn. Then: the first-launch extraction window deserves its own fix before any
tester sees it, and the packaging gate should grow a real-document step so this class cannot recur.

---
## 2026-08-03 (3) — v0.4.1: the first installer since June, KI-33 contained before it ships — and **RG-012 Tier-2 still has no evidence**

**What changed.** KI-33 containment (`config.py` default + `SourceEvaluation.svelte`), version 0.4.1
across **seven** strings, a CHANGELOG entry, `@tauri-apps/cli` added as a devDependency, and two
release artifacts. User priority for this stretch: *"focus on the binary release … to finally have a
beta-release"*, with the KI-33 surfacing fix landed first.

**The containment, and why it is a default rather than a deletion.** `EPISTEMICS_MARKERS_ENABLED`
**true → false** — the same lever R7 used for KI-7 containment, now for a defect one layer down —
and the strip's coverage + `superseded` chips commented out with the reason at the line, markup and
CSS together so restoring is one contiguous uncomment. **All three coverage values go, not just
`contested`:** `ns` and `nc` both derive from `stance_by_doc`, so `corroborated` and `unique` inherit
the same defect. The strip keeps what is sound — year, relevance score, graph freshness.

**Non-vacuous by construction.** The flip failed **exactly one** test —
`test_markers_enabled_by_default`, the one encoding the old default. Renamed to
`test_markers_disabled_by_default` and paired with a new
`test_markers_still_available_when_explicitly_enabled`, so the suite now pins **both** the new
default and that the opt-in still works. A containment nobody can prove reversible is a deletion.

**The release.** Sidecar re-frozen on a CPU sync (KI-3): **1545.5 MB**, replacing a **2026-06-24**
build — pre-rename, pre-icon, pre-ADR-034. Smoke-tested standalone *before* bundling, which is the
step that catches what tests cannot: `/api/health` in ~30 s, **chunk_count 33,105**, no frozen-import
failures. Then `Provenote_0.4.1_x64-setup.exe` (**1555.4 MB**) and `Provenote_0.4.1_x64_en-US.msi`
(1546.7 MB) — the first installers since June and the first carrying the Provenote identity.

**A root cause worth naming: the build recipe had rotted because it was never declared.**
`npx tauri build` failed outright — `@tauri-apps/cli` was not a devDependency, not global, not
anywhere. The June installer was built against undocumented machine state, so "how to build the
installer" was unreproducible the moment that state changed. Now pinned at `^2.11.4` in
`devDependencies`. **This is the same class of failure as the `uv.lock` miss** — a build input that
nothing declared and nothing checked.

**⚠ RG-012 Tier-2 IS STILL OPEN, and this session produced no evidence about it.** Two Windows
Sandbox launches, ~30 minutes: the `.wsb` parses, all four mapped folders resolve, the output folder
is host-writable, the VM boots and burns CPU — and `LogonCommand` writes **nothing**, even hardened
to sleep 25 s and then immediately create a file before doing anything else. It is not executing in
this Sandbox configuration. Root cause unknown. **Nothing here licenses any claim about the
installer on a clean box**, and the CHANGELOG's *Known limits* says so in the release itself. The
harness is left at `C:\rg012-host\` (ASCII path on purpose — `.wsb` parsing is unreliable with the
accented profile path) with a self-contained `rg012-run.ps1` that installs silently, seeds three
PDFs, ingests, asks one question and writes a PASS/FAIL verdict.

**Machine state touched and restored.** Ollama was rebound to `0.0.0.0` (user-approved) so a sandbox
could reach it, and is back to **127.0.0.1** with the env var cleared. The venv was CPU for the
freeze and is back to **`cu130`, CUDA available**.

**Rejected.** Shipping the binary as 0.4.0 — the user's choice, made when the delta was docs-only;
landing KI-33 first made the binary behave differently from the `v0.4.0` tag, so it was re-raised and
became **0.4.1**. Fixing the `postcss` advisory (build-time only, via `vite@6`, processes only
first-party CSS, and a clean fix exists) — deferred rather than applied because it would have changed
the toolchain underneath the gate run testing that exact build.

**What it opens.** RG-012 Tier-2 needs a path that does not depend on `LogonCommand`: a hand-run in
the sandbox, computer-use driving the VM, or a real second machine. Until one of them produces a
cited turn, **this is a beta by its own CHANGELOG**. Then: `npm audit fix`, and the KL1–KL4 plan.

---
## 2026-08-03 (2) — Full review of the knowledge layer against the stated goal: **the acquisition half has no implementation**, and the one suggestion engine runs on the detector graded noise

**What changed.** No source code. New `docs/PLAN_2026-08-03_knowledge-layer-to-goal.md` (local-only,
ADR-029) — the in-depth review; new **§6b State of play** in the tracked `docs/knowledge-layer.md`;
new ROADMAP rows **KL1–KL4** so the plan's items are visible in git rather than only in a gitignored
file.

**Method.** Read every governing artifact rather than working from memory: ROADMAP rows S1–S2 ·
G1–G8 · E0–E5 · TX1–TX3 · MM1–MM3 · PF1–PF4; ADR-004/008/015/017/018/027/028/030–033/036–041; the
concept-graph, gap-detection and taxonomy specs; RG-014/015/019; KI-18/19/33.

**The goal decomposed into five testable capabilities** — C1 unsubstantiated claims · C2 per-concept
classification · C3 gap exposure · C4 **acquisition direction** · C5 navigation — then mapped onto
every built component.

**Finding 1 — C4 is the goal's operative capability and has no sound implementation anywhere.** The
goal's verb is *"should find more resources and documents."* **Every built detector looks inward** at
what the corpus already holds. ADR-004 named the outward half precisely and deferred it — Tier-2b
needs *"a representation of 'outside the known space'"* — and ADR-032 has been a stub since. No
amount of repairing the inward detectors closes this; it is the largest distance between the product
and the goal.

**Finding 2 — the one suggestion engine runs on the one detector graded noise.** `gap_suggest` (G5)
is the closest thing to C4 in the tree, and it fires one LLM call per **`under_connected`** concept —
the kind RG-014 graded ❌ *"mostly noise… measures graph degree, dominated by vocabulary sparsity"*.
Meanwhile **`single_source`**, the kind RG-014 graded *"TRUE POSITIVE — the product thesis"*, gets
**no suggestion pass at all**. Re-pointing it is hours of work and the cheapest real progress toward
the stated goal.

**Finding 3 — C1 already works, on the wrong layer.** The answer path classifies claims by
retrieval-derived support and says outright why (*markers never come from model confidence*). The
concept layer ignores it and asks an LLM about labels. That is ADR-041 option 6, now KL1.

**Finding 4 — C3 and C5 are genuinely strong**, which is worth stating plainly after two sessions of
finding defects: `single_source`, the graph, ego navigation, the gap list with durable triage, the
Connections panel and the taxonomy view are all built and sound. The per-document map (MM1–MM3,
absorbing PR-G2c) is the one navigation piece missing, still gated on the ADR-030 stub.

**The plan, phased so nothing is built on top of something that lies.** **A** make what exists tell
the truth (the surfacing deadline · option 6 · `unsourced_claim` contamination · encode RG-014's
grades in the gap list) → **B** the acquisition half (re-point `gap_suggest` · grill ADR-032 · the
taxonomy as reference class) → **C** the per-file map (grill ADR-030 → MM1 → MM2 → MM3) → **D**
Node-B rebuilt on evidence, ground-truth study as the gate → **E** measure the two unmeasured shipped
layers (RG-015 taxonomy placement, RG-018 wiki flip). **B before C** deliberately: B1 is hours and
moves the operative capability, C is weeks and improves navigation that already works.

**Rejected.** Putting the whole review in the tracked docs (the dated-PLAN convention is local-only,
ADR-029) — mirrored as §6b + KL1–KL4 instead, so the conclusions survive outside a gitignored file.
Attaching effort estimates (the ordering is by dependency and goal-value; guessing hours would have
dressed judgement as measurement).

**What it opens.** Five things the review did **not** verify, listed in the plan's §6 — most
importantly that `gap_suggest`'s restriction to `under_connected` is taken from CONTEXT.md and
ADR-004 and **not re-read in `gap_suggest.py`** (confirm before doing B1), and that RG-014's verdict
dates from 76 docs / 26 concepts against today's 97 / 13 — direction stable, numbers not.

---
## 2026-08-03 (1) — ADR-041 (rebuild-or-retire Node B) + the knowledge layer finally has a map with a trust table

**What changed.** No source code. New **ADR-041**, new **`docs/knowledge-layer.md`**, a new
**"Read before you touch — never assume"** section in `.claude/CONTEXT.md`, plus pointers from
`AGENTS.md`, `architecture.md` and `src/doc_assistant/CLAUDE.md` so none of it depends on knowing
it exists.

**The CONTEXT.md table is the durable half.** Eight rows mapping *area of the app* → *what to read
first* → *what assuming instead has actually cost* (the epistemics threshold chase; reverting the
lazy reranker or the `_sparse is None` guard; page markers in the evidence block; curated structure
in `concept_edges`; a "local" run billing the API). Plus two standing rules: **a spec's surface
description is not its purpose** — with today's retire recommendation as the worked example, and the
corollary that *months of deliberate work on a feature is evidence of intent, so ask what it is for
before proposing to remove it* — and **re-measure per box; check a "known" fact before inheriting
it** (three inherited claims were false this week: the private eval set, retrieval determinism, and
`gh`/Docker being absent).

**Also answered, since it came up as "I don't know if it was done":** the *explore concepts within a
given file* feature is **not built**. It was **PR-G2c** (Library entry, doc → its concepts);
`feature-concept-graph.md:19` records E4 shipping a related-papers Connections panel instead, and it
has since been absorbed into **ROADMAP MM2** (`knowledge/doc_map.py` + `GET
/api/library/documents/{id}/map`), gated on **ADR-030** — still a proposed stub needing `grill-me`,
and already flagged in the baton as the one to do first because it blocks MM1.

> **Date correction.** The three entries below are headed `2026-08-02`; that is wrong — **all of that
> work was done on 2026-08-03**, in the same session as this entry, and the two baselines it
> committed carry `2026-08-02` in their filenames for the same reason. Left as-is rather than
> rewritten: the entries are append-only and already committed (`40888b1`), and a rename would
> break the links pointing at them. Corrected here so the record is not silently off by a day.

**Why the doc, and why now.** The user's read was that the docs are behind what we have on the
concept graph — *"it is not clear what we are doing and why."* Checking rather than agreeing: the
purpose **is** written down, and well. `docs/specs/feature-concept-graph.md` § *The job* (locked with
the user 2026-07-17) states it as three questions — **corroboration** (*"is this concept backed by
more than one source?"*), **coverage**, **navigation** — plus ADR-004's north star, *"the graph is
the substrate; the gaps are the payload."* The mechanism is in `architecture.md`; the decisions are
in nine ADRs. **Nothing connected them**, so nothing noticed when an output stopped matching the
purpose. That is not hypothetical — it is precisely how `contested` shipped saturated and how
RG-019's prescription went two weeks unchallenged. The missing artifact was never a description; it
was a **trust table**.

`docs/knowledge-layer.md` is that page: the job · the one-vocabulary rule · the two graph layers ·
the end-to-end flow · who consumes what · **a per-signal trust table** · how to run it · the ADR
reading order. It grades `single_source` ✅ (RG-014's "TRUE POSITIVE — the product thesis"),
`unsourced_claim` ⚠️ (~33% contaminated), `under_connected` ❌ (noise at small vocabularies), and
`contested`/`superseded_trend` ❌ **not a corpus measurement** (KI-33).

**The finding that shaped ADR-041, and it came from re-reading the spec rather than the code.**
**All three of B1's jobs are answered by counting documents** — corroboration is `len(doc_ids) >= 2`,
coverage is presence per field, navigation is `node → doc_ids → chunk_keys`. All Node-A,
deterministic, zero-LLM. **Stance answers a question B1 never asked** ("do the sources *disagree*?"),
and it arrived later, via the 7d currency work and ADR-027's strip. The spec's own grounding note
from design-lock records *"The epistemic dimension is empty … `contested_edges()` → `[]`"* — **the
feature was designed, locked and graded useful before stance existed at all.**

**So ADR-041 is not only "how to rebuild Node B" — it is whether to.** Five options: rebuild with the
co-occurrence passages + a neutral label + one pair per call · **retire stance, keep Node A** ·
keep the relation verb only · a deterministic tension proxy · park it. **Recommendation: retire**,
reopening as a properly-scoped feature (with the ground-truth study budgeted from the start) only on
an explicit product decision. Option 3 is explicitly warned against as a false compromise — the
relation verb comes from the same text-free, position-sensitive prompt (the position probe produced
*is used with · uses · is improved by · compared to · is compared to · improves on* for one pair) and
`relation_by_pair` keeps whichever document answered first.

**Costs stated honestly rather than waved through.** Retiring takes `contested` and
`superseded_trend` out of the product — the CHANGELOG feature list, ADR-027 D3's strip column, the
reviewer's `contested_evidence` tag — and **G3/G6's year-aware `superseded_trend` is collateral**: it
is deterministic and correct in itself but rides on stance-derived direction, so it goes too unless
re-based on `doc_years` alone. Whether that re-basing is feasible **has not been checked in code**,
and ADR-041 says so in its Confidence section.

**Rejected.** Writing a new "what is the concept graph" doc from scratch (the purpose was already
written; duplicating it would have created a second source to drift). Agreeing that the docs were
missing without looking — they were **scattered and stale, not absent**, and the fix for those is
different.

**Then the user supplied the decision input the ADR was waiting on, and it flipped the
recommendation.** The stated intent: *"see which claims are unsubstantiated and where are the
knowledge gaps … classify knowledge per concept in order to find the gaps where the user, for a given
subject, should find more resources … We want epistemics feature. That is the idea."*

**My "retire" recommendation was wrong, and the way it was wrong is worth keeping.** I read B1's spec
text — corroboration/coverage/navigation, all document counts — and concluded stance "answers a
question B1 never asked". That is true of the *text* and false of the *intent*. **A spec's surface
description is not its purpose**, and I had just written a whole page arguing that the docs' problem
was exactly this kind of disconnection. The measurement's finding is untouched (the current Node B
cannot serve the goal); what changed is that "this implementation is invalid" and "the feature is
unwanted" are different claims and only the first was ever supported.

**Two things the correction produced that the retire framing had hidden.**
1. **A unit mismatch nobody had named.** The goal is knowledge classified *per concept*; today's
   epistemics classifies **edges** (concept pairs) and reaches concepts only by aggregation.
2. **A cheaper first move — new ADR-041 option 6.** *"Which claims are unsubstantiated"* is a
   **support** question, not a polarity one, and the project already has a working, deterministic,
   retrieval-derived claim layer (`AnswerClaim`, `weakly grounded`/`unsupported`, `unsourced_claim`)
   built on the principle `how-answers-work.md` states outright — markers come from retrieval
   signals, never the model's own confidence. Re-basing per-concept status on it serves the headline
   goal with no new LLM pass. It cannot do polarity, so option 1 still owns "these sources disagree".

**Recommendation now: 6 → 1**, with option 4 (deterministic structure from the taxonomy) kept in view,
because the user's *"concepts are linked in general to predictable things"* makes a gap **a deviation
from expected structure** — which needs a source for the expectation, and ADR-028's curated taxonomy
is the one already in the tree. Same reference-class argument ADR-040 reached from the other side.

**What it opens.** The build sequence is open; the ADR no longer is. One thing keeps its deadline
regardless: **the surfaces must stop presenting stance-derived output as an epistemic finding** until
6 or 1 lands. Unscoped: whether `superseded_trend` survives on years alone.

---
## 2026-08-02 (3) — ADR-040 option 5 executed: Node-B stance is judged **without the document** and flips with **list position**. `contested` is not measuring the corpus (KI-33)

**What changed.** No source code. New instrument `scripts/validate_node_b_stance.py`, new baseline
`tests/eval/baselines/node_b_stance_validity_2026-08-02.md`, **KI-33** filed, ADR-040 given an
*Update* section that blocks every surfacing option behind a Node-B fix.

**Why.** Entry (2) concluded `contested` was a surfacing problem and put "validate the stance
extractor" first because every other option's value depended on the answer. It ran. The answer is
worse than expected: the signal is not a measurement of the corpus at all.

**Two structural facts, from the code, before any measurement.**
1. **The model never sees the document.** `build_messages(present_labels, pair_labels)` composes the
   entire user turn from concept labels and a numbered pair list — `annotate_relations`' own
   docstring says so — while the system prompt asks for stance *"from the document's apparent
   framing"*. There is no document in the prompt to have a framing.
2. **There is no neutral stance.** `POLARITIES` = supports/refines/contradicts/supersedes, two of
   them opposing, all mandatory. Citation-polarity corpora put neutral above 60% as the *majority*
   class. The vocabulary cannot express the common case, and its boundary is a hair wide: `refines`
   ("improves") supports, `supersedes` ("replaces") opposes, and the prompt's own example verb is
   *"improves on"*.

**The controlled experiment — one variable, four verdicts.** One document, same 7 present concepts,
same 17 pairs, `llama3.1:8b`, temperature **0.0** (all shipped settings), varying **only the target
pair's index** in the numbered list:

| index | 0 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| stance | supports | supports | **supersedes** | **contradicts** | **supersedes** | refines |

Four distinct verdicts **crossing the supporting/opposing boundary**, from a list position. Replaying
the five real documents' actual prompts reproduced **5/5** recorded stances, so this is the shipped
pipeline's deterministic behaviour, not sampling noise.

**A hypothesis refuted en route, kept because it cost a run.** The first guess was that the
*co-present concept set* drove the variation. Nine realistic contexts with the pair held at index 0
returned **`supports` 9/9** — stable and deterministic. It was position, not context. Two experiments
to get there; *reasoning proposed, measurement decided* — the third time that pattern has paid this
week.

**Supporting evidence from the artifact.** The whole integrity layer is **65 stance assignments**,
**30.8% opposing**, and **14 of 19 annotated edges carry more than one stance** (`re-ranking <-> BM25`
takes all four across 5 documents). Relation and stance contradict each other: `is a component of` →
`supersedes` ×2, `uses` → `contradicts` ×3, `builds upon` → all four.

**This subsumes entry (2)'s domain confound rather than competing with it.** Pair-list length scales
with a document's concept count, so dense documents → long lists → deep indices → opposing →
`contested`; sparse documents → one pair → index 0 → `supports`. That *is* the 7/9-vs-0/4
parent-field table. `cre`, `dbs`, `ntsr1`, `pddl` were never settled — their documents yield one pair.

**Rejected.** Swapping the model (facts 1 and 2 are structural — any model inherits them); raising
`max_tokens` (the JSON parses; KI-28 was a different failure); tuning the threshold (measured inert
in entry 2). Also rejected: implementing option 2 now — it is still correct that `ns=0` is not
contested, but the node it fixes came from a single `supersedes` on a one-pair prompt, so it was an
artifact, not the "one real defect" entry (2) called it. **That correction is the honest cost of
having measured the input second instead of first.**

**What it opens.** A **Node-B redesign** with its own ADR: pass the passages where the two concepts
co-occur, add a neutral/no-stance option, remove the position dependence (one pair per call or
equivalent), and carry a hand-labelled ground-truth study on the RG-015 template — which is the only
way to make an accuracy claim, since nothing here scores against ground truth. Until then `contested`
should not be presented as an epistemic signal, and ADR-040's options 1/2/3/4/6 cannot be evaluated.
Unquantified and worth knowing: what *share* of the 30.8% opposing is position-driven rather than
prior-driven.

---
## 2026-08-02 (2) — RG-019 measured: the `contested` floor everyone planned to add is **inert**, and the saturation is a surfacing problem (ADR-040)

**What changed.** No source code, and deliberately so. New instrument
`scripts/measure_contested_density.py`, new baseline
`tests/eval/baselines/contested_density_2026-08-02.md`, new **ADR-040** with its decision left open,
and RG-019 rewritten from an untested hypothesis into a measured negative result.

**Why.** The v0.4.0 walkthrough recorded the integrity strip — the product thesis — reading as
noise: **53.3% of assessed chunks marked `contested`** against 3.9% `corroborated`, 8 of 10 sources
in a live turn marked contested on a question that is not a controversy. Two records already agreed
on the cause and the cure. RG-019: *"triggers on `nc >= 1` … derive a named floor (min disputing
docs and/or an agreement-ratio band — the `MIN_DATED_DOCS_PER_SIDE` pattern)"*. ADR-027 shipped the
always-on strip without it, noting the strip would otherwise "ship saturated". **Neither had been
measured, and both are wrong.**

**Measured, $0, every counterfactual re-projected onto the real 18,831 chunk segments.**

| lever | density |
|---|---|
| shipped (`nc >= 1`) | **53.3%** — reproduces the walkthrough's 396/743 exactly, which is what validates the instrument |
| `nc >= 2` — *the prescribed fix* | **53.0%** |
| `nc >= 3` | 52.9% |
| `agreement_ratio < 0.70` | 53.2% |
| chunk rule "majority of claims contested" | 53.2% |

**The prescription accounts for two chunks.** Only **1 of 7** contested nodes has `nc == 1`. The
other six are the corpus's core vocabulary — BM25, dense retrieval, passage retrieval, contrastive
learning, re-ranking, hard negatives — each with genuine two-sided stance across **5–11 documents**.
`contested` is not misfiring; it is firing correctly on ordinary scholarly disagreement that the UI
then presents as cautionary.

**Three findings that outlive this item.**
1. **The denominator was half-quoted.** 53.3% is of *assessed* chunks, and only **3.9% of the store
   carries any claim** — marked chunks are **2.1%** of the store. Both true; the first alone
   overstates the marker's reach ~25x. It is still the number a user sees, because retrieval returns
   the chunks that carry claims.
2. **There are two stacked `>= 1` thresholds, not one.** The node rule, and `derive_markers` marking
   a chunk if *any* claim is contested. But **89% of assessed chunks carry exactly one claim**, so
   any/majority/all are the same rule here (53.3 / 53.2 / 53.0%). `n_contested >= 2` does cut to
   7.9% — by requiring two contested concepts in a chunk that mostly mentions one. A structural
   silencer, not an epistemic threshold.
3. **`agreement_ratio` is the only lever with range and the one that must not be used.** `<0.60` →
   27.1%, `<0.50` → 0.5%. But the seven observed values sit in **0.545–0.714**: every effective
   threshold is fitted to seven points on a 13-concept vocabulary. That is the
   over-optimise-on-the-current-corpus failure KI-19 exists to forbid.

**One real defect found.** `knowledge distillation` — `ns=0, nc=1, agreement=0.000`: zero supporting
sources, one disputing. Coverage is decided **contested-first**, so the unique-source neutrality
rule (Decision 4, "a sole source is never contested") never gets to judge it. A node with no support
is not contested, it is unsourced. Structural, no constant, fixable in a line — **not** done here,
because it lands with whichever ADR-040 option is chosen.

**Rejected.** Landing `nc >= 2` to close the item (it would record a fix that changes 0.3 points and
spend the corpus-tuning budget doing it). Picking a surfacing option unilaterally — the measurement
kills option 1, it does not choose between "surface the ratio", "re-frame the label" and "validate
the extractor first"; that is the user's call and ADR-040 says so in its status line.

**The finding that reframed all of it, found by asking what the threshold was measuring *against*.**
Every graph concept carries exactly one ANZSRC parent field (ADR-028) — **13/13 placed**. Joined:

| parent field | concepts | contested |
|---|---|---|
| Machine learning | 6 | **4** |
| Data management and data science | 3 | **3** |
| Artificial intelligence · Neurosciences · Med. chemistry · Biochemistry | 1 each | **0** |

**7 of 9 concepts in the two IR/ML fields are contested; 0 of 4 outside them.** `cre`, `dbs`,
`ntsr1`, `pddl` are not uncontested because they are settled — they have **one source each**. The
marker tracks *how densely the corpus covers a field*, not whether a claim is disputed, and all
three levers operate on a per-concept rate whose dominant term is a variable that rate does not
contain. **That is why no cut point works, and it is a stronger statement than "the levers are
inert".**

**The literature was checked, and it is unfavourable to every lever.** `agreement_ratio` is raw
percent agreement — the statistic Cohen's κ and Krippendorff's α exist to replace, with Landis &
Koch's bands as the standard cautionary tale about arbitrary cut points. Meta-analysis, the
discipline that actually owns "do sources disagree" (Cochran's Q, Higgins' I²), offered its
25/50/75% bands as tentative, is cautioned against mechanical application by the Cochrane Handbook,
and finds I² non-discriminative for prevalence meta-analyses — the closest analogue; practice
reports it with τ² and a prediction interval. Citation-polarity corpora put neutral citations above
60% with contrasting/negative the **rarest** class, against ~45% opposing sources per node here —
independent evidence that Node-B, not the corpus, produces this. Partial pooling / empirical-Bayes
shrinkage toward a parent mean is the named method for the hierarchy, with James–Stein dominating
raw group means for k≥3.

**ADR-040 gained a sixth option — score contestedness against the parent field's base rate.** It is
the only option that addresses the confound rather than routing around it, it removes the tunable
(the reference class is derived from data), and **this project already made the same argument once**:
ADR-006 rejected absolute keyword frequency for contrastive termhood against a background
distribution. Deferred rather than rejected, for a falsifiable reason: four of six fields hold one
concept, so no field base rate is estimable yet — a blocker that expires as `graph_include` grows
past 13. The instrument prints the cross-tab, so the re-test is a re-run.

**A framing kept even if every option is rejected:** *insufficient evidence is a state, not a low
score.* The schema already encodes it (`unique` = sole source, held NEUTRAL, Decision 4) and
contested-first precedence is what steals it — which makes the `ns=0` fix the first instance of a
principle rather than a one-node patch.

**What it opens.** The recommendation recorded in ADR-040 is **5 → 2 → 3, with 6 as the target
shape**: validate the Node-B
stance extractor first, because `llama3.1:8b` biased toward disagreement would reproduce this entire
picture and nothing yet separates the two — and its calibration is already recorded as suspect
(flat `rating` output, `gap_suggest_ollama_2026-07-08.md`). Then the `ns=0` gate regardless. Then
prefer the continuous surface over a rename. Still owed and unchanged: RG-019's precision
spot-check (this run argues structural correctness, it never reads the chunks) and density at a
second corpus size — **monotonicity in corpus size, the original worry, is still untested**.

---
## 2026-08-02 (1) — the v0.4.0 release commit left `uv.lock` at 0.3.0; **CI has been red on `main` since**, and the Docker build could never have worked

**What changed.** Two config lines, no source code. `uv.lock`'s own project entry
**0.3.0 → 0.4.0** (produced by `uv lock`, not hand-edited — the diff is exactly one line, zero
dependency churn), and **`README.md` removed from `.dockerignore`**.

**Why — and this is the part worth carrying.** The session opened on the baton's item 1,
`docker compose build`, the verification owed for `a052703`. The build never ran (Docker Desktop on
this box will not start its engine, below), but the two things that would have failed it were found
anyway, and the first is much larger than the Docker item.

**1. `uv sync --locked` fails at the v0.4.0 tag.** `47aabdd` bumped five version strings —
`pyproject.toml` · `package.json` · `tauri.conf.json` · `Cargo.toml` · `Cargo.lock` — and
**`uv.lock` was not one of them**. A uv lockfile records the project's *own* version, so it went
stale the moment `pyproject.toml` said 0.4.0. Both `.github/workflows/ci.yml:36`
(`uv sync --locked --extra cpu --extra dev`) and `Dockerfile:34` (`uv sync --locked --extra cpu`)
pass `--locked`, whose entire job is to fail rather than silently re-resolve.

**Confirmed against GitHub, not inferred** (`gh` is installed on this box now — the baton says it is
not; that fact is stale). The public Actions API says CI went red **exactly at the release commit**
and stayed red:

| run | sha | conclusion |
|---|---|---|
| 2026-08-01T21:09Z | `a052703` | **failure** |
| 2026-08-01T20:45Z | `47aabdd` | **failure** |
| 2026-08-01T14:10Z | `0cc2c3d` | success |

In both failed runs the failing step is **5, "Install dependencies"**, and steps 6–12 — ruff, ruff
format, mypy, pytest, bandit, pip-audit, detect-secrets — are all **`skipped`**. So **no lint, type,
test or security gate has run on `main` since the release**, and the v0.4.0 tag is CI-unverified.

**Reproduced and fixed on Linux, with the exact commands.** In the `~/pv-clean` clean-room tree left
warm by the 08-01 session, at `47aabdd` with the tag's shipped lockfile restored:
`uv sync --locked --extra cpu --extra dev` → **exit 1**, `error: The lockfile at uv.lock needs to be
updated, but --locked was provided`. After `uv lock`: the same command → **exit 0**, and the
Dockerfile's `uv sync --locked --extra cpu` → **exit 0**.

**Why five green local gate batteries missed it, which is the real lesson.** Nothing run locally
passes `--locked`: `just`/`uv run`/pre-commit all use the plain form, which re-resolves in silence.
**And the 08-01 clean-room run — the one ceremony designed to catch exactly this — used
`uv sync --extra cpu --extra dev`, not CI's `uv sync --locked …`.** It therefore *repaired* the
lockfile inside its own clone instead of failing on it. That is not a reconstruction: `~/pv-clean`
still carried an uncommitted `M uv.lock` whose whole diff is `-version = "0.3.0"` /
`+version = "0.4.0"`. **A clean-room check that does not run the shipped command validates a path
nobody ships.**

**2. `.dockerignore` excluded `README.md`, which the Dockerfile copies.** `pyproject.toml` declares
`readme = "README.md"`, so setuptools needs the file present to build this project's own metadata
during `uv sync`, and `Dockerfile:32` copies it for that reason. Docker matches `.dockerignore`
exactly and drops excluded paths from the build context, so the `COPY` fails before uv ever runs.
Both halves arrived together in the same unbuilt commit (`a052703` created `.dockerignore`; before
it the file was inert under the wrong name, so every earlier build had `README.md`). The exclusion
is now removed with the reason written at the line, so it is not re-added as tidy-up.

**Verified / not verified — stated separately, because they are not the same.** The lockfile fix is
verified end-to-end on Linux with both shipped commands. **The `.dockerignore` fix is reasoned, not
built** — the same caveat entry (6) carries for the rest of the Dockerfile, and it does not clear
yet. `docker compose build` is **still owed**.

**Docker Desktop 4.84.0 will not start on this box.** Installed per-user
(`%LOCALAPPDATA%\Programs\DockerDesktop`), CLI 29.6.2 / Compose v5.3.1. `docker version`, `docker
info` and `docker desktop status` all **hang on the named pipe** rather than erroring. Diagnosed:
WSL 2.7.11 is healthy and the `docker-desktop` distro boots by hand (kernel 6.18.33.2) but contains
**no `dockerd`** — only `/init`; **no Docker Windows service exists at all** (`Get-Service` and
`HKLM:\SYSTEM\CurrentControlSet\Services` both empty of docker entries); and `com.docker.backend` is
alive and answering, with the GUI polling `ErrorReportAPI GET /diagnostics/status` once a second —
the pattern of a startup-error screen waiting for a human. A clean kill-and-restart did not change
it. Needs eyes on the window; not fixable from a shell.

**Rejected.** Hand-editing the version line in `uv.lock` (ran `uv lock` instead — it is the
canonical producer, and it proves no dependency churn rode along). Installing Docker Engine natively
inside WSL Ubuntu to get a build (a sudo-level system change nobody asked for). Pre-emptively
"fixing" the Dockerfile further while it remains unbuildable — the whole point of this item is that
unbuilt Docker changes are how the repo got here.

**A third stale version, found by checking the rest of the class.** `apps/desktop/package-lock.json`
recorded `doc-assistant-desktop` at **0.1.0** against `package.json`'s 0.4.0 — stale since before
0.2.0, and unlike `uv.lock` **harmless**: there is no frontend job in CI at all, so nothing gates on
it, and npm reads the version from `package.json` regardless. Aligned anyway (both the root and
`packages[""]` fields, the two npm itself writes) so the release ritual has no exceptions to
remember.

**What it opens.** **A release bumps seven version strings, not five** — the checklist is missing
`uv.lock` and `package-lock.json`, and the first of those is the one that takes CI down. More
useful than the checklist: **`uv lock --check` is the cheap gate that would have caught this before
the tag** (it runs in ~1 s and needs no network), and it belongs either in the pre-commit battery or
at the release keypoint. Worth pairing with the wider lesson — a local battery that never runs the
*shipped* command can be green while `main` is red.
